"""Розрахунок «бюджету заліза» (розділ 1.2): які моделі й квантизації вмістяться в памʼять і яка стеля tokens/s.

Формули (лекція 5, блок «Бюджет заліза»):
    ваги, ГБ      = параметри × біт_на_вагу / 8
    KV-кеш, ГБ    = 2 (K і V) × шарів × KV-голів × розмір_голови × контекст × байт_на_елемент × паралельних_слотів
    усього, ГБ    = ваги + KV-кеш + запас на рантайм (буфери обчислень, ~1 ГБ)
    стеля tok/s   = пропускна_здатність / байт, прочитаних на один токен (≈ активні ваги + KV-кеш)
Для MoE на кожен токен читаються лише активні параметри, тому стеля швидкості рахується за ними,
а памʼять - за всіма параметрами. Реальна швидкість зазвичай 50–80 % від стелі.

Запуск:
    python3 tools/budget.py                          # бере RAM і пропускну здатність з reports/hardware.json (tools/mac_hw.py)
    python3 tools/budget.py --ram 16 --gpu-budget 10.7 --bandwidth 120 --ctx 8192
    python3 tools/budget.py --ctx 8192 --parallel 4 --kv-type q8_0   # з паралельними слотами й квантизованим KV-кешем
Результат - reports/budget.md.
"""

import argparse
import json
from pathlib import Path

REPORTS = Path(__file__).resolve().parent.parent / "reports"

# Архітектура з config.json моделей на Hugging Face (Qwen/Qwen3-*)
MODELS = {
    #  назва              (параметри, млрд; активні, млрд; шарів; KV-голів; розмір голови)
    "Qwen3-1.7B":        (2.03, 2.03, 28, 8, 128),
    "Qwen3-4B":          (4.02, 4.02, 36, 8, 128),
    "Qwen3-8B":          (8.19, 8.19, 36, 8, 128),
    "Qwen3-14B":         (14.8, 14.8, 40, 8, 128),
    "Qwen3-30B-A3B (MoE)": (30.5, 3.3, 48, 4, 128),
    "Qwen3-32B":         (32.8, 32.8, 64, 8, 128),
}

# Середня кількість біт на вагу у форматах GGUF (з урахуванням масштабів блоків)
QUANTS = {"Q4_K_M": 4.85, "Q5_K_M": 5.69, "Q6_K": 6.56, "Q8_0": 8.5, "F16": 16.0}
KV_BYTES = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
RUNTIME_OVERHEAD_GB = 1.0


def weights_gb(params_b, bits):
    return params_b * 1e9 * bits / 8 / 1e9


def kv_gb(layers, kv_heads, head_dim, ctx, kv_bytes, parallel=1):
    return 2 * layers * kv_heads * head_dim * ctx * kv_bytes * parallel / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ram", type=float, help="RAM, ГБ")
    parser.add_argument("--gpu-budget", type=float, help="скільки памʼяті доступно GPU, ГБ (на Mac ≈ 67–75 %% RAM)")
    parser.add_argument("--bandwidth", type=float, help="пропускна здатність памʼяті, ГБ/с")
    parser.add_argument("--ctx", type=int, default=8192, help="потрібна довжина контексту, токенів")
    parser.add_argument("--parallel", type=int, default=1, help="паралельних слотів (OLLAMA_NUM_PARALLEL)")
    parser.add_argument("--kv-type", choices=list(KV_BYTES), default="f16", help="тип KV-кешу (OLLAMA_KV_CACHE_TYPE)")
    parser.add_argument("--reserve", type=float, default=4.0,
                        help="скільки RAM лишити macOS і відкритим програмам, ГБ (браузер, IDE)")
    args = parser.parse_args()

    hw = {}
    hw_file = REPORTS / "hardware.json"
    if hw_file.exists():
        hw = json.loads(hw_file.read_text(encoding="utf-8"))
    ram = args.ram or hw.get("ram_gb")
    gpu_budget = args.gpu_budget or hw.get("gpu_budget_gb") or (ram * 0.67 if ram else None)
    bandwidth = args.bandwidth or hw.get("bandwidth_gbps")
    if not (ram and bandwidth):
        raise SystemExit("Невідомі RAM чи пропускна здатність: спершу запустіть tools/mac_hw.py або задайте --ram і --bandwidth.")
    budget = min(gpu_budget, ram - args.reserve)

    lines = [
        "## 1.2. Розрахунок",
        "",
        f"Вхідні дані: RAM {ram:.0f} ГБ, доступно GPU ≈ {gpu_budget:.1f} ГБ, лишаємо системі {args.reserve:.0f} ГБ → "
        f"**бюджет на модель {budget:.1f} ГБ**; пропускна здатність **{bandwidth:g} ГБ/с**; "
        f"контекст {args.ctx} токенів × {args.parallel} слот(ів), KV-кеш {args.kv_type}; запас на рантайм {RUNTIME_OVERHEAD_GB} ГБ.",
        "",
        "Стеля tok/s - для короткого контексту (як у bench.py: читаються лише ваги); у дужках - коли контекст заповнений "
        "повністю (на кожен токен додатково читається весь KV-кеш одного слота).",
        "",
        "| Модель | Квант. | Ваги, ГБ | KV-кеш, ГБ | Разом, ГБ | Вміщається? | Стеля tok/s | Реалістично (×0.6–0.8) |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    fits = []
    for name, (params, active, layers, kv_heads, head_dim) in MODELS.items():
        kv = kv_gb(layers, kv_heads, head_dim, args.ctx, KV_BYTES[args.kv_type], args.parallel)
        for quant, bits in QUANTS.items():
            w = weights_gb(params, bits)
            total = w + kv + RUNTIME_OVERHEAD_GB
            ok = total <= budget
            # на кожен токен читаємо всі активні ваги; при повному контексті - ще й KV-кеш одного слота
            ceiling = bandwidth / weights_gb(active, bits)
            ceiling_full = bandwidth / (weights_gb(active, bits) + kv / args.parallel)
            mark = "✅" if ok else ("⚠️ впритул" if total <= budget * 1.1 else "❌")
            lines.append(f"| {name} | {quant} | {w:.1f} | {kv:.2f} | {total:.1f} | {mark} | {ceiling:.0f} ({ceiling_full:.0f}) | "
                         f"{ceiling * 0.6:.0f}–{ceiling * 0.8:.0f} |")
            if ok:
                fits.append((params, name, quant, total, ceiling))
    lines += ["", "Приклад для Qwen3-4B Q4_K_M (36 шарів, 8 KV-голів, розмір голови 128):", ""]
    w4 = weights_gb(4.02, QUANTS["Q4_K_M"])
    kv4 = kv_gb(36, 8, 128, args.ctx, KV_BYTES[args.kv_type], args.parallel)
    lines += [
        f"- ваги: 4.02·10⁹ × 4.85 / 8 = **{w4:.2f} ГБ**",
        f"- KV-кеш: 2 × 36 × 8 × 128 × {args.ctx} × {KV_BYTES[args.kv_type]} Б × {args.parallel} = **{kv4:.2f} ГБ**",
        f"- разом із запасом {RUNTIME_OVERHEAD_GB} ГБ: **{w4 + kv4 + RUNTIME_OVERHEAD_GB:.2f} ГБ** з {budget:.1f} ГБ бюджету",
        f"- стеля швидкості: {bandwidth:g} ГБ/с / {w4:.2f} ГБ на токен ≈ **{bandwidth / w4:.0f} tok/s** "
        f"(з повністю заповненим контекстом: {bandwidth:g} / {w4 + kv4 / args.parallel:.2f} ≈ {bandwidth / (w4 + kv4 / args.parallel):.0f} tok/s)",
    ]
    if fits:
        top = max(f[0] for f in fits)
        options = ", ".join(f"{q} ({t:.1f} ГБ, ≤{c:.0f} tok/s)" for p, n, q, t, c in fits if p == top)
        name = next(n for p, n, *_ in fits if p == top)
        lines += ["", f"Найбільша модель, що вміщається: **{name}** - квантизації: {options}.",
                  "Обирайте компроміс: більша модель/квантизація - краща якість, але повільніше (швидкість ∝ 1 / розмір ваг)."]

    md = "\n".join(lines)
    print(md)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "budget.md").write_text(md + "\n", encoding="utf-8")
    print(f"\nЗаписано {REPORTS / 'budget.md'}")


if __name__ == "__main__":
    main()
