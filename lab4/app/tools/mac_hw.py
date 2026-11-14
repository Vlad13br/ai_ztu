"""Збирає характеристики Mac для розділу 1.1 лаби й записує таблицю з джерелами кожного значення.

Запуск (на своєму Mac):
    python3 tools/mac_hw.py                 # друкує таблицю й пише reports/hardware.md і reports/hardware.json
    python3 tools/mac_hw.py --bandwidth 120 # якщо чип не розпізнано - пропускну здатність задайте вручну, ГБ/с

Нічого не встановлює, працює без сторонніх пакетів. Значення беруться з:
  - sysctl (процесор, ядра, RAM, ліміт памʼяті GPU),
  - system_profiler (модель Mac, чип, кількість ядер GPU),
  - логу Ollama (recommendedMaxWorkingSetSize - скільки памʼяті Metal реально дає GPU),
  - таблиці пропускної здатності памʼяті зі специфікацій Apple (apple.com → Tech Specs) - її треба звірити з вашою моделлю.
"""

import argparse
import json
import platform
import re
import subprocess
from pathlib import Path

REPORTS = Path(__file__).resolve().parent.parent / "reports"

# Пропускна здатність уніфікованої памʼяті, ГБ/с - зі сторінок Tech Specs на apple.com.
# Для M3 Max і M4 Max вона залежить від конфігурації (кількості ядер GPU), тому ключ - (чип, мін. ядер GPU).
BANDWIDTH_GBPS = {
    "Apple M1": 68.25, "Apple M1 Pro": 200, "Apple M1 Max": 400, "Apple M1 Ultra": 800,
    "Apple M2": 100, "Apple M2 Pro": 200, "Apple M2 Max": 400, "Apple M2 Ultra": 800,
    "Apple M3": 100, "Apple M3 Pro": 150, "Apple M3 Ultra": 819,
    "Apple M4": 120, "Apple M4 Pro": 273,
    "Apple M5": 153,
}
BANDWIDTH_BY_GPU = {
    "Apple M3 Max": [(40, 400), (0, 300)],  # 40-core GPU → 400 ГБ/с, 30-core → 300 ГБ/с
    "Apple M4 Max": [(40, 546), (0, 410)],  # 40-core GPU → 546 ГБ/с, 32-core → 410 ГБ/с
}


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def sysctl(name):
    return run(["sysctl", "-n", name])


def ollama_working_set_mb():
    """Ollama (llama.cpp/Metal) пише в лог, скільки памʼяті GPU дозволяє використати macOS."""
    log = Path.home() / ".ollama" / "logs" / "server.log"
    if not log.exists():
        return None
    matches = re.findall(r"recommendedMaxWorkingSetSize\s*=\s*([\d.]+)\s*MB", log.read_text(errors="ignore"))
    return float(matches[-1]) if matches else None


def detect(bandwidth_override=None):
    rows = []  # (параметр, значення, джерело)
    info = {}

    if platform.system() != "Darwin":
        raise SystemExit("Цей скрипт для macOS. На іншій ОС заповніть таблицю вручну.")

    model_name = run(["sysctl", "-n", "hw.model"])
    hw = {}
    try:
        hw = json.loads(run(["system_profiler", "SPHardwareDataType", "-json"]))["SPHardwareDataType"][0]
    except (ValueError, KeyError, IndexError):
        pass
    gpu = {}
    try:
        gpu = json.loads(run(["system_profiler", "SPDisplaysDataType", "-json"]))["SPDisplaysDataType"][0]
    except (ValueError, KeyError, IndexError):
        pass

    mac = hw.get("machine_name", "Mac") + (f" ({model_name})" if model_name else "")
    rows.append(("Компʼютер", mac, "system_profiler SPHardwareDataType, sysctl hw.model"))
    rows.append(("macOS", f"{run(['sw_vers', '-productVersion'])} ({platform.machine()})", "sw_vers, uname -m"))

    chip = sysctl("machdep.cpu.brand_string") or hw.get("chip_type", "?")
    info["chip"] = chip
    p_cores, e_cores = sysctl("hw.perflevel0.physicalcpu"), sysctl("hw.perflevel1.physicalcpu")
    cores = sysctl("hw.physicalcpu")
    core_text = f"{cores} ядер" + (f" ({p_cores} продуктивних + {e_cores} енергоефективних)" if p_cores and e_cores else "")
    rows.append(("Процесор", f"{chip}, {core_text}", "sysctl machdep.cpu.brand_string, hw.physicalcpu, hw.perflevel*"))

    ram_gb = int(sysctl("hw.memsize") or 0) / 2**30
    info["ram_gb"] = round(ram_gb, 1)
    rows.append(("Оперативна памʼять (RAM)", f"{ram_gb:.0f} ГБ уніфікованої памʼяті (спільна для CPU і GPU)", "sysctl hw.memsize"))

    gpu_cores = gpu.get("sppci_cores")
    info["gpu_cores"] = int(gpu_cores) if str(gpu_cores).isdigit() else None
    rows.append(("Відеокарта", f"{gpu.get('sppci_model', chip)} (вбудована в чип), ядер GPU: {gpu_cores or '?'}",
                 "system_profiler SPDisplaysDataType"))

    # Скільки з RAM може взяти GPU
    wired_limit = int(sysctl("iogpu.wired_limit_mb") or 0)
    working_set = ollama_working_set_mb()
    if working_set:
        vram_gb, src = working_set / 1024, "лог Ollama ~/.ollama/logs/server.log → recommendedMaxWorkingSetSize"
    elif wired_limit:
        vram_gb, src = wired_limit / 1024, "sysctl iogpu.wired_limit_mb (задано вручну)"
    else:
        share = 0.75 if ram_gb > 36 else 0.67
        vram_gb, src = ram_gb * share, (f"оцінка: macOS за замовчуванням дає GPU ≈{share:.0%} RAM; точне число зʼявиться "
                                        "в логі Ollama після першого запуску моделі - перезапустіть скрипт")
    info["gpu_budget_gb"] = round(vram_gb, 1)
    rows.append(("«VRAM» (скільки RAM доступно GPU)", f"≈ {vram_gb:.1f} ГБ", src))

    # Пропускна здатність
    if bandwidth_override:
        bw, bw_src = bandwidth_override, "задано вручну (--bandwidth)"
    elif chip in BANDWIDTH_BY_GPU and info["gpu_cores"]:
        bw = next(v for min_cores, v in BANDWIDTH_BY_GPU[chip] if info["gpu_cores"] >= min_cores)
        bw_src = f"специфікація Apple для {chip} з {info['gpu_cores']}-core GPU (apple.com → Tech Specs)"
    elif chip in BANDWIDTH_GBPS:
        bw, bw_src = BANDWIDTH_GBPS[chip], f"специфікація Apple для {chip} (apple.com → Tech Specs)"
    else:
        bw, bw_src = None, "чип не розпізнано - знайдіть «memory bandwidth» у Tech Specs і запустіть з --bandwidth"
    info["bandwidth_gbps"] = bw
    rows.append(("Пропускна здатність памʼяті", f"{bw} ГБ/с" if bw else "?", bw_src))
    return rows, info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bandwidth", type=float, help="пропускна здатність памʼяті вручну, ГБ/с")
    args = parser.parse_args()

    rows, info = detect(args.bandwidth)
    table = ["| Параметр | Значення | Звідки взято |", "|---|---|---|"]
    table += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    md = "\n".join(table)
    print(md)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "hardware.md").write_text("## 1.1. Залізо\n\n" + md + "\n", encoding="utf-8")
    (REPORTS / "hardware.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nЗаписано {REPORTS / 'hardware.md'} і hardware.json (його читає tools/budget.py)")


if __name__ == "__main__":
    main()
