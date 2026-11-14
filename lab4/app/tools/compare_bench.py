"""Порівнює запуски bench.py з різними мітками (before / after) з lab4/bench_results.csv.

Запуск:
    python3 tools/compare_bench.py                       # усі мітки з файлу
    python3 tools/compare_bench.py --labels before after
    python3 tools/compare_bench.py --predicted 85        # додати прогноз стелі tok/s з budget.py для порівняння
Результат друкується й записується в reports/bench_compare.md. Якщо для мітки є кілька запусків,
береться останній.
"""

import argparse
import csv
from pathlib import Path

LAB4 = Path(__file__).resolve().parents[2]
REPORTS = LAB4 / "app" / "reports"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(LAB4 / "bench_results.csv"))
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--predicted", type=float, help="прогноз tok/s з розрахунку 1.2")
    args = parser.parse_args()

    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("bench_results.csv порожній")

    # останній запуск для кожної пари (мітка, модель, кількість клієнтів)
    latest = {}
    for row in rows:
        latest[(row["label"], row["model"], int(row["concurrency"]))] = row
    labels = args.labels or list(dict.fromkeys(r["label"] for r in rows))
    runs = {}
    for (label, model, conc), row in latest.items():
        if label in labels:
            runs.setdefault((label, model), {})[conc] = row

    def num(row, key):
        try:
            return float(row[key])
        except (TypeError, ValueError, KeyError):
            return None

    md = ["## Вимірювання bench.py", ""]
    md += ["| Мітка | Модель | Клієнтів | Помилок | TTFT p50, с | TTFT p95, с | tok/s на користувача | Throughput, tok/s |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (label, model), by_conc in runs.items():
        for conc in sorted(by_conc):
            r = by_conc[conc]
            md.append(f"| {label} | `{model}` | {conc} | {r['errors']} | {r['ttft_p50_s']} | {r['ttft_p95_s']} | "
                      f"{r['tokens_per_s_per_user_p50']} | {r['throughput_tokens_per_s']} |")

    keys = list(runs)
    if len(keys) >= 2:
        (l0, m0), (l1, m1) = keys[0], keys[-1]
        md += ["", f"### Зміна `{l1}` відносно `{l0}`", "",
               "| Клієнтів | tok/s на користувача | Throughput | TTFT p50 |", "|---:|---:|---:|---:|"]
        for conc in sorted(set(runs[keys[0]]) & set(runs[keys[-1]])):
            a, b = runs[keys[0]][conc], runs[keys[-1]][conc]

            def delta(key, lower_is_better=False):
                x, y = num(a, key), num(b, key)
                if not x or y is None:
                    return "-"
                change = (y - x) / x * 100
                good = change < 0 if lower_is_better else change > 0
                return f"{x:g} → {y:g} ({change:+.0f} % {'✅' if good else '🔻'})"

            md.append(f"| {conc} | {delta('tokens_per_s_per_user_p50')} | {delta('throughput_tokens_per_s')} | "
                      f"{delta('ttft_p50_s', lower_is_better=True)} |")

    if args.predicted:
        md += ["", "### Прогноз проти вимірювання (1 клієнт)", "", "| Мітка | Прогноз стелі | Виміряно | Частка від стелі |",
               "|---|---:|---:|---:|"]
        for (label, model), by_conc in runs.items():
            if 1 in by_conc and num(by_conc[1], "tokens_per_s_per_user_p50"):
                got = num(by_conc[1], "tokens_per_s_per_user_p50")
                md.append(f"| {label} | {args.predicted:g} | {got:g} | {got / args.predicted:.0%} |")

    text = "\n".join(md)
    print(text)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "bench_compare.md").write_text(text + "\n", encoding="utf-8")
    print(f"\nЗаписано {REPORTS / 'bench_compare.md'}")


if __name__ == "__main__":
    main()
