"""Перевірка якості на 10 власних промптах (розділи 1.4 і 1.5 лаби).

Промпти - 10 конспектів з tools/data/quality_notes.json, типових для застосунку. Для кожного модель генерує
картки з тим самим системним промптом і параметрами, що й вебзастосунок (prompt_config.json). Зберігаються всі
відповіді й автоматичні показники: чи валідний JSON, скільки карток, яку частку ключових фактів покрито, TTFT, tok/s.

Запуск:
    python3 tools/quality_check.py --model qwen3:4b-instruct-2507-q4_K_M --label before
    python3 tools/quality_check.py --model qwen3:4b-instruct-2507-q8_0  --label after
    python3 tools/quality_check.py --compare before after      # таблиця порівняння → reports/quality_compare.md

Результати: reports/quality_<label>.json і reports/quality_<label>.md (повні відповіді - для ручної оцінки).
"""

import argparse
import json
from pathlib import Path

from common import REPORTS, Client, slug

NOTES_FILE = Path(__file__).resolve().parent / "data" / "quality_notes.json"


def norm(s):
    return "".join(s.lower().replace("'", "ʼ").replace("’", "ʼ").split())


def coverage(cards, key_facts):
    """Частка ключових фактів (підрядків), які зустрічаються в картках."""
    blob = norm(" ".join(c["question"] + " " + c["answer"] for c in cards))
    hits = [k for k in key_facts if norm(k) in blob]
    return len(hits) / len(key_facts), [k for k in key_facts if k not in hits]


def run(args):
    notes = json.loads(NOTES_FILE.read_text(encoding="utf-8"))
    client = Client(args.base_url, args.model, args.reasoning_effort, not args.no_schema)
    print(f"Прогрів {args.model}…")
    client.complete([{"role": "user", "content": "Привіт!"}], 5, 0)

    results = []
    for item in notes:
        r = client.generate_cards(item["notes"], item["count"], item["language"])
        cards = r["parsed"]["cards"] if r["parsed"] else []
        cov, missed = coverage(cards, item["key_facts"]) if cards else (0.0, item["key_facts"])
        row = {"id": item["id"], "title": item["title"], "requested": item["count"], "cards": len(cards),
               "json_ok": r["json_ok"], "coverage": round(cov, 2), "missed_facts": missed,
               "finish_reason": r["finish_reason"], "ttft_s": r["ttft_s"], "tokens": r["tokens"],
               "tokens_per_s": r["tokens_per_s"], "latency_s": r["latency_s"],
               "topic": r["parsed"]["topic"] if r["parsed"] else "", "parsed_cards": cards,
               "raw": None if r["json_ok"] else r["text"], "error": r.get("json_error")}
        results.append(row)
        print(f"  {item['id']:<22} JSON {'✓' if r['json_ok'] else '✗'}  карток {len(cards)}/{item['count']}  "
              f"покриття {cov:.0%}  {r['tokens_per_s'] or 0:.1f} tok/s")

    meta = {"label": args.label, "model": args.model, "base_url": args.base_url,
            "reasoning_effort": client.reasoning_effort, "structured_output": client.use_schema}
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"quality_{slug(args.label)}.json"
    out.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [f"# Якість на 10 промптах - `{args.label}`", "",
          f"Модель: `{args.model}` · {args.base_url} · structured output: {client.use_schema}", "",
          summary_table(results), ""]
    for row in results:
        md += [f"## {row['id']}: {row['title']}", "",
               f"JSON: {'✓' if row['json_ok'] else '✗ ' + str(row['error'])} · карток {row['cards']}/{row['requested']} · "
               f"покриття ключових фактів {row['coverage']:.0%}" + (f" (бракує: {', '.join(row['missed_facts'])})" if row["missed_facts"] else "")
               + f" · finish_reason {row['finish_reason']}", ""]
        md += [f"{i}. **{c['question']}** - {c['answer']} _({c['kind']})_" for i, c in enumerate(row["parsed_cards"], 1)]
        if row["raw"]:
            md += ["", "```", row["raw"][:2000], "```"]
        md += ["", "Ручна оцінка (вигадані факти? неточності?): …", ""]
    (REPORTS / f"quality_{slug(args.label)}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nЗаписано {out} і {out.with_suffix('.md')}")


def summary_table(results):
    n = len(results)
    ok = sum(r["json_ok"] for r in results)
    speeds = [r["tokens_per_s"] for r in results if r["tokens_per_s"]]
    ttfts = [r["ttft_s"] for r in results if r["ttft_s"]]
    return "\n".join([
        "| Показник | Значення |", "|---|---|",
        f"| Валідний JSON | {ok}/{n} |",
        f"| Карток (отримано/запитано) | {sum(r['cards'] for r in results)}/{sum(r['requested'] for r in results)} |",
        f"| Середнє покриття ключових фактів | {sum(r['coverage'] for r in results) / n:.0%} |",
        f"| Обрізано лімітом токенів | {sum(r['finish_reason'] == 'length' for r in results)} |",
        f"| Середній TTFT | {sum(ttfts) / len(ttfts):.2f} с |" if ttfts else "| Середній TTFT | - |",
        f"| Середня швидкість | {sum(speeds) / len(speeds):.1f} tok/s |" if speeds else "| Середня швидкість | - |",
    ])


def compare(labels):
    data = {}
    for label in labels:
        f = REPORTS / f"quality_{slug(label)}.json"
        if not f.exists():
            raise SystemExit(f"Немає {f} - спершу запустіть з --label {label}")
        data[label] = json.loads(f.read_text(encoding="utf-8"))
    md = ["# Порівняння якості: " + " vs ".join(f"`{l}`" for l in labels), ""]
    md += ["| | " + " | ".join(f"{l} (`{data[l]['meta']['model']}`)" for l in labels) + " |",
           "|---|" + "---|" * len(labels)]
    rows = {l: summary_table(data[l]["results"]).split("\n")[2:] for l in labels}
    for i, line in enumerate(rows[labels[0]]):
        name = line.split("|")[1].strip()
        md.append(f"| {name} | " + " | ".join(rows[l][i].split("|")[2].strip() for l in labels) + " |")
    md += ["", "### По промптах (покриття ключових фактів, карток)", "",
           "| Промпт | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for i, row in enumerate(data[labels[0]]["results"]):
        cells = []
        for l in labels:
            r = data[l]["results"][i]
            cells.append(f"{r['coverage']:.0%}, {r['cards']}/{r['requested']}" + ("" if r["json_ok"] else " ✗JSON"))
        md.append(f"| {row['title']} | " + " | ".join(cells) + " |")
    md += ["", "Повні відповіді для ручного порівняння: " + ", ".join(f"reports/quality_{slug(l)}.md" for l in labels)]
    text = "\n".join(md)
    print(text)
    (REPORTS / "quality_compare.md").write_text(text + "\n", encoding="utf-8")
    print(f"\nЗаписано {REPORTS / 'quality_compare.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model")
    parser.add_argument("--label", default="before")
    parser.add_argument("--reasoning-effort", default="none", help='"none" вимикає міркування; "" - як у моделі')
    parser.add_argument("--no-schema", action="store_true", help="не передавати JSON-схему (лише промпт)")
    parser.add_argument("--compare", nargs="+", metavar="LABEL", help="порівняти збережені запуски")
    args = parser.parse_args()
    args.reasoning_effort = args.reasoning_effort or None
    if args.compare:
        compare(args.compare)
    elif not args.model:
        parser.error("потрібен --model (або --compare)")
    else:
        run(args)


if __name__ == "__main__":
    main()
