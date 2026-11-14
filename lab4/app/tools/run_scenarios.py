"""Тестові сценарії застосунку (розділ 1.7): звичайні, складні й крайові випадки з автоматичною перевіркою.

Сценарії - у tools/data/scenarios.json: для кожного описано очікуваний результат і автоматичні перевірки
(кількість карток, обовʼязкові/заборонені слова, мова, verdict оцінювача). Промпти й параметри ті самі,
що у вебзастосунку (prompt_config.json).

Запуск:
    python3 tools/run_scenarios.py --model qwen3:4b-instruct-2507-q4_K_M
    python3 tools/run_scenarios.py --model ... --only S3 S10       # лише вибрані
    python3 tools/run_scenarios.py --model ... --label after-prompt-fix
Результат: reports/scenarios_<label>.md (таблиця + повні відповіді) і .json.

Сценарії інтерфейсу (сервер вимкнено, неправильна модель, довгий конспект, кнопка «Зупинити») перевіряються вручну
у браузері - див. README.md, розділ «Тестові сценарії».
"""

import argparse
import json
import re
from pathlib import Path

from common import REPORTS, Client, slug

SCENARIOS = Path(__file__).resolve().parent / "data" / "scenarios.json"


def cyrillic_share(text):
    letters = re.findall(r"[^\W\d_]", text)
    return sum("Ѐ" <= ch <= "ӿ" for ch in letters) / len(letters) if letters else 0.0


def check_generate(sc, r):
    problems = []
    if not r["json_ok"]:
        return [f"некоректний JSON: {r.get('json_error')}"]
    parsed = r["parsed"]
    cards = parsed["cards"]
    blob = " ".join(c["question"] + " " + c["answer"] for c in cards)
    low = blob.lower()
    ch = sc["checks"]
    if "min_cards" in ch and len(cards) < ch["min_cards"]:
        problems.append(f"карток {len(cards)} < {ch['min_cards']}")
    if "max_cards" in ch and len(cards) > ch["max_cards"]:
        problems.append(f"карток {len(cards)} > {ch['max_cards']}")
    if parsed.get("too_many"):
        problems.append("модель дала більше карток, ніж просили")
    if ch.get("note_not_empty") and not str(parsed.get("note", "")).strip():
        problems.append("порожнє поле note")
    if "must_contain_any" in ch and cards and not any(w.lower() in low for w in ch["must_contain_any"]):
        problems.append(f"немає жодного з {ch['must_contain_any']}")
    for w in ch.get("must_contain_all", []):
        if w.lower().replace(" ", "") not in low.replace(" ", "").replace(" ", ""):
            problems.append(f"немає «{w}»")
    for w in ch.get("must_not_contain", []):
        if w.lower() in low:
            problems.append(f"є заборонене «{w}»")
    if "cyrillic_min" in ch and cards and cyrillic_share(blob) < ch["cyrillic_min"]:
        problems.append(f"частка кирилиці {cyrillic_share(blob):.0%} < {ch['cyrillic_min']:.0%}")
    if r["finish_reason"] == "length":
        problems.append("відповідь обрізано лімітом токенів")
    return problems


def check_grade(sc, r):
    if not r["json_ok"]:
        return [f"некоректний JSON: {r.get('json_error')}"]
    verdict = r["parsed"]["verdict"]
    return [] if verdict in sc["checks"]["verdict"] else [f"verdict {verdict}, очікувався {'/'.join(sc['checks']['verdict'])}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="final")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--no-schema", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="ID")
    args = parser.parse_args()

    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in args.only]
    client = Client(args.base_url, args.model, args.reasoning_effort or None, not args.no_schema)

    rows, details = [], []
    for sc in scenarios:
        if sc["type"] == "generate":
            r = client.generate_cards(sc["notes"], sc["count"], sc["language"])
            problems = check_generate(sc, r)
            got = (f"{len(r['parsed']['cards'])} карток" + (f"; note: {r['parsed']['note']}" if r["parsed"]["note"] else "")
                   if r["json_ok"] else "некоректний JSON")
        else:
            r = client.grade(sc["question"], sc["reference"], sc["answer"])
            problems = check_grade(sc, r)
            got = f"{r['parsed']['verdict']}: {r['parsed'].get('feedback', '')}" if r["json_ok"] else "некоректний JSON"
        passed = not problems
        print(f"{sc['id']:<4} {'✅' if passed else '❌'} {sc['title']}" + ("" if passed else f" - {'; '.join(problems)}"))
        rows.append({"id": sc["id"], "title": sc["title"], "expected": sc["expected"], "got": got,
                     "passed": passed, "problems": problems, "tokens_per_s": r["tokens_per_s"],
                     "response": r["parsed"] if r["json_ok"] else r["text"]})

        details += [f"### {sc['id']}. {sc['title']}", "", f"**Очікується:** {sc['expected']}", ""]
        if sc["type"] == "generate":
            details += ["<details><summary>Вхідний конспект</summary>", "", sc["notes"], "", "</details>", ""]
            if r["json_ok"]:
                details += [f"{i}. **{c['question']}** - {c['answer']}" for i, c in enumerate(r["parsed"]["cards"], 1)]
                if r["parsed"]["note"]:
                    details += ["", f"_note:_ {r['parsed']['note']}"]
            else:
                details += ["```", r["text"][:2000], "```"]
        else:
            details += [f"Питання: {sc['question']}  ", f"Еталон: {sc['reference']}  ", f"Відповідь студента: «{sc['answer']}»", "",
                        f"Оцінка моделі: **{got}**"]
        details += ["", f"**Результат:** {'✅ пройдено' if passed else '❌ ' + '; '.join(problems)}", ""]

    passed = sum(r["passed"] for r in rows)
    md = [f"# Тестові сценарії - `{args.label}`", "", f"Модель: `{args.model}` · пройдено **{passed}/{len(rows)}**", "",
          "| # | Сценарій | Що вважається правильним | Що отримали | Впорався? |", "|---|---|---|---|---|"]
    for r in rows:
        got = r["got"].replace("|", "/").replace("\n", " ")[:160]
        md.append(f"| {r['id']} | {r['title']} | {r['expected']} | {got} | {'✅' if r['passed'] else '❌ ' + '; '.join(r['problems'])} |")
    md += ["", "## Деталі", ""] + details
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"scenarios_{slug(args.label)}"
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"model": args.model, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nПройдено {passed}/{len(rows)}. Записано {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
