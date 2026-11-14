"""Спільний код для quality_check.py і run_scenarios.py: ті самі промпти й параметри, що й у вебзастосунку
(читаються з prompt_config.json), стрімінговий запит з вимірюваннями й розбір JSON-відповіді."""

import json
import re
import time
from pathlib import Path

from openai import BadRequestError, OpenAI

APP_DIR = Path(__file__).resolve().parent.parent
REPORTS = APP_DIR / "reports"
CFG = json.loads((APP_DIR / "prompt_config.json").read_text(encoding="utf-8"))
LANGUAGES = {"auto": "як у конспекті", "uk": "українська", "en": "англійська (English)"}
KINDS = {"definition", "fact", "why", "how", "formula", "example"}


def max_tokens_for(count):
    g = CFG["generate"]
    return min(g["base_tokens"] + g["tokens_per_card"] * count, g["max_tokens_cap"])


def detect_language(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "uk"
    return "uk" if sum("\u0400" <= ch <= "\u04ff" for ch in letters) / len(letters) >= 0.5 else "en"


def language_line(notes, language="auto"):
    if language != "auto":
        return f"Мова карток: {LANGUAGES[language]}."
    if detect_language(notes) == "en":
        return ("Мова карток: англійська (English) - та сама, що в конспекті. "
                "Пиши питання й відповіді англійською, не перекладай терміни.")
    return "Мова карток: українська - та сама, що в конспекті."


def strip_injections(text):
    """Вирізає рядки, схожі на інструкції для моделі (як stripInjections у js/core.js)."""
    patterns = [re.compile(p, re.I) for p in CFG.get("injection_patterns", [])]
    kept, removed = [], []
    for line in text.split("\n"):
        (removed if any(p.search(line) for p in patterns) else kept).append(line)
    return "\n".join(kept), [r.strip() for r in removed]


def generate_messages(notes, count, language="auto"):
    messages = [{"role": "system", "content": CFG["system_generate"]}]
    for shot in CFG["few_shot"]:
        messages += [{"role": "user", "content": shot["user"]}, {"role": "assistant", "content": shot["assistant"]}]
    safe = strip_injections(notes)[0].replace("КОНСПЕКТ>>>", "КОНСПЕКТ»»»").strip()
    messages.append({"role": "user", "content": f"Кількість карток: до {count}.\n{language_line(safe, language)}\n\n"
                                                f"<<<КОНСПЕКТ\n{safe}\nКОНСПЕКТ>>>"})
    return messages


def grade_messages(question, reference, answer):
    safe = (answer or "").replace("ВІДПОВІДЬ>>>", "ВІДПОВІДЬ»»»").strip() or "(порожньо)"
    return [
        {"role": "system", "content": CFG["system_grade"]},
        {"role": "user", "content": f"Питання: {question}\nЕталонна відповідь: {reference}\n\n<<<ВІДПОВІДЬ\n{safe}\nВІДПОВІДЬ>>>"},
    ]


def extract_json(text):
    """Перший JSON-обʼєкт у тексті (модель інколи додає ```json або пояснення)."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).replace("```json", "").replace("```", "")
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("у відповіді немає JSON-обʼєкта")
    obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    return obj


def validate_cards(obj, max_cards):
    if not isinstance(obj, dict) or not isinstance(obj.get("cards"), list):
        raise ValueError("немає масиву cards")
    seen, cards = set(), []
    for c in obj["cards"]:
        q, a = str(c.get("question", "")).strip(), str(c.get("answer", "")).strip()
        key = re.sub(r"\W+", " ", q.lower()).strip()
        if q and a and key not in seen:
            seen.add(key)
            cards.append({"question": q, "answer": a, "kind": c.get("kind") if c.get("kind") in KINDS else "fact"})
    return {"topic": obj.get("topic", ""), "note": obj.get("note", ""), "cards": cards[:max_cards],
            "too_many": len(cards) > max_cards}


class Client:
    def __init__(self, base_url, model, reasoning_effort=None, use_schema=True, timeout=240):
        self.client = OpenAI(base_url=base_url, api_key="local", timeout=timeout)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.use_schema = use_schema

    def complete(self, messages, max_tokens, temperature, top_p=None, schema=None, schema_name="answer"):
        """Стрімінговий запит; повертає текст і метрики (як у вебзастосунку)."""
        extra = {}
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        if schema is not None and self.use_schema:
            extra["response_format"] = {"type": "json_schema",
                                        "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
        if top_p is not None:
            extra["top_p"] = top_p
        start = time.perf_counter()
        first = usage = finish = None
        text = ""
        try:
            stream = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                stream=True, stream_options={"include_usage": True}, **extra,
            )
        except BadRequestError as error:
            # модель без режиму міркування або рантайм без structured outputs - повторюємо без цих параметрів
            if "reasoning_effort" in extra and re.search(r"think|reason", str(error), re.I):
                self.reasoning_effort = None
            elif "response_format" in extra:
                self.use_schema = False
            else:
                raise
            return self.complete(messages, max_tokens, temperature, top_p, schema, schema_name)
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish = choice.finish_reason or finish
            if choice.delta.content:
                if first is None:
                    first = time.perf_counter()
                text += choice.delta.content
        end = time.perf_counter()
        tokens = usage.completion_tokens if usage else None
        return {
            "text": text,
            "finish_reason": finish,
            "tokens": tokens,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "ttft_s": round(first - start, 3) if first else None,
            "latency_s": round(end - start, 3),
            "tokens_per_s": round((tokens - 1) / (end - first), 1) if tokens and first and end > first else None,
        }

    def generate_cards(self, notes, count, language="auto", temperature=None):
        g = CFG["generate"]
        r = self.complete(generate_messages(notes, count, language), max_tokens_for(count),
                          g["temperature"] if temperature is None else temperature, g["top_p"],
                          CFG["schema_cards"], "flashcards")
        try:
            r["parsed"] = validate_cards(extract_json(r["text"]), count)
            r["json_ok"] = True
        except (ValueError, json.JSONDecodeError) as error:
            r["parsed"], r["json_ok"], r["json_error"] = None, False, str(error)
        return r

    def grade(self, question, reference, answer):
        g = CFG["grade"]
        r = self.complete(grade_messages(question, reference, answer), g["max_tokens"], g["temperature"],
                          schema=CFG["schema_grade"], schema_name="grade")
        try:
            obj = extract_json(r["text"])
            if obj.get("verdict") not in ("correct", "partial", "wrong"):
                raise ValueError(f"невідомий verdict {obj.get('verdict')!r}")
            r["parsed"], r["json_ok"] = obj, True
        except (ValueError, json.JSONDecodeError) as error:
            r["parsed"], r["json_ok"], r["json_error"] = None, False, str(error)
        return r


def slug(text):
    return re.sub(r"[^\w.-]+", "_", text).strip("_")
