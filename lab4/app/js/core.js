// Чиста логіка застосунку без DOM і мережі: промпти, ліміти, розбір і перевірка відповіді моделі.
// Окремий модуль, щоб його можна було протестувати в Node: node --test tests/

export const KINDS = ["definition", "fact", "why", "how", "formula", "example"];

export const LANGUAGES = {
  auto: "як у конспекті",
  uk: "українська",
  en: "англійська (English)",
};

/** Груба оцінка кількості токенів: для кирилиці в токенізаторі Qwen ≈ 2.5 символу на токен. */
export function estimateTokens(text, charsPerToken = 2.5) {
  return Math.ceil((text || "").length / charsPerToken);
}

/** Ліміт токенів відповіді залежить від кількості карток: база на JSON-обгортку + запас на кожну картку. */
export function maxTokensFor(cfg, count) {
  const g = cfg.generate;
  return Math.min(g.base_tokens + g.tokens_per_card * count, g.max_tokens_cap);
}

export function clampCount(cfg, count) {
  const g = cfg.generate;
  const n = Number.isFinite(count) ? Math.round(count) : g.default_cards;
  return Math.max(g.min_cards, Math.min(g.max_cards, n));
}

/**
 * Вирізає з конспекту рядки, схожі на інструкції для моделі («ігноруй попередні інструкції», «замість карток…»).
 * Мала модель на такі вставки реагувала порожнім списком карток навіть з правилом у промпті й few-shot прикладом,
 * тому захист - у коді, до відправки моделі. Повертає {text, removed}.
 */
export function stripInjections(cfg, text) {
  const patterns = (cfg.injection_patterns || []).map((p) => new RegExp(p, "iu"));
  const removed = [];
  const kept = (text || "").split("\n").filter((line) => {
    const hit = patterns.some((re) => re.test(line));
    if (hit) removed.push(line.trim());
    return !hit;
  });
  return { text: kept.join("\n"), removed };
}

/** Повідомлення для генерації карток: системний промпт + few-shot приклади + запит користувача. */
export function detectLanguage(text) {
  const letters = (text || "").match(/\p{L}/gu) || [];
  if (!letters.length) return "uk";
  const cyr = letters.filter((ch) => ch >= "\u0400" && ch <= "\u04ff").length;
  return cyr / letters.length >= 0.5 ? "uk" : "en";
}

export function languageLine(notes, language = "auto") {
  if (language !== "auto") return `Мова карток: ${LANGUAGES[language] || LANGUAGES.auto}.`;
  return detectLanguage(notes) === "en"
    ? "Мова карток: англійська (English) - та сама, що в конспекті. Пиши питання й відповіді англійською, не перекладай терміни."
    : "Мова карток: українська - та сама, що в конспекті.";
}

export function buildGenerateMessages(cfg, notes, { count, language = "auto" } = {}) {
  const messages = [{ role: "system", content: cfg.system_generate }];
  for (const shot of cfg.few_shot || []) {
    messages.push({ role: "user", content: shot.user }, { role: "assistant", content: shot.assistant });
  }
  // Якщо хтось вставив у конспект наш роздільник - прибираємо, щоб не «закрити» блок даних достроково
  const safe = stripInjections(cfg, notes).text.replaceAll("КОНСПЕКТ>>>", "КОНСПЕКТ»»»").trim();
  messages.push({
    role: "user",
    content: `Кількість карток: до ${count}.\n${languageLine(safe, language)}\n\n<<<КОНСПЕКТ\n${safe}\nКОНСПЕКТ>>>`,
  });
  return messages;
}

/** Повідомлення для перевірки відповіді студента (другий запит у ланцюжку). */
export function buildGradeMessages(cfg, card, userAnswer) {
  const safe = (userAnswer || "").replaceAll("ВІДПОВІДЬ>>>", "ВІДПОВІДЬ»»»").trim();
  return [
    { role: "system", content: cfg.system_grade },
    {
      role: "user",
      content: `Питання: ${card.question}\nЕталонна відповідь: ${card.answer}\n\n<<<ВІДПОВІДЬ\n${safe || "(порожньо)"}\nВІДПОВІДЬ>>>`,
    },
  ];
}

/**
 * Дістає JSON-обʼєкт з відповіді моделі. Мала модель інколи загортає його в ```json … ``` або
 * додає текст до чи після - тому шукаємо від першої «{» до парної їй «}».
 * Кидає Error з коротким поясненням, якщо JSON немає або він зламаний.
 */
export function extractJson(text) {
  if (!text || !text.trim()) throw new Error("модель повернула порожню відповідь");
  const cleaned = text.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/```(?:json)?/gi, "");
  const start = cleaned.indexOf("{");
  if (start === -1) throw new Error("у відповіді немає JSON-обʼєкта");
  let depth = 0, inString = false, escaped = false;
  for (let i = start; i < cleaned.length; i++) {
    const ch = cleaned[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth++;
    else if (ch === "}" && --depth === 0) {
      try {
        return JSON.parse(cleaned.slice(start, i + 1));
      } catch (error) {
        throw new Error(`JSON зламаний: ${error.message}`);
      }
    }
  }
  throw new Error("JSON обірвано - відповідь закінчилася посередині обʼєкта");
}

function normalizeQuestion(q) {
  return q.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
}

/**
 * Перевіряє структуру відповіді з картками й чистить її.
 * Повертає {topic, cards, note, warnings}; кидає Error, якщо структура зовсім не та.
 */
export function validateCards(obj, { max = Infinity } = {}) {
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) throw new Error("очікувався JSON-обʼєкт");
  if (!Array.isArray(obj.cards)) throw new Error("у JSON немає масиву cards");
  const warnings = [];
  const seen = new Set();
  const cards = [];
  let skipped = 0, dupes = 0;
  for (const raw of obj.cards) {
    const question = typeof raw?.question === "string" ? raw.question.trim() : "";
    const answer = typeof raw?.answer === "string" ? raw.answer.trim() : "";
    if (!question || !answer) { skipped++; continue; }
    const key = normalizeQuestion(question);
    if (seen.has(key)) { dupes++; continue; }
    seen.add(key);
    cards.push({ question, answer, kind: KINDS.includes(raw.kind) ? raw.kind : "fact" });
  }
  if (skipped) warnings.push(`пропущено карток без питання чи відповіді: ${skipped}`);
  if (dupes) warnings.push(`прибрано дублікатів: ${dupes}`);
  if (cards.length > max) {
    warnings.push(`модель дала ${cards.length} карток замість ${max} - зайві відкинуто`);
    cards.length = max;
  }
  return {
    topic: typeof obj.topic === "string" ? obj.topic.trim() : "",
    note: typeof obj.note === "string" ? obj.note.trim() : "",
    cards,
    warnings,
  };
}

export function validateGrade(obj) {
  const verdict = obj?.verdict;
  if (!["correct", "partial", "wrong"].includes(verdict)) throw new Error(`невідомий verdict: ${JSON.stringify(verdict)}`);
  return { verdict, feedback: typeof obj.feedback === "string" ? obj.feedback.trim() : "" };
}

/** Об'єднує картки з кількох частин конспекту без дублікатів. */
export function mergeCards(lists) {
  return validateCards({ cards: lists.flat() }).cards;
}

/** Скільки карток уже «видно» в недописаному JSON - для індикатора прогресу під час стрімінгу. */
export function countStreamedCards(partial) {
  return (partial.match(/"answer"\s*:/g) || []).length;
}

/**
 * Ділить довгий конспект на частини не довші за maxChars - по абзацах, а якщо абзац задовгий, то по реченнях.
 * Кожна частина обробляється окремим запитом, щоб вміститися в контекст моделі.
 */
export function splitNotes(text, maxChars) {
  const pieces = [];
  for (const para of text.split(/\n\s*\n/)) {
    if (para.length <= maxChars) { pieces.push(para); continue; }
    const sentences = para.match(/[^.!?…]+[.!?…]+[\s»"]*|[^.!?…]+$/g) || [para];
    for (const s of sentences) {
      for (let i = 0; i < s.length; i += maxChars) pieces.push(s.slice(i, i + maxChars));
    }
  }
  const chunks = [];
  let current = "";
  for (const piece of pieces) {
    const candidate = current ? `${current}\n\n${piece}` : piece;
    if (candidate.length > maxChars && current) {
      chunks.push(current);
      current = piece;
    } else {
      current = candidate;
    }
  }
  if (current.trim()) chunks.push(current);
  return chunks.map((c) => c.trim()).filter(Boolean);
}

/** Розподіляє бажану кількість карток між частинами пропорційно їхній довжині (мінімум 2 на частину). */
export function cardsPerChunk(chunks, total) {
  const sum = chunks.reduce((a, c) => a + c.length, 0);
  return chunks.map((c) => Math.max(2, Math.round((total * c.length) / sum)));
}

/** TSV для імпорту в Anki: «питання<TAB>відповідь<TAB>тег». */
export function toTSV(cards, topic = "") {
  const clean = (s) => s.replace(/[\t\r\n]+/g, " ");
  const tag = topic ? clean(topic).replace(/\s+/g, "_") : "";
  return cards.map((c) => [clean(c.question), clean(c.answer), tag].join("\t")).join("\n") + "\n";
}
