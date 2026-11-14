// Інтерфейс застосунку «Конспект → картки».
import { LlmError, listModels, streamChat } from "./api.js";
import {
  buildGenerateMessages, buildGradeMessages, cardsPerChunk, clampCount, countStreamedCards, estimateTokens,
  extractJson, maxTokensFor, mergeCards, splitNotes, stripInjections, toTSV, validateCards, validateGrade,
} from "./core.js";

const $ = (id) => document.getElementById(id);
const STORE = "notecards.v1";
const KIND_LABEL = { definition: "визначення", fact: "факт", why: "чому", how: "як", formula: "формула", example: "приклад" };
const VERDICT = {
  correct: ["✅ Правильно", "ok"],
  partial: ["🟡 Частково", "partial"],
  wrong: ["❌ Неправильно", "bad"],
};

const EXAMPLE = `Лекція 5. Локальний запуск моделей.
Квантизація - це зберігання ваг моделі меншою кількістю біт, наприклад 4 замість 16. Вона зменшує памʼять, яку займає модель, але може погіршити якість.
Памʼять під ваги рахують так: кількість параметрів × біт на вагу / 8. Модель на 4 млрд параметрів у fp16 займає близько 8 ГБ.
Крім ваг, памʼять займає KV-кеш: він зберігає ключі й значення уваги для всіх токенів контексту і росте лінійно з довжиною контексту.
Швидкість генерації обмежена пропускною здатністю памʼяті: на кожен токен треба прочитати всі ваги, тому стеля tokens/s ≈ пропускна здатність / розмір ваг.
MoE-моделі (Mixture of Experts) на кожен токен використовують лише частину параметрів, тому генерують швидше за щільну модель того ж розміру.`;

let cfg;
let state = loadState();
let abortController = null;
let practice = null;

init().catch((error) => showError($("error"), error));

async function init() {
  cfg = await (await fetch("prompt_config.json")).json();
  $("baseUrl").value = state.settings.baseUrl || cfg.default_base_url;
  $("model").value = state.settings.model || cfg.default_model;
  $("reasoningEffort").value = state.settings.reasoningEffort ?? "none";
  $("temperature").value = state.settings.temperature ?? cfg.generate.temperature;
  $("useSchema").value = state.settings.useSchema ?? "1";
  $("firstTimeout").value = state.settings.firstTimeout ?? cfg.timeouts_s.first_token;
  $("count").value = state.settings.count ?? cfg.generate.default_cards;
  $("count").min = cfg.generate.min_cards;
  $("count").max = cfg.generate.max_cards;
  $("notes").value = state.notes || "";

  for (const id of ["baseUrl", "model", "reasoningEffort", "temperature", "useSchema", "firstTimeout", "count"]) {
    $(id).addEventListener("change", () => { state.settings[id] = $(id).value; saveState(); updateHints(); });
  }
  $("baseUrl").addEventListener("change", checkServer);
  $("model").addEventListener("change", checkServer);
  $("notes").addEventListener("input", () => { state.notes = $("notes").value; saveState(); updateHints(); });
  $("file").addEventListener("change", loadFile);
  $("example").addEventListener("click", () => { $("notes").value = EXAMPLE; state.notes = EXAMPLE; saveState(); updateHints(); });
  $("generate").addEventListener("click", generate);
  $("stop").addEventListener("click", () => abortController?.abort());
  $("exportTsv").addEventListener("click", () => download(`${slug()}.tsv`, toTSV(state.cards, state.topic), "text/tab-separated-values"));
  $("exportJson").addEventListener("click", () => download(`${slug()}.json`, JSON.stringify({ topic: state.topic, cards: state.cards }, null, 2), "application/json"));
  $("clearDeck").addEventListener("click", () => { if (confirm("Видалити всі картки?")) { state.cards = []; state.topic = ""; saveState(); renderDeck(); } });
  $("practice").addEventListener("click", startPractice);
  $("check").addEventListener("click", gradeAnswer);
  $("reveal").addEventListener("click", () => revealAnswer());
  $("next").addEventListener("click", nextCard);
  $("endPractice").addEventListener("click", () => { $("practicePanel").hidden = true; practice = null; });
  $("pAnswer").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) gradeAnswer(); });

  updateHints();
  renderDeck();
  checkServer();
  setInterval(checkServer, 15000);
}

// ---------- стан ----------

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE) || "{}");
    return { settings: {}, cards: [], topic: "", notes: "", ...saved };
  } catch {
    return { settings: {}, cards: [], topic: "", notes: "" };
  }
}

function saveState() {
  try { localStorage.setItem(STORE, JSON.stringify(state)); } catch {}
}

function settings() {
  return {
    baseUrl: $("baseUrl").value.trim() || cfg.default_base_url,
    model: $("model").value.trim() || cfg.default_model,
    reasoningEffort: $("reasoningEffort").value || undefined,
    temperature: Number($("temperature").value || cfg.generate.temperature),
    useSchema: $("useSchema").value === "1",
    firstTokenTimeoutMs: Number($("firstTimeout").value || cfg.timeouts_s.first_token) * 1000,
    totalTimeoutMs: cfg.timeouts_s.total * 1000,
  };
}

// ---------- сервер ----------

async function checkServer() {
  const pill = $("status");
  const { baseUrl, model } = settings();
  try {
    const models = await listModels(baseUrl);
    $("models").innerHTML = models.map((m) => `<option value="${escapeHtml(m)}">`).join("");
    if (models.length && !models.includes(model)) {
      setPill(pill, "warn", `сервер онлайн, але моделі «${model}» немає - ollama pull ${model}`);
    } else {
      setPill(pill, "ok", `● ${model}`);
    }
  } catch (error) {
    setPill(pill, "bad", `● сервер недоступний (${baseUrl})`);
    pill.title = error.hint || error.message;
  }
}

function setPill(el, cls, text) {
  el.className = `pill ${cls}`;
  el.textContent = text;
  el.title = text;
}

// ---------- генерація карток ----------

function updateHints() {
  if (!cfg) return;
  const notes = $("notes").value;
  const count = clampCount(cfg, Number($("count").value));
  const tokens = estimateTokens(notes, cfg.chars_per_token);
  const parts = notes.length > cfg.max_note_chars ? splitNotes(notes, cfg.max_note_chars).length : 1;
  $("noteStats").textContent = notes
    ? `${notes.length} симв. ≈ ${tokens} токенів${parts > 1 ? ` · задовгий для одного запиту → ${parts} частини` : ""}`
    : "";
  $("paramsHint").textContent =
    `Генерація: temperature ${settings().temperature}, top_p ${cfg.generate.top_p}, ліміт ${maxTokensFor(cfg, count)} токенів ` +
    `(${cfg.generate.base_tokens} + ${cfg.generate.tokens_per_card} × ${count} карток). Перевірка відповіді: temperature ${cfg.grade.temperature}, ` +
    `ліміт ${cfg.grade.max_tokens} токенів. Контекст моделі має бути ≥ ${cfg.context_tokens} токенів - див. README.`;
}

async function loadFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  if (file.size > 2_000_000) return showError($("error"), new LlmError("input", "файл завеликий (більше 2 МБ) - це не схоже на конспект"));
  const text = await file.text();
  $("notes").value = text;
  state.notes = text;
  saveState();
  updateHints();
  event.target.value = "";
}

async function generate() {
  const notes = $("notes").value.trim();
  hide($("error"));
  if (notes.length < 20) {
    return showError($("error"), new LlmError("input", "конспект порожній або закороткий", "Вставте хоча б кілька речень навчального тексту."));
  }
  const s = settings();
  const count = clampCount(cfg, Number($("count").value));
  $("count").value = count;
  const chunks = notes.length > cfg.max_note_chars ? splitNotes(notes, cfg.max_note_chars) : [notes];
  const counts = chunks.length > 1 ? cardsPerChunk(chunks, count) : [count];

  abortController = new AbortController();
  setBusy(true);
  $("raw").textContent = "";
  $("rawBox").hidden = false;
  $("metrics").textContent = "";
  const lists = [], warnings = [], notesFromModel = [];
  const { removed } = stripInjections(cfg, notes);
  if (removed.length) warnings.push(`вилучено рядків, схожих на інструкції для моделі: ${removed.length} («${removed[0].slice(0, 60)}»)`);
  let topic = "";
  const metrics = [];

  try {
    for (let i = 0; i < chunks.length; i++) {
      const prefix = chunks.length > 1 ? `Частина ${i + 1}/${chunks.length}: ` : "";
      const started = performance.now();
      let phase = "wait";
      const tick = setInterval(() => {
        const sec = ((performance.now() - started) / 1000).toFixed(1);
        if (phase === "wait") progress(`${prefix}чекаю першого токена… ${sec} с (перший запит довший - модель завантажується в памʼять)`);
        if (phase === "think") progress(`${prefix}модель міркує… ${sec} с`);
      }, 200);
      let result;
      try {
        result = await streamChat({
          baseUrl: s.baseUrl,
          model: s.model,
          messages: buildGenerateMessages(cfg, chunks[i], { count: counts[i], language: $("language").value }),
          maxTokens: maxTokensFor(cfg, counts[i]),
          temperature: s.temperature,
          topP: cfg.generate.top_p,
          schema: s.useSchema ? cfg.schema_cards : undefined,
          schemaName: "flashcards",
          reasoningEffort: s.reasoningEffort,
          firstTokenTimeoutMs: s.firstTokenTimeoutMs,
          totalTimeoutMs: s.totalTimeoutMs,
          signal: abortController.signal,
        }, {
          onReasoning: () => { phase = "think"; },
          onFirstToken: () => { phase = "stream"; },
          onDelta: (_, full) => {
            $("raw").textContent = full;
            progress(`${prefix}генерую… карток: ${countStreamedCards(full)} з ~${counts[i]}`);
          },
        });
      } finally {
        clearInterval(tick);
      }
      metrics.push(result);

      if (result.finishReason === "length") {
        warnings.push(`${prefix}відповідь обрізано лімітом ${maxTokensFor(cfg, counts[i])} токенів - частину карток могло бути втрачено`);
      }
      let parsed;
      try {
        parsed = validateCards(extractJson(result.text), { max: counts[i] });
      } catch (error) {
        throw new LlmError("bad-json", `модель повернула некоректну відповідь: ${error.message}`,
          result.finishReason === "length"
            ? "Відповідь обрізано лімітом токенів - зменшіть кількість карток."
            : "Спробуйте ще раз, увімкніть «Structured output» або оберіть більшу модель. Сирий вивід - нижче.");
      }
      lists.push(parsed.cards);
      warnings.push(...parsed.warnings.map((w) => prefix + w));
      if (parsed.note) notesFromModel.push(prefix + parsed.note);
      topic ||= parsed.topic;
    }

    const cards = mergeCards(lists);
    if (!cards.length) {
      throw new LlmError("empty", "модель не знайшла в тексті матеріалу для карток",
        notesFromModel.join(" ") || "Перевірте, що це навчальний конспект.");
    }
    state.cards = cards;
    state.topic = topic;
    saveState();
    renderDeck(warnings.concat(notesFromModel.map((n) => `примітка моделі: ${n}`)));
    $("rawBox").open = false;
  } catch (error) {
    showError($("error"), error);
    if (error.kind === "bad-json") $("rawBox").open = true;
  } finally {
    setBusy(false);
    abortController = null;
    showMetrics(metrics);
  }
}

function showMetrics(list) {
  if (!list.length) return;
  const sum = (k) => list.reduce((a, m) => a + (m[k] || 0), 0);
  const tps = list.map((m) => m.tokensPerS).filter(Boolean);
  $("metrics").textContent =
    `TTFT ${list[0].ttft.toFixed(2)} с · ${tps.length ? (tps.reduce((a, b) => a + b, 0) / tps.length).toFixed(1) : "?"} tok/s · ` +
    `${sum("tokens")} токенів відповіді, ${sum("promptTokens")} токенів промпту · ${sum("latency").toFixed(1)} с · ` +
    `finish_reason: ${list.map((m) => m.finishReason || "?").join(", ")}`;
}

function setBusy(busy) {
  $("generate").disabled = busy;
  $("stop").hidden = !busy;
  $("progress").hidden = !busy;
}

function progress(text) {
  $("progressText").textContent = text;
}

// ---------- колода ----------

function renderDeck(warnings = []) {
  $("deckPanel").hidden = !state.cards.length;
  $("topic").textContent = state.topic ? `· ${state.topic} · ${state.cards.length}` : `· ${state.cards.length}`;
  const w = $("warnings");
  w.hidden = !warnings.length;
  w.innerHTML = warnings.map((x) => `⚠️ ${escapeHtml(x)}`).join("<br>");
  $("cards").innerHTML = state.cards.map((c, i) => `
    <li>
      <div class="q" contenteditable="true" data-i="${i}" data-f="question">${escapeHtml(c.question)}</div>
      <div class="a" contenteditable="true" data-i="${i}" data-f="answer">${escapeHtml(c.answer)}</div>
      <div class="meta"><span class="badge">${KIND_LABEL[c.kind] || c.kind}</span>
        <button class="link" data-del="${i}" title="видалити картку">видалити</button></div>
    </li>`).join("");
  for (const el of $("cards").querySelectorAll("[contenteditable]")) {
    el.addEventListener("blur", () => { state.cards[el.dataset.i][el.dataset.f] = el.textContent.trim(); saveState(); });
  }
  for (const el of $("cards").querySelectorAll("[data-del]")) {
    el.addEventListener("click", () => { state.cards.splice(Number(el.dataset.del), 1); saveState(); renderDeck(); });
  }
}

// ---------- тренування ----------

function startPractice() {
  if (!state.cards.length) return;
  const order = state.cards.map((_, i) => i).sort(() => Math.random() - 0.5);
  practice = { order, pos: 0, score: { correct: 0, partial: 0, wrong: 0 }, graded: false };
  $("practicePanel").hidden = false;
  showCard();
  $("practicePanel").scrollIntoView({ behavior: "smooth" });
}

function showCard() {
  const card = state.cards[practice.order[practice.pos]];
  $("practiceCounter").textContent = `· ${practice.pos + 1} / ${practice.order.length}`;
  $("pQuestion").textContent = card.question;
  $("pAnswer").value = "";
  practice.graded = false;
  hide($("verdict")); hide($("pReference")); hide($("pError"));
  $("score").textContent = `✅ ${practice.score.correct} · 🟡 ${practice.score.partial} · ❌ ${practice.score.wrong}`;
  $("pAnswer").focus();
}

function revealAnswer() {
  const card = state.cards[practice.order[practice.pos]];
  $("pReference").innerHTML = `<b>Еталон:</b> ${escapeHtml(card.answer)}`;
  $("pReference").hidden = false;
}

function nextCard() {
  if (!practice) return;
  if (practice.pos + 1 >= practice.order.length) {
    const s = practice.score;
    $("pQuestion").textContent = `Готово! Правильно ${s.correct}, частково ${s.partial}, неправильно ${s.wrong} з ${practice.order.length}.`;
    $("practiceCounter").textContent = "";
    hide($("verdict")); hide($("pReference"));
    return;
  }
  practice.pos++;
  showCard();
}

async function gradeAnswer() {
  if (!practice || practice.pos >= practice.order.length) return;
  const card = state.cards[practice.order[practice.pos]];
  const answer = $("pAnswer").value.trim();
  hide($("pError"));
  if (!answer) return showError($("pError"), new LlmError("input", "спершу напишіть відповідь", "Або натисніть «Показати відповідь»."));
  const s = settings();
  $("gradeProgress").hidden = false;
  $("check").disabled = true;
  try {
    const result = await streamChat({
      baseUrl: s.baseUrl,
      model: s.model,
      messages: buildGradeMessages(cfg, card, answer),
      maxTokens: cfg.grade.max_tokens,
      temperature: cfg.grade.temperature,
      schema: s.useSchema ? cfg.schema_grade : undefined,
      schemaName: "grade",
      reasoningEffort: s.reasoningEffort,
      firstTokenTimeoutMs: s.firstTokenTimeoutMs,
      totalTimeoutMs: 60000,
    });
    let grade;
    try {
      grade = validateGrade(extractJson(result.text));
    } catch (error) {
      throw new LlmError("bad-json", `модель повернула некоректну оцінку: ${error.message}`, `Сирий вивід: ${result.text.slice(0, 200)}`);
    }
    const [label, cls] = VERDICT[grade.verdict];
    $("verdict").className = `verdict ${cls}`;
    $("verdict").innerHTML = `<b>${label}</b> ${escapeHtml(grade.feedback)}`;
    $("verdict").hidden = false;
    if (!practice.graded) { practice.score[grade.verdict]++; practice.graded = true; }
    $("score").textContent = `✅ ${practice.score.correct} · 🟡 ${practice.score.partial} · ❌ ${practice.score.wrong}`;
    revealAnswer();
  } catch (error) {
    showError($("pError"), error);
  } finally {
    $("gradeProgress").hidden = true;
    $("check").disabled = false;
  }
}

// ---------- допоміжне ----------

function showError(el, error) {
  const message = error instanceof LlmError ? error.message : `неочікувана помилка: ${error.message || error}`;
  const hint = error.hint ? `<div class="hint">${escapeHtml(error.hint)}</div>` : "";
  el.innerHTML = `<b>${error.kind === "aborted" ? "⏹" : "⚠️"} ${escapeHtml(message)}</b>${hint}`;
  el.hidden = false;
}

function hide(el) { el.hidden = true; }

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function slug() {
  return (state.topic || "cards").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "_").replace(/^_|_$/g, "") || "cards";
}

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
