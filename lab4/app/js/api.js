// Клієнт OpenAI-сумісного API (/v1/chat/completions) зі стрімінгом, таймаутами й зрозумілими помилками.
// Працює з Ollama, llama-server, LM Studio, vLLM - будь-яким рантаймом з OpenAI-протоколом.

/** Помилка з типом, за яким інтерфейс показує людське повідомлення й підказку. */
export class LlmError extends Error {
  constructor(kind, message, hint = "") {
    super(message);
    this.kind = kind;  // offline | timeout | model | http | truncated | bad-json | aborted
    this.hint = hint;
  }
}

const trim = (url) => url.replace(/\/+$/, "");

/** Перевіряє, що сервер живий, і повертає список моделей. */
export async function listModels(baseUrl, timeoutMs = 4000) {
  let response;
  try {
    response = await fetch(`${trim(baseUrl)}/models`, { signal: AbortSignal.timeout(timeoutMs) });
  } catch (error) {
    throw offlineError(baseUrl, error);
  }
  if (!response.ok) throw new LlmError("http", `сервер відповів ${response.status} на /models`);
  const data = await response.json();
  return (data.data || []).map((m) => m.id);
}

/**
 * Один стрімінговий запит.
 * @param {object} req - {baseUrl, model, messages, maxTokens, temperature, topP, schema, schemaName,
 *                        reasoningEffort, firstTokenTimeoutMs, totalTimeoutMs, signal}
 * @param {object} cb - {onDelta(text), onReasoning(), onFirstToken()}
 * @returns {Promise<{text, finishReason, tokens, promptTokens, ttft, tokensPerS, latency}>}
 */
export async function streamChat(req, cb = {}) {
  // Сервери відрізняються підтримкою необовʼязкових параметрів: якщо сервер їх відкидає,
  // повторюємо запит без них. Так застосунок працює і з моделлю без режиму міркування,
  // і з рантаймом без structured outputs.
  const attempt = { ...req };
  for (let i = 0; i < 3; i++) {
    try {
      return await streamOnce(attempt, cb);
    } catch (error) {
      if (error.kind !== "http-400") throw error;
      if (attempt.reasoningEffort && /think|reason/i.test(error.message)) { attempt.reasoningEffort = undefined; continue; }
      if (attempt.schema && /response_format|schema|json/i.test(error.message)) { attempt.schema = undefined; continue; }
      throw new LlmError("http", error.message);
    }
  }
  throw new LlmError("http", "сервер відкинув запит навіть без необовʼязкових параметрів");
}

async function streamOnce(req, cb) {
  const controller = new AbortController();
  let timeoutKind = null;
  const abortWith = (kind) => { timeoutKind = kind; controller.abort(); };
  const onUserAbort = () => abortWith("aborted");
  req.signal?.addEventListener("abort", onUserAbort);
  let firstTimer = setTimeout(() => abortWith("first"), req.firstTokenTimeoutMs ?? 90000);
  const totalTimer = setTimeout(() => abortWith("total"), req.totalTimeoutMs ?? 240000);

  const body = {
    model: req.model,
    messages: req.messages,
    stream: true,
    stream_options: { include_usage: true },
    max_tokens: req.maxTokens,  // обовʼязковий ліміт: інакше мала модель може зациклитися
    temperature: req.temperature,
    top_p: req.topP,
  };
  if (req.reasoningEffort) body.reasoning_effort = req.reasoningEffort;
  if (req.schema) {
    body.response_format = {
      type: "json_schema",
      json_schema: { name: req.schemaName || "answer", strict: true, schema: req.schema },
    };
  }

  const start = performance.now();
  let first = null, text = "", finishReason = null, tokens = null, promptTokens = null;
  try {
    let response;
    try {
      response = await fetch(`${trim(req.baseUrl)}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: "Bearer local" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (error) {
      if (timeoutKind) throw timeoutError(timeoutKind, req);
      throw offlineError(req.baseUrl, error);
    }
    if (!response.ok) throw await httpError(response, req.model);

    // Server-Sent Events: рядки "data: {...}", в кінці "data: [DONE]"
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      let chunk;
      try {
        chunk = await reader.read();
      } catch (error) {
        if (timeoutKind) throw timeoutError(timeoutKind, req);
        throw new LlmError("offline", "зʼєднання з сервером обірвалося під час генерації", "Перевірте, чи не впав рантайм (наприклад, через нестачу памʼяті).");
      }
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
        let event;
        try { event = JSON.parse(line.slice(6)); } catch { continue; }
        if (event.error) throw new LlmError("http", `помилка сервера: ${event.error.message || JSON.stringify(event.error)}`);
        if (event.usage) { tokens = event.usage.completion_tokens; promptTokens = event.usage.prompt_tokens; }
        const choice = event.choices?.[0];
        if (!choice) continue;
        if (choice.finish_reason) finishReason = choice.finish_reason;
        const delta = choice.delta || {};
        // Ollama віддає міркування в полі reasoning, llama-server і vLLM - у reasoning_content
        if (delta.reasoning || delta.reasoning_content) {
          if (firstTimer) { clearTimeout(firstTimer); firstTimer = null; }
          cb.onReasoning?.();
        }
        if (delta.content) {
          if (first === null) {
            first = performance.now();
            clearTimeout(firstTimer); firstTimer = null;
            cb.onFirstToken?.();
          }
          text += delta.content;
          cb.onDelta?.(delta.content, text);
        }
      }
    }
  } finally {
    clearTimeout(firstTimer);
    clearTimeout(totalTimer);
    req.signal?.removeEventListener("abort", onUserAbort);
  }

  const end = performance.now();
  if (first === null) {
    throw new LlmError("bad-json", "модель не повернула жодного токена відповіді",
      "Якщо модель міркує, оберіть у налаштуваннях «без міркування» або збільшіть ліміт токенів.");
  }
  return {
    text,
    finishReason,
    tokens,
    promptTokens,
    ttft: (first - start) / 1000,
    latency: (end - start) / 1000,
    tokensPerS: tokens > 1 && end > first ? (tokens - 1) / ((end - first) / 1000) : null,
  };
}

function offlineError(baseUrl, error) {
  const fromFile = typeof location !== "undefined" && location.protocol === "file:";
  if (fromFile) {
    return new LlmError("offline", "сторінку відкрито з диска - браузер блокує запити до сервера (CORS)",
      "Запустіть ./start.sh у теці app і відкрийте http://localhost:8000");
  }
  return new LlmError("offline", `не вдалося підключитися до ${baseUrl}`,
    "Запустіть рантайм: «ollama serve» (або застосунок Ollama). Якщо сервер працює - перевірте адресу API, " +
    `а для іншого порту/хоста дозвольте CORS: OLLAMA_ORIGINS="http://localhost:8000". (${error?.message || error})`);
}

function timeoutError(kind, req) {
  if (kind === "aborted") return new LlmError("aborted", "генерацію зупинено");
  if (kind === "first") {
    return new LlmError("timeout", `модель не почала відповідати за ${Math.round((req.firstTokenTimeoutMs ?? 90000) / 1000)} с`,
      "Перший запит після запуску включає завантаження моделі в памʼять - спробуйте ще раз. " +
      "Якщо повторюється, конспект задовгий для вашого заліза або модель завелика.");
  }
  return new LlmError("timeout", `генерація триває довше за ${Math.round((req.totalTimeoutMs ?? 240000) / 1000)} с - зупинено`,
    "Зменшіть кількість карток або оберіть меншу/швидшу модель.");
}

async function httpError(response, model) {
  let detail = "";
  try {
    const raw = await response.text();
    try { detail = JSON.parse(raw).error?.message || raw; } catch { detail = raw; }
  } catch {}
  detail = String(detail).slice(0, 300);
  if (response.status === 404 || /not found/i.test(detail)) {
    return new LlmError("model", `модель «${model}» не знайдена на сервері`,
      `Завантажте її: ollama pull ${model} - або виберіть іншу модель зі списку в налаштуваннях.`);
  }
  if (response.status === 400) {
    const e = new LlmError("http-400", detail || "сервер відхилив запит (400)");
    return e;
  }
  if (response.status >= 500 && /memory|alloc|oom/i.test(detail)) {
    return new LlmError("http", `серверу не вистачило памʼяті: ${detail}`, "Оберіть меншу модель/квантизацію або зменшіть контекст.");
  }
  return new LlmError("http", `сервер відповів ${response.status}: ${detail}`);
}
