import { ReviewState, highlightSegments } from "/static/editor_state.mjs";
import { LanguageUI } from "/static/language_ui.js";

const $ = id => document.getElementById(id);
const editor = $("editor"), state = new ReviewState();
const number = new Intl.NumberFormat("ru-RU"), scoreNumber = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const DRAFT_KEY = "publication-editor-draft-v1";
const sample = "Короче говоря, студия представила обновление игры с 3 новыми картами. Это обновление, как бы, можно бесплатно попробовать всем игрокам. Ну и ещё разработчики, значит, исправили ошибки сетевого режима.";
const aspects = { clarity: "Ясность", structure: "Структура", engagement: "Вовлечение", style: "Стиль", grammar: "Грамматика" };
let workspace = null, prediction = null, predictionKey = null, evaluating = false, predictionRequest = 0;
let correctionState = "idle", correctionText = "", correctionSequence = 0, jobId = null;
let correctionWarnings = [];
let saveTimer, toastTimer, requestAbort, draftSettings = {};
let pendingCreation = null, cancellation = Promise.resolve();

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const use = document.createElementNS(svg.namespaceURI, "use");
  use.setAttribute("href", `#i-${name}`); svg.append(use); svg.setAttribute("aria-hidden", "true");
  return svg;
}
function topic() { return $("topic").value || null; }
function signature() { return JSON.stringify([state.text, topic(), $("resource").value, "TG"]); }
function correctionSignature() { return JSON.stringify([state.text, topic()]); }
function showNotice(message, isError = false) {
  $("notice-text").textContent = message; $("notice").classList.toggle("error", isError); $("notice").hidden = false;
}
function toast(message) {
  clearTimeout(toastTimer); $("toast").textContent = message; $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3200);
}
async function api(path, { method = "GET", body, signal } = {}) {
  const response = await fetch(path, { method, signal, headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Не удалось выполнить запрос. Проверьте подключение к приложению.");
  return data;
}
function plural(value, forms) {
  const a = value % 10, b = value % 100;
  return forms[b > 10 && b < 20 ? 2 : a === 1 ? 0 : a >= 2 && a <= 4 ? 1 : 2];
}
function saveDraft() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try {
      localStorage.setItem(DRAFT_KEY, JSON.stringify({ text: state.text, topic: topic(), resource: $("resource").value }));
      $("saved-status").textContent = "Сохранено в этом браузере";
    } catch { $("saved-status").textContent = "Локальное сохранение недоступно"; }
  }, 300);
}
function syncScroll() {
  $("editor-highlights").scrollTop = editor.scrollTop;
  $("editor-highlights").scrollLeft = editor.scrollLeft;
}
function renderHighlight() {
  const host = $("highlight-content"); host.replaceChildren();
  for (const segment of highlightSegments(state.text, state.active, state.languagePending)) {
    if (segment.classes.length) {
      const mark = el("mark", segment.classes.join(" "), segment.text); mark.dataset.start = segment.start; host.append(mark);
    } else host.append(document.createTextNode(segment.text));
  }
  host.append(document.createTextNode("\n"));
  $("editor-highlights").style.width = `${editor.clientWidth}px`;
  syncScroll();
}
function scrollToActive() {
  const mark = $("highlight-content").querySelector("mark.editorial");
  if (!mark) return;
  const y = mark.getBoundingClientRect().top - editor.getBoundingClientRect().top + editor.scrollTop;
  editor.scrollTop = Math.max(0, y - editor.clientHeight * .35); syncScroll();
}
function renderCounts() {
  const chars = Array.from(state.text).length, words = state.text.trim() ? state.text.trim().split(/\s+/u).length : 0;
  $("text-count").textContent = `${number.format(words)} ${plural(words, ["слово", "слова", "слов"])} · ${number.format(chars)} ${plural(chars, ["символ", "символа", "символов"])}`;
  $("insert-example").hidden = Boolean(state.text.trim());
  $("undo-edit").disabled = !state.history.length;
}
function renderButtons() {
  const hasText = Boolean(state.text.trim());
  $("evaluate").disabled = !workspace || !hasText || !$("resource").value || evaluating;
  $("evaluate").querySelector("span").textContent = evaluating ? "Рассчитываем оценку и метрики…" : "Обновить оценку соответствия";
  $("evaluate").classList.toggle("busy", evaluating);
  $("get-corrections").disabled = !workspace || !hasText || !workspace.corrections_configured || correctionState === "loading";
  $("get-corrections").querySelector("span").textContent = correctionState === "loading" ? "Ищем улучшения…" : state.cards.length ? "Найти новые рекомендации" : "Найти улучшения";
  $("cancel-corrections").hidden = correctionState !== "loading";
}
function renderPrediction() {
  const stale = prediction && predictionKey !== signature();
  document.querySelector(".forecast-panel").classList.toggle("stale", Boolean(stale));
  $("score-state").textContent = evaluating ? "Рассчитываем…" : stale ? "Обновите оценку" : prediction ? "Рассчитан" : "Ещё не рассчитан";
  if (prediction) {
    const score = Math.max(-10, Math.min(10, prediction.score)), percent = (score + 10) * 5;
    $("score-value").textContent = (score > 0 ? "+" : "") + scoreNumber.format(score);
    $("score-fill").style.width = `${percent}%`; $("score-marker").style.left = `${percent}%`; $("score-marker").hidden = false;
    $("score-topic").textContent = `Тема: ${prediction.topic}`;
    for (const metric of ["views", "reactions", "comments"]) $("metric-" + metric).textContent = number.format(prediction.metrics[metric]);
    $("forecast-footnote").textContent = stale ? "Текст или параметры изменились. Пересчитайте прогноз для новой версии." :
      prediction.audience_source === "model_snapshot" ? "Прогноз по сохранённой аудитории канала. Для актуальной аудитории подключите БД." :
      "Прогноз для выбранного канала. Обучение выполнено на метриках статей возраста 2–14 дней.";
  }
}
function emptyState(title, description, loading = false) {
  const box = el("div", "empty-state" + (loading ? " loading-state" : ""));
  const graphic = el("div", "empty-icon"); graphic.append(loading ? el("span", "spinner") : icon("spark"));
  box.append(graphic, el("h3", "", title), el("p", "", description));
  return box;
}
function renderCorrectionWarnings() {
  const host = $("retrieval-warning"); host.replaceChildren();
  host.hidden = !correctionWarnings.length;
  for (const warning of correctionWarnings) host.append(el("p", "", warning));
}
function renderSuggestions() {
  renderCorrectionWarnings();
  const host = $("suggestions"); host.replaceChildren();
  $("suggestion-count").textContent = state.pending.length;
  const accepted = state.cards.filter(card => card.status === "accepted").length;
  const rejected = state.cards.filter(card => card.status === "rejected").length;
  $("review-summary").textContent = accepted || rejected ? `Принято: ${accepted} · Отклонено: ${rejected}` : "";
  if (!state.pending.length) {
    if (correctionState === "loading") host.append(emptyState("Ищем подходящие формулировки", "Сравниваем текст с популярными статьями и готовим рекомендации. Оценку можно пересчитать уже сейчас.", true));
    else if (correctionState === "error") host.append(emptyState("Не удалось получить рекомендации", correctionText));
    else if (state.cards.length) host.append(emptyState("Все рекомендации рассмотрены", "Обновите оценку, чтобы увидеть прогноз для отредактированного текста."));
    else if (correctionState === "stale") host.append(emptyState("Текст изменился", "Запросите рекомендации для новой версии. Карточки всегда относятся к конкретным предложениям."));
    else host.append(emptyState("Свежий взгляд на ваш текст", "Здесь появятся рекомендации и новые формулировки. Нажмите «Найти улучшения», когда текст будет готов."));
  }
  for (const card of state.pending) {
    const box = el("article", "suggestion-card" + (card.id === state.activeId ? " active" : ""));
    box.dataset.cardId = card.id; box.tabIndex = 0; box.setAttribute("role", "group");
    box.setAttribute("aria-label", `Рекомендация к предложению ${card.id}`);
    if (card.id === state.activeId) box.setAttribute("aria-current", "true");
    const top = el("div", "suggestion-topline");
    top.append(el("span", "sentence-label", `Предложение ${card.id}`),
      el("span", "aspect-label", aspects[card.recommendations[0]?.aspect] || "Редактура"));
    box.append(top, el("p", "original-caption", card.original));
    const same = card.original === card.rewrite;
    if (!same) box.append(el("p", "rewrite-caption", card.rewrite));
    for (const recommendation of card.recommendations) {
      box.append(el("p", "recommendation-text", recommendation.recommendation), el("p", "recommendation-reason", recommendation.reason));
    }
    const actions = el("div", "card-actions");
    const accept = el("button", "accept-button"), reject = el("button", "reject-button");
    accept.type = reject.type = "button";
    accept.dataset.action = "accept"; reject.dataset.action = "reject";
    accept.append(icon("check"), document.createTextNode(same ? "Оставить" : "Принять"));
    reject.append(icon("close"), document.createTextNode("Отклонить")); actions.append(accept, reject); box.append(actions); host.append(box);
  }
  renderButtons();
}
function renderAll() { renderCounts(); renderHighlight(); renderSuggestions(); renderPrediction(); renderButtons(); language.render(); }

async function cancelCorrection() {
  correctionSequence++;
  correctionWarnings = [];
  const active = jobId, creation = pendingCreation; jobId = null; pendingCreation = null;
  if (correctionState === "loading") correctionState = "idle";
  cancellation = cancellation.then(async () => {
    let identifier = active;
    if (creation) { try { identifier = (await creation).id; } catch { return; } }
    if (identifier) { try { await api(`/api/corrections/${identifier}`, { method: "DELETE" }); } catch { /* Ignore late results locally. */ } }
  });
  return cancellation;
}
function invalidateText(text, manual = true) {
  language.invalidate();
  const hadCards = state.cards.length || correctionState === "loading";
  void cancelCorrection();
  state.manualEdit(text); correctionState = hadCards ? "stale" : "idle";
  if (!manual) editor.value = text;
  renderAll(); saveDraft();
}
async function evaluate() {
  if (!state.text.trim()) return;
  requestAbort?.abort(); requestAbort = new AbortController();
  const request = ++predictionRequest, key = signature();
  evaluating = true; renderPrediction(); renderButtons();
  try {
    const result = await api("/api/predictions", { method: "POST", signal: requestAbort.signal,
      body: { text: state.text, topic: topic(), resource_id: $("resource").value, source: "TG" } });
    if (request === predictionRequest && key === signature()) {
      prediction = result; predictionKey = key; toast("Оценка и прогноз обновлены");
    }
  } catch (error) { if (error.name !== "AbortError" && request === predictionRequest && key === signature()) showNotice(error.message, true); }
  finally { if (request === predictionRequest) { evaluating = false; renderPrediction(); renderButtons(); } }
}
async function getCorrections() {
  if (!state.text.trim()) return;
  await cancelCorrection();
  const sequence = ++correctionSequence, key = correctionSignature();
  state.invalidate({ language: false }); correctionState = "loading"; renderAll();
  try {
    const creation = api("/api/corrections", { method: "POST", body: { text: state.text, topic: topic() } });
    pendingCreation = creation;
    const job = await creation;
    if (pendingCreation === creation) pendingCreation = null;
    if (sequence !== correctionSequence || key !== correctionSignature()) {
      void api(`/api/corrections/${job.id}`, { method: "DELETE" }).catch(() => {}); return;
    }
    jobId = job.id;
    while (sequence === correctionSequence && key === correctionSignature()) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      if (sequence !== correctionSequence) return;
      const current = await api(`/api/corrections/${job.id}`);
      if (sequence !== correctionSequence || key !== correctionSignature()) return;
      correctionWarnings = current.warnings || [];
      renderCorrectionWarnings();
      if (current.status === "succeeded") {
        if (current.result.offset_unit !== "utf16") throw new Error("Неизвестный формат позиций предложений.");
        correctionWarnings = current.result.warnings || correctionWarnings;
        state.load(current.result.cards); correctionState = "ready"; jobId = null; renderAll(); scrollToActive();
        toast(`Рекомендаций к предложениям: ${state.pending.length}`); return;
      }
      if (current.status === "failed") throw new Error(current.error || "Не удалось получить рекомендации.");
      if (current.status === "cancelled") { correctionState = "idle"; jobId = null; renderAll(); return; }
    }
  } catch (error) {
    if (sequence === correctionSequence && key === correctionSignature()) {
      correctionState = "error"; correctionText = error.message; jobId = null; renderAll();
    }
  }
}
function applyCard(action, id) {
  try {
    if (action === "accept") {
      const previous = state.text; state.accept(id);
      if (previous !== state.text) language.invalidate();
      editor.value = state.text; saveDraft(); toast("Правка принята. Оценку можно обновить.");
    }
    else { state.reject(id); toast("Рекомендация отклонена"); }
    renderAll(); scrollToActive();
  } catch (error) { showNotice(error.message, true); invalidateText(editor.value); }
}
function undoReview() {
  void cancelCorrection(); void language.cancel(); language.close();
  if (!state.undo()) return;
  language.status = state.issues.length ? "ready" : "idle";
  language.partial = Boolean(state.languageCheck?.partial); language.message = state.languageCheck?.message || "";
  editor.value = state.text; renderAll(); saveDraft(); toast("Последнее действие отменено");
}
function renderAudience() {
  const selected = workspace?.resources.find(resource => resource.id === $("resource").value);
  $("audience-caption").textContent = selected ? `${number.format(selected.audience)} подписчиков` : "Выберите канал для прогноза";
}
async function loadWorkspace(refresh = false) {
  $("connection-label").textContent = "Подключаем модели…";
  $("refresh-workspace").disabled = true;
  try {
    const previousResource = workspace ? $("resource").value : draftSettings.resource;
    const previousTopic = workspace ? $("topic").value : draftSettings.topic;
    workspace = await api("/api/workspace" + (refresh ? "?refresh=true" : ""));
    $("resource").replaceChildren();
    for (const resource of workspace.resources) {
      const option = el("option", "", resource.name); option.value = resource.id; $("resource").append(option);
    }
    if (workspace.resources.some(item => item.id === previousResource)) $("resource").value = previousResource;
    else { const jeteed = workspace.resources.find(item => item.name.toLowerCase() === "jeteed"); if (jeteed) $("resource").value = jeteed.id; }
    $("topic").replaceChildren(); const auto = el("option", "", "Определить автоматически"); auto.value = ""; $("topic").append(auto);
    for (const name of workspace.topics) { const option = el("option", "", name); option.value = name; $("topic").append(option); }
    if (workspace.topics.includes(previousTopic)) $("topic").value = previousTopic;
    $("resource").disabled = !workspace.resources.length; $("topic").disabled = false;
    $("connection-dot").className = "status-dot " + (workspace.database_available ? "ready" : "warning");
    $("connection-label").textContent = workspace.database_available ? "Модели и БД подключены" : "Модели подключены · БД недоступна";
    if (!workspace.database_available) showNotice("БД недоступна. Прогноз использует сохранённую аудиторию каналов. Для рекомендаций запустите PostgreSQL в Docker.");
    else if (!workspace.corrections_configured) showNotice("Для текстовых рекомендаций настройте OPENROUTER_API_KEY в .env и перезапустите приложение. Прогноз уже доступен.");
    else $("notice").hidden = true;
    const resource = workspace.resources.find(item => item.id === $("resource").value);
    if (prediction && resource?.audience !== prediction.resource.audience) predictionKey = null;
    renderAudience(); renderAll();
  } catch (error) {
    $("connection-label").textContent = "Не удалось подключить модели";
    $("connection-dot").className = "status-dot warning"; showNotice(error.message, true);
  } finally { $("refresh-workspace").disabled = false; }
}

const language = new LanguageUI(state, api, { render: renderAll, toast, changed: () => {
  void cancelCorrection(); correctionState = "stale"; saveDraft();
} });
editor.addEventListener("input", () => invalidateText(editor.value));
editor.addEventListener("scroll", syncScroll);
editor.addEventListener("click", event => {
  const card = state.pending.find(item => editor.selectionStart >= item.start && editor.selectionStart <= item.end);
  if (card && card.id !== state.activeId) { state.select(card.id); renderHighlight(); renderSuggestions(); }
  language.openAt(editor.selectionStart, event);
});
editor.addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && !event.shiftKey && state.history.length) {
    event.preventDefault(); undoReview();
  }
});
$("suggestions").addEventListener("click", event => {
  const card = event.target.closest("[data-card-id]"); if (!card) return;
  const action = event.target.closest("[data-action]");
  if (action) applyCard(action.dataset.action, card.dataset.cardId);
  else { state.select(card.dataset.cardId); renderHighlight(); renderSuggestions(); scrollToActive(); }
});
$("suggestions").addEventListener("keydown", event => {
  if ((event.key === "Enter" || event.key === " ") && event.target.matches("[data-card-id]")) {
    event.preventDefault(); const id = event.target.dataset.cardId;
    state.select(id); renderHighlight(); renderSuggestions(); scrollToActive();
    $("suggestions").querySelector(`[data-card-id="${CSS.escape(id)}"]`)?.focus({ preventScroll: true });
  }
});
$("resource").addEventListener("change", () => { renderAudience(); renderPrediction(); saveDraft(); });
$("topic").addEventListener("change", () => {
  const hadCards = state.cards.length || correctionState === "loading";
  void cancelCorrection(); state.invalidate({ language: false }); correctionState = hadCards ? "stale" : "idle";
  renderAll(); saveDraft();
});
$("evaluate").addEventListener("click", evaluate);
$("get-corrections").addEventListener("click", getCorrections);
$("cancel-corrections").addEventListener("click", async () => { await cancelCorrection(); correctionState = "idle"; renderAll(); });
$("undo-edit").addEventListener("click", undoReview);
$("insert-example").addEventListener("click", () => { if (!state.text.trim()) { invalidateText(sample, false); editor.focus(); } });
$("dismiss-notice").addEventListener("click", () => { $("notice").hidden = true; });
$("refresh-workspace").addEventListener("click", () => loadWorkspace(true));
new ResizeObserver(renderHighlight).observe(editor);
try { draftSettings = JSON.parse(localStorage.getItem(DRAFT_KEY) || "{}"); state.text = typeof draftSettings.text === "string" ? draftSettings.text : ""; } catch { draftSettings = {}; }
editor.value = state.text; renderAll(); void loadWorkspace();
