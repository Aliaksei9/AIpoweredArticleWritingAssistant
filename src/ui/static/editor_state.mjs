/** Review operations use UTF-16 offsets, as provided by the controller. */
export class ReviewState {
  constructor(text = "") { this.text = text; this.cards = []; this.activeId = null; this.history = []; this.issues = []; this.languageCheck = null; }
  get pending() { return this.cards.filter(card => card.status === "pending"); }
  get active() { return this.pending.find(card => card.id === this.activeId) || null; }
  invalidate({ language = true } = {}) { this.cards = []; this.activeId = null; this.history = []; if (language) { this.issues = []; this.languageCheck = null; } }
  manualEdit(text) { this.text = text; this.invalidate(); }
  load(cards) {
    const ordered = cards.map(card => ({ ...structuredClone(card), id: String(card.id), status: "pending" }))
      .sort((a, b) => a.start - b.start);
    const ids = new Set(); let end = 0;
    for (const card of ordered) {
      if (!Number.isInteger(card.start) || !Number.isInteger(card.end) || card.start < end || card.end <= card.start
          || card.end > this.text.length || this.text.slice(card.start, card.end) !== card.original
          || typeof card.rewrite !== "string" || !card.rewrite.trim() || ids.has(card.id)) {
        throw new Error("Рекомендации не соответствуют текущему тексту. Повторите поиск.");
      }
      ids.add(card.id); end = card.end;
    }
    this.cards = ordered; this.activeId = ordered[0]?.id || null; this.history = [];
  }
  select(id) { if (this.pending.some(card => card.id === id)) this.activeId = id; }
  remember() {
    this.history.push({ text: this.text, cards: structuredClone(this.cards), activeId: this.activeId, issues: structuredClone(this.issues), languageCheck: structuredClone(this.languageCheck) });
  }
  accept(id) {
    const card = this.pending.find(item => item.id === id);
    if (!card || this.text.slice(card.start, card.end) !== card.original) {
      throw new Error("Предложение уже изменено. Запросите новые рекомендации.");
    }
    this.remember();
    const change = card.rewrite.length - (card.end - card.start), oldEnd = card.end;
    this.text = this.text.slice(0, card.start) + card.rewrite + this.text.slice(oldEnd);
    if (card.original !== card.rewrite) { this.issues = []; this.languageCheck = null; }
    card.status = "accepted";
    for (const next of this.pending) {
      if (next.start >= oldEnd) { next.start += change; next.end += change; }
    }
    this.activeId = this.pending[0]?.id || null;
  }
  reject(id) {
    const card = this.pending.find(item => item.id === id);
    if (!card) return;
    this.remember(); card.status = "rejected"; this.activeId = this.pending[0]?.id || null;
  }
  undo() {
    const previous = this.history.pop();
    if (previous) { this.text = previous.text; this.cards = previous.cards; this.activeId = previous.activeId; this.issues = previous.issues; this.languageCheck = previous.languageCheck; }
    return Boolean(previous);
  }
  loadLanguage(issues) {
    const copy = structuredClone(issues), ids = new Set(), allEdits = [];
    for (const issue of copy) {
      if (ids.has(issue.id) || !["spelling", "punctuation", "grammar", "syntax"].includes(issue.kind)
          || !Number.isInteger(issue.start) || !Number.isInteger(issue.end) || issue.start < 0
          || issue.end <= issue.start || issue.end > this.text.length || this.text.slice(issue.start, issue.end) !== issue.original
          || !Array.isArray(issue.edits) || !issue.edits.length) throw new Error("Проверка относится к другой версии текста.");
      ids.add(issue.id);
      for (const edit of issue.edits) {
        if (!Number.isInteger(edit.start) || !Number.isInteger(edit.end) || edit.start < issue.start || edit.end < edit.start
            || edit.end > issue.end || this.text.slice(edit.start, edit.end) !== edit.original || typeof edit.replacement !== "string"
            || edit.original === edit.replacement || allEdits.some(other => editsConflict(edit, other))) {
          throw new Error("Не удалось сопоставить исправления с текстом.");
        }
        allEdits.push(edit);
      }
      issue.status = "pending";
    }
    this.issues = copy; this.history = []; this.languageCheck = null;
    for (const issue of this.issues) issue.rewrite = preview(this.text, issue);
  }
  get languagePending() { return this.issues.filter(issue => issue.status === "pending"); }
  issuesAt(position) { return this.languagePending.filter(issue => position >= issue.start && position < issue.end); }
  acceptLanguage(id) {
    const issue = this.languagePending.find(item => item.id === id);
    if (!issue || this.text.slice(issue.start, issue.end) !== issue.original) throw new Error("Текст уже изменён. Повторите проверку.");
    this.remember(); issue.status = "accepted";
    for (const edit of [...issue.edits].sort((a, b) => b.start - a.start)) {
      if (this.text.slice(edit.start, edit.end) !== edit.original) throw new Error("Исправление больше не соответствует тексту.");
      const delta = edit.replacement.length - (edit.end - edit.start);
      this.text = this.text.slice(0, edit.start) + edit.replacement + this.text.slice(edit.end);
      for (const other of this.languagePending) {
        for (const next of other.edits) {
          if (next.start >= edit.end) { next.start += delta; next.end += delta; }
        }
        if (other.start >= edit.end) other.start += delta;
        if (other.end > edit.end || (other.end === edit.end && edit.start < edit.end)) other.end += delta;
      }
    }
    for (const other of this.languagePending) {
      other.original = this.text.slice(other.start, other.end); other.rewrite = preview(this.text, other);
    }
    // Whole-sentence editorial rewrites belong to the old document version.
    this.cards = []; this.activeId = null;
  }
  rejectLanguage(id) {
    const issue = this.languagePending.find(item => item.id === id);
    if (issue) { this.remember(); issue.status = "rejected"; }
  }
}

function editsConflict(a, b) {
  if (a.start === a.end && b.start === b.end) return a.start === b.start;
  if (a.start === a.end) return a.start > b.start && a.start < b.end;
  if (b.start === b.end) return b.start > a.start && b.start < a.end;
  return a.start < b.end && b.start < a.end;
}
function preview(text, issue) {
  let value = text.slice(issue.start, issue.end);
  for (const edit of [...issue.edits].sort((a, b) => b.start - a.start)) {
    const start = edit.start - issue.start, end = edit.end - issue.start;
    value = value.slice(0, start) + edit.replacement + value.slice(end);
  }
  return value;
}

/** Non-overlapping mirror segments retain every overlapping highlight type. */
export function highlightSegments(text, active, issues) {
  const ranges = [...issues.map(i => ({ ...i, style: i.kind === "spelling" ? "spelling" : "sentence-error" })),
    ...(active ? [{ ...active, style: "editorial" }] : [])];
  const boundaries = [...new Set([0, text.length, ...ranges.flatMap(r => [r.start, r.end])])].sort((a, b) => a - b);
  return boundaries.slice(0, -1).map((start, i) => {
    const end = boundaries[i + 1], hits = ranges.filter(r => start < r.end && end > r.start);
    return { start, end, text: text.slice(start, end), classes: [...new Set(hits.map(r => r.style))] };
  });
}
