/** A separate, cancellable proofreading request and an overlap-aware popup. */
const $ = id => document.getElementById(id);
const labels = { spelling: "Орфография", punctuation: "Пунктуация", grammar: "Грамматика", syntax: "Синтаксис" };
function node(tag, className, text) {
  const value = document.createElement(tag); value.className = className;
  if (text !== undefined) value.textContent = text;
  return value;
}

export class LanguageUI {
  constructor(state, api, { render, changed, toast }) {
    this.state = state; this.api = api; this.renderAll = render; this.changed = changed; this.toast = toast;
    this.status = "idle"; this.message = ""; this.sequence = 0; this.job = null; this.creation = null;
    this.partial = false;
    this.cancellation = Promise.resolve(); this.openIds = []; this.anchor = { x: 20, y: 120 }; this.enabled = true;
    $("check-language").addEventListener("click", () => this.check());
    $("cancel-language").addEventListener("click", async () => { await this.cancel(); this.status = "idle"; this.render(); });
    $("close-language-popup").addEventListener("click", () => this.close());
    $("next-language-error").addEventListener("click", () => this.next());
    $("language-fixes").addEventListener("click", event => {
      const action = event.target.closest("[data-language-action]"); if (!action) return;
      const id = action.dataset.issueId;
      try {
        if (action.dataset.languageAction === "accept") {
          this.state.acceptLanguage(id); $("editor").value = this.state.text; this.changed(); this.toast("Исправление принято");
        } else { this.state.rejectLanguage(id); this.toast("Исправление пропущено"); }
        this.openIds = this.openIds.filter(value => this.state.languagePending.some(i => i.id === value));
        this.renderAll(); this.renderPopup();
      } catch (error) { this.message = error.message; this.invalidate(); this.status = "error"; this.renderAll(); }
    });
    document.addEventListener("pointerdown", event => {
      if (!event.target.closest("#language-popup, #editor, #next-language-error")) this.close();
    });
    document.addEventListener("keydown", event => { if (event.key === "Escape") this.close(); });
    $("editor").addEventListener("scroll", () => this.close());
    window.addEventListener("resize", () => this.close());
  }
  close() { this.openIds = []; $("language-popup").hidden = true; }
  async cancel() {
    this.sequence++;
    const active = this.job, creation = this.creation; this.job = this.creation = null;
    if (this.status === "loading") this.status = "idle";
    this.cancellation = this.cancellation.then(async () => {
      let id = active;
      if (creation) { try { id = (await creation).id; } catch { return; } }
      if (id) { try { await this.api(`/api/language-checks/${id}`, { method: "DELETE" }); } catch { /* Stale output stays ignored. */ } }
    });
    return this.cancellation;
  }
  invalidate() {
    const checked = this.status !== "idle" || this.state.issues.length;
    void this.cancel(); this.state.issues = []; this.state.languageCheck = null; this.partial = false; this.close();
    this.status = checked ? "stale" : "idle";
  }
  async check() {
    if (!this.state.text.trim()) return;
    await this.cancel(); const sequence = ++this.sequence, text = this.state.text;
    this.state.issues = []; this.state.languageCheck = null; this.partial = false; this.close(); this.status = "loading"; this.renderAll();
    try {
      const creation = this.api("/api/language-checks", { method: "POST", body: { text } }); this.creation = creation;
      const job = await creation; if (this.creation === creation) this.creation = null;
      if (sequence !== this.sequence || text !== this.state.text) {
        void this.api(`/api/language-checks/${job.id}`, { method: "DELETE" }).catch(() => {}); return;
      }
      this.job = job.id;
      while (sequence === this.sequence && text === this.state.text) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        if (sequence !== this.sequence || text !== this.state.text) return;
        const current = await this.api(`/api/language-checks/${job.id}`);
        if (sequence !== this.sequence || text !== this.state.text) return;
        if (current.status === "succeeded") {
          if (current.result.offset_unit !== "utf16") throw new Error("Неизвестный формат позиций ошибок.");
          this.state.loadLanguage(current.result.issues); this.status = "ready"; this.job = null;
          this.partial = Boolean(current.result.unchecked_sentences?.length);
          this.message = (current.result.warnings || []).join(" ");
          this.state.languageCheck = { partial: this.partial, message: this.message }; this.renderAll();
          this.toast(this.partial ? "Проверка выполнена частично — см. предупреждение" : this.state.languagePending.length ? `Найдено исправлений: ${this.state.languagePending.length}` : "Языковых ошибок не найдено"); return;
        }
        if (current.status === "failed") throw new Error(current.error || "Не удалось проверить текст.");
        if (current.status === "cancelled") { this.status = "idle"; this.job = null; this.renderAll(); return; }
      }
    } catch (error) {
      if (sequence === this.sequence && text === this.state.text) {
        this.status = "error"; this.message = error.message; this.job = null; this.renderAll();
      }
    }
  }
  render() {
    $("check-language").disabled = !this.state.text.trim() || !this.enabled || this.status === "loading";
    $("check-language").querySelector("span").textContent = this.status === "loading" ? "Проверяем текст…" : "Проверить орфографию и пунктуацию";
    $("cancel-language").hidden = this.status !== "loading";
    $("next-language-error").hidden = !this.state.languagePending.length;
    const status = $("language-status"), pending = this.state.languagePending;
    status.classList.toggle("language-failure", this.status === "error");
    status.textContent = this.status === "loading" ? "Проверяем орфографию, пунктуацию, грамматику и синтаксис…"
      : this.status === "error" ? this.message : this.status === "stale" ? "Текст изменился. Повторите проверку."
      : this.status === "ready" ? this.partial ? `Проверка выполнена частично. Исправлений: ${pending.length}.` : pending.length ? `Исправлений: ${pending.length}. Нажмите на подсвеченный фрагмент.`
      : this.state.issues.length ? "Все исправления рассмотрены" : "Языковых ошибок не найдено"
      : "Грамматика и синтаксис тоже проверяются";
    status.title = this.status === "ready" ? this.message : "";
    $("language-warning").hidden = this.status !== "ready" || !this.partial;
    $("language-warning").textContent = this.status === "ready" && this.partial ? this.message : "";
    this.renderPopup();
  }
  openAt(position, event) {
    // Use the clicked mirror fragment: the textarea caret can fall just after a word.
    if (event) {
      for (const mark of $("highlight-content").querySelectorAll("mark[data-start]")) {
        if ([...mark.getClientRects()].some(r => event.clientX >= r.left && event.clientX <= r.right && event.clientY >= r.top && event.clientY <= r.bottom)) {
          position = Number(mark.dataset.start); break;
        }
      }
      this.anchor = { x: event.clientX, y: event.clientY + 20 };
    }
    this.openIds = this.state.issuesAt(position).map(i => i.id);
    this.renderPopup();
  }
  renderPopup() {
    const popup = $("language-popup"), host = $("language-fixes"); host.replaceChildren();
    const issues = this.state.languagePending.filter(i => this.openIds.includes(i.id));
    popup.hidden = !issues.length; if (!issues.length) return;
    for (const issue of issues) {
      const box = node("article", "language-fix " + (issue.kind === "spelling" ? "red-fix" : "blue-fix"));
      box.append(node("strong", "language-kind", labels[issue.kind]), node("p", "language-original", issue.original),
        node("p", "language-replacement", issue.rewrite), node("p", "language-explanation", issue.explanation));
      const actions = node("div", "card-actions");
      for (const [action, label] of [["accept", "Применить"], ["reject", "Пропустить"]]) {
        const button = node("button", action === "accept" ? "accept-button" : "reject-button", label);
        button.type = "button"; button.dataset.languageAction = action; button.dataset.issueId = issue.id;
        button.setAttribute("aria-label", `${label}: ${labels[issue.kind]}`); actions.append(button);
      }
      box.append(actions); host.append(box);
    }
    const editorBox = $("editor").getBoundingClientRect();
    const x = editorBox.right + popup.offsetWidth + 24 < window.innerWidth ? editorBox.right + 12 : this.anchor.x;
    popup.style.left = `${Math.max(10, Math.min(x, window.innerWidth - popup.offsetWidth - 10))}px`;
    popup.style.top = `${Math.max(10, Math.min(this.anchor.y, window.innerHeight - popup.offsetHeight - 10))}px`;
  }
  next() {
    const pending = this.state.languagePending;
    if (!pending.length) return;
    const last = pending.findIndex(i => this.openIds.includes(i.id));
    const issue = pending[(last + 1) % pending.length];
    const editor = $("editor"); editor.focus(); editor.setSelectionRange(issue.start, issue.start);
    const marks = [...$("highlight-content").querySelectorAll("mark[data-start]")];
    const mark = marks.find(m => Number(m.dataset.start) >= issue.start && Number(m.dataset.start) < issue.end);
    if (mark) {
      const rect = mark.getBoundingClientRect(), box = editor.getBoundingClientRect();
      editor.scrollTop += rect.top - box.top - 50;
      $("editor-highlights").scrollTop = editor.scrollTop;
      const updated = mark.getBoundingClientRect(); this.anchor = { x: updated.left, y: updated.bottom + 12 };
    }
    this.openAt(issue.start); $("language-fixes").querySelector("button")?.focus({ preventScroll: true });
  }
}
