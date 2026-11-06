import assert from "node:assert/strict";
import { test } from "node:test";
import { ReviewState, highlightSegments } from "../src/ui/static/editor_state.mjs";

function setup() {
  const text = "🎮 Я незнаю почему он опаздал. Ещё текст.";
  const state = new ReviewState(text), end = text.indexOf(". ") + 1;
  const a = text.indexOf("незнаю"), b = text.indexOf("опаздал");
  state.loadLanguage([
    { id: "word1", kind: "spelling", start: a, end: a+6, original: "незнаю", edits: [{ start: a, end: a+6, original: "незнаю", replacement: "не знаю" }] },
    { id: "word2", kind: "spelling", start: b, end: b+7, original: "опаздал", edits: [{ start: b, end: b+7, original: "опаздал", replacement: "опоздал" }] },
    { id: "blue", kind: "punctuation", start: 0, end, original: text.slice(0, end), edits: [{ start: a+6, end: a+6, original: "", replacement: "," }] }
  ]);
  return state;
}
for (const order of [["word1", "blue", "word2"], ["blue", "word2", "word1"], ["word2", "word1", "blue"]]) {
  test(`Overlapping fixes commute: ${order.join(", ")}`, () => {
    const state = setup();
    for (const id of order) {
      state.acceptLanguage(id);
      for (const issue of state.languagePending) {
        assert.equal(state.text.slice(issue.start, issue.end), issue.original);
        for (const edit of issue.edits) assert.equal(state.text.slice(edit.start, edit.end), edit.original);
      }
    }
    assert.equal(state.text, "🎮 Я не знаю, почему он опоздал. Ещё текст.");
  });
}
test("Intersection returns both suggestions and keeps both highlight colors", () => {
  const state = setup(), position = state.text.indexOf("незнаю")+2;
  assert.equal(state.issuesAt(position).length, 2);
  const segment = highlightSegments(state.text, null, state.languagePending).find(s => position >= s.start && position < s.end);
  assert.deepEqual(segment.classes.sort(), ["sentence-error", "spelling"]);
});
test("Punctuation preview retains an accepted spelling fix; undo restores both", () => {
  const state = setup(), text = state.text;
  state.acceptLanguage("word1");
  assert.equal(state.languagePending.find(i => i.id === "blue").rewrite, "🎮 Я не знаю, почему он опаздал.");
  state.acceptLanguage("blue"); state.undo(); state.undo();
  assert.equal(state.text, text); assert.equal(state.languagePending.length, 3);
});
test("Applying editorial or manual changes invalidates language results; undo is coherent", () => {
  const state = setup();
  state.load([{ id: "1", start: 0, end: state.text.length, original: state.text, rewrite: "Другой текст." }]);
  state.accept("1"); assert.equal(state.languagePending.length, 0);
  state.undo(); assert.equal(state.languagePending.length, 3);
  state.manualEdit("Ещё другой текст."); assert.equal(state.languagePending.length, 0);
});
test("Stale edits and conflicting replacements cannot enter the review state", () => {
  const state = setup(), bad = structuredClone(state.issues);
  bad[0].edits[0].original = "другое"; assert.throws(() => state.loadLanguage(bad));
  const conflict = structuredClone(state.issues); conflict.push({ ...structuredClone(conflict[0]), id: "duplicate" });
  assert.throws(() => state.loadLanguage(conflict));
});
test("Undo restores the partial-check warning alongside the original highlighted issues", () => {
  const state = setup(); state.languageCheck = { partial: true, message: "Не проверено предложение 2." };
  state.load([{ id: "1", start: 0, end: state.text.length, original: state.text, rewrite: "Другой текст." }]);
  state.accept("1"); assert.equal(state.languageCheck, null);
  state.undo(); assert.equal(state.languageCheck.partial, true);
  assert.equal(state.languageCheck.message, "Не проверено предложение 2.");
});
