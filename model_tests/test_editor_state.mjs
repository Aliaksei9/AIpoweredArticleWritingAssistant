import assert from "node:assert/strict";
import { test } from "node:test";
import { ReviewState } from "../src/ui/static/editor_state.mjs";

function setup() {
  const text = "🎮 Игра вышла.\n\nВторое предложение. Ещё текст.";
  const state = new ReviewState(text);
  const parts = ["🎮 Игра вышла.", "Второе предложение.", "Ещё текст."];
  state.load(parts.map((original, index) => ({ id: String(index + 1), original,
    start: text.indexOf(original), end: text.indexOf(original) + original.length,
    rewrite: index === 0 ? "🎮 Новая игра уже доступна." : original, recommendations: [] })));
  return state;
}

test("Accepting a longer rewrite preserves every later UTF-16 sentence span", () => {
  const state = setup(); state.accept("1");
  assert.ok(state.text.startsWith("🎮 Новая игра уже доступна.\n\n"));
  for (const card of state.pending) assert.equal(state.text.slice(card.start, card.end), card.original);
  state.accept("2");
  assert.equal(state.pending.length, 1);
  assert.equal(state.text.slice(state.active.start, state.active.end), "Ещё текст.");
});
test("Reject leaves text untouched; undo restores review state and exact whitespace", () => {
  const state = setup(), original = state.text;
  state.reject("2"); assert.equal(state.text, original);
  state.accept("1"); state.undo(); assert.equal(state.text, original);
  assert.equal(state.cards[1].status, "rejected");
  state.undo(); assert.equal(state.cards[1].status, "pending");
});
test("A manual edit invalidates old recommendations instead of applying stale spans", () => {
  const state = setup(); state.manualEdit("Совсем другой текст.");
  assert.equal(state.pending.length, 0); assert.equal(state.history.length, 0);
  assert.throws(() => state.accept("1"));
});
test("Corrupt or duplicate spans cannot be loaded", () => {
  const state = setup(), bad = structuredClone(state.cards);
  bad[1].start = 0;
  assert.throws(() => state.load(bad));
  const duplicate = structuredClone(state.cards); duplicate[1].id = duplicate[0].id;
  assert.throws(() => state.load(duplicate));
});
