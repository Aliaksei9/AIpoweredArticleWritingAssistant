import assert from "node:assert/strict";
import { test } from "node:test";
import { LanguageUI } from "../src/ui/static/language_ui.js";

// A small DOM double tests the user-visible result without a browser or API.
function setup({ partial }) {
  const nodes = new Map();
  globalThis.document = { getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, { textContent: "", classList: { toggle() {} }, querySelector() { return this; } });
    return nodes.get(id);
  } };
  const ui = Object.create(LanguageUI.prototype);
  Object.assign(ui, { state: { text: "Тест.", languagePending: [], issues: [] }, enabled: true, status: "ready",
    partial, message: "Проверка выполнена частично: не удалось проверить предложение 2.", renderPopup() {} });
  return { ui, nodes };
}

test("Partial checking with no found fixes never claims there are no language errors", () => {
  const { ui, nodes } = setup({ partial: true }); ui.render();
  assert.match(nodes.get("language-status").textContent, /частично/);
  assert.doesNotMatch(nodes.get("language-status").textContent, /ошибок не найдено/);
  assert.equal(nodes.get("language-warning").hidden, false);
  assert.match(nodes.get("language-warning").textContent, /предложение 2/);
});
test("A fully checked text with no issues reports the successful result", () => {
  const { ui, nodes } = setup({ partial: false }); ui.render();
  assert.equal(nodes.get("language-status").textContent, "Языковых ошибок не найдено");
  assert.equal(nodes.get("language-warning").hidden, true);
});
