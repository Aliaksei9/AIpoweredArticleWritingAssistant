"""Independent LLM proofreading with source-owned spans and composable edits."""

from difflib import SequenceMatcher
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator

from .context import request_token_bound, sentence_batches
from .errors import ContextBudgetError, InvalidLLMResponse
from .factual_anchors import validate_rewrite_anchors
from .json_response import load_json_response
from .sentences import split_sentences

WORD = re.compile(r"[^\W_]+(?:[-‑–’'][^\W_]+)*", re.UNICODE)
REPLACEMENT = re.compile(r"[^\W_]+(?:[ \u00a0\u202f’'\-‑–][^\W_]+)*", re.UNICODE)
QUOTE_STYLE = str.maketrans({c: '"' for c in '«»“”„'})


def response_schema(count, sentence_ids=None):
    spelling = dict(type="object", additionalProperties=False,
        properties=dict(token_id=dict(type="integer", minimum=1),
                        replacement=dict(type="string", minLength=1), explanation=dict(type="string", minLength=1)),
        required=["token_id", "replacement", "explanation"])
    sentence = dict(type="object", additionalProperties=False,
        properties=dict(sentence_id=(dict(type="integer", enum=list(sentence_ids)) if sentence_ids else dict(type="integer", minimum=1)),
            spelling=dict(type="array", items=spelling), rewrite=dict(type="string", minLength=1),
            sentence_kind=dict(type="string", enum=["none", "punctuation", "grammar", "syntax"]),
            explanation=dict(type="string")),
        required=["sentence_id", "spelling", "rewrite", "sentence_kind", "explanation"])
    return dict(type="object", additionalProperties=False, required=["sentences"],
        properties=dict(sentences=dict(type="array", minItems=count, maxItems=count, items=sentence)))


def preserve_quote_style(original, rewrite):
    """A stylistic straight/curly quote conversion is not a language error."""
    pieces, cursor = [], 0
    for operation, a, b, c, d in SequenceMatcher(None, original, rewrite, autojunk=False).get_opcodes():
        if operation == "equal":
            continue
        old, new = original[a:b], rewrite[c:d]
        if old and new and old.translate(QUOTE_STYLE) == new.translate(QUOTE_STYLE):
            pieces.extend([rewrite[cursor:c], old]); cursor = d
    return "".join([*pieces, rewrite[cursor:]])


def apply_edits(text, edits):
    for edit in sorted(edits, key=lambda e: e["start"], reverse=True):
        if text[edit["start"]:edit["end"]] != edit["original"]:
            raise InvalidLLMResponse("Proofreading edit does not match the source")
        text = text[:edit["start"]]+edit["replacement"]+text[edit["end"]:]
    return text


def inspect_sentence(sentence, row):
    """Extract red word edits and blue sentence edits that never undo the red ones."""
    original = sentence.original
    words = list(WORD.finditer(original))
    spelling, seen = [], set()
    for item in row["spelling"]:
        index = item["token_id"]-1
        if index in seen or not 0 <= index < len(words):
            raise InvalidLLMResponse("Unknown or duplicate proofreading token ID")
        seen.add(index)
        token, replacement = words[index], item["replacement"].strip()
        # Some JSON-mode models wrap the spelling suggestion in display quotes.
        # The token span excludes these wrappers; never add them to the document.
        if len(replacement) > 2 and (replacement[0], replacement[-1]) in (("«", "»"), ('"', '"'), ("“", "”"), ("`", "`")):
            replacement = replacement[1:-1]
        if not replacement or not REPLACEMENT.fullmatch(replacement):
            raise InvalidLLMResponse(f"Spelling token_id={index+1}: replacement must be a word or words, without surrounding punctuation")
        if token.group() == replacement:
            continue  # A no-op is not an error and must not reject the other fixes.
        if not item["explanation"].strip():
            raise InvalidLLMResponse(f"Spelling token_id={index+1}: explain this correction")
        spelling.append(dict(start=token.start(), end=token.end(), original=token.group(),
                             replacement=replacement, explanation=item["explanation"]))
    spelling.sort(key=lambda e: e["start"])
    canonical = apply_edits(original, spelling)
    validate_rewrite_anchors(original, canonical)
    rewrite = preserve_quote_style(canonical, row["rewrite"])
    validate_rewrite_anchors(original, rewrite)
    if not rewrite.strip():
        raise InvalidLLMResponse("Empty proofreading rewrite")

    # Map boundaries in the spelling-corrected sentence back to the source.
    # Boundaries inside a replaced word are deliberately unavailable.
    boundaries, cursor, corrected_words = [0], 0, []
    for edit in spelling:
        boundaries.extend(range(cursor+1, edit["start"]+1))
        left = len(boundaries)-1
        boundaries.extend([None]*(len(edit["replacement"])-1)+[edit["end"]])
        corrected_words.append((left, len(boundaries)-1))
        cursor = edit["end"]
    boundaries.extend(range(cursor+1, len(original)+1))
    blue = []
    for operation, a, b, c, d in SequenceMatcher(None, canonical, rewrite, autojunk=False).get_opcodes():
        if operation == "equal":
            continue
        if boundaries[a] is None or boundaries[b] is None or any(a < right and b > left for left, right in corrected_words):
            raise InvalidLLMResponse("Sentence correction overlaps a spelling replacement; separate the fixes")
        start, end = boundaries[a], boundaries[b]
        blue.append(dict(start=start, end=end, original=original[start:end], replacement=rewrite[c:d]))
    kind = row["sentence_kind"]
    if not blue:
        kind = "none"  # A label without actual changes is not a correction.
    if kind == "none" and blue:
        if WORD.findall(canonical) == WORD.findall(rewrite):
            kind = "punctuation"
        else:
            raise InvalidLLMResponse("Rewrite changes words but sentence_kind is none; specify grammar or syntax and explain the changes")
    if blue and kind != "punctuation" and not row["explanation"].strip():
        raise InvalidLLMResponse("Explain the grammar or syntax correction")
    if kind == "punctuation" and WORD.findall(canonical) != WORD.findall(rewrite):
        raise InvalidLLMResponse("Punctuation correction changed words; use grammar or syntax for word changes")

    def absolute(edit):
        return {key: value for key, value in dict(edit, start=edit["start"]+sentence.start,
                                                end=edit["end"]+sentence.start).items() if key != "explanation"}

    issues = [dict(id=f"s{sentence.sentence_id}-w{i}", sentence_id=sentence.sentence_id, kind="spelling",
        start=sentence.start+edit["start"], end=sentence.start+edit["end"], original=edit["original"],
        rewrite=edit["replacement"], explanation=edit["explanation"], edits=[absolute(edit)])
        for i, edit in enumerate(spelling, 1)]
    if blue:
        issues.append(dict(id=f"s{sentence.sentence_id}-sentence", sentence_id=sentence.sentence_id, kind=kind,
            start=sentence.start, end=sentence.end, original=original, rewrite=apply_edits(original, blue),
            explanation=row["explanation"].strip() or "Исправьте знаки препинания в предложении.", edits=[absolute(edit) for edit in blue]))
    return issues


class LanguageCheck:
    """Check language only: no retrieval, database, popularity model or topic needed."""
    def __init__(self, client, *, batch_size=6):
        self.client, self.batch_size = client, batch_size
        self.prompt = (Path(__file__).with_name("prompts") / "language_check.txt").read_text(encoding="utf-8")

    def messages(self, text, sentences, feedback=None):
        data = dict(full_text=text, sentences_to_check=[dict(sentence_id=s.sentence_id, original=s.original,
            words=[dict(token_id=i, text=m.group()) for i, m in enumerate(WORD.finditer(s.original), 1)])
            for s in sentences])
        if feedback:
            data["validation_feedback"] = feedback
        return [dict(role="system", content=self.prompt+"\nJSON Schema:\n"+json.dumps(response_schema(len(sentences), [s.sentence_id for s in sentences]), ensure_ascii=False)),
                dict(role="user", content=json.dumps(data, ensure_ascii=False))]

    @staticmethod
    def inspect_parts(content, sentences, schema):
        decoded = load_json_response(content, "Proofreading response is not valid JSON")
        if not isinstance(decoded, dict) or set(decoded) != {"sentences"} or not isinstance(decoded["sentences"], list):
            raise InvalidLLMResponse("Proofreading JSON does not match its schema")
        rows = {}
        expected = {s.sentence_id for s in sentences}
        for row in decoded["sentences"]:
            identifier = row.get("sentence_id") if isinstance(row, dict) else None
            if type(identifier) is not int or identifier not in expected or identifier in rows:
                raise InvalidLLMResponse("Proofreading duplicated or invented sentence IDs")
            rows[identifier] = row
        validator = Draft202012Validator(schema["properties"]["sentences"]["items"])
        issues, valid_ids, invalid = [], [], []
        for sentence in sentences:
            row = rows.get(sentence.sentence_id)
            if row is None:
                invalid.append((sentence, "Return this missing sentence_id, including when there are no errors")); continue
            errors = list(validator.iter_errors(row))
            if errors:
                invalid.append((sentence, "Sentence fields do not match the proofreading schema")); continue
            try:
                issues.extend(inspect_sentence(sentence, row))
                valid_ids.append(sentence.sentence_id)
            except InvalidLLMResponse as error:
                invalid.append((sentence, str(error)))
        return issues, valid_ids, invalid

    @classmethod
    def inspect(cls, content, sentences, schema):
        issues, _, invalid = cls.inspect_parts(content, sentences, schema)
        if invalid:
            raise InvalidLLMResponse(invalid[0][1])
        return issues

    async def check(self, text):
        sentences = split_sentences(text)
        info = await self.client.model_info()
        output = min(self.client.config.max_output_tokens, info.max_completion_tokens)
        budget = min(self.client.config.max_input_tokens, info.context_length-output-256)
        batches = sentence_batches(sentences, output, self.batch_size)
        issues, calls, warnings, unchecked, checked_ids, validation_errors = [], [], [], [], [], []
        usage = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        costs = []
        for batch in batches:
            pending, feedback = batch, None
            for attempt in range(2):
                schema = response_schema(len(pending), [s.sentence_id for s in pending])
                messages = self.messages(text, pending, feedback)
                if request_token_bound(messages, schema) > budget:
                    raise ContextBudgetError("Full text and proofreading instructions exceed the input budget")
                completion = await self.client.complete(messages, schema, max_tokens=output)
                calls.append(dict(model=completion["model"], response_id=completion["response_id"], attempt=attempt+1,
                                  sentence_ids=[s.sentence_id for s in pending]))
                for key in usage:
                    value = completion.get("usage", {}).get(key, 0)
                    if isinstance(value, int) and value >= 0:
                        usage[key] += value
                cost = completion.get("usage", {}).get("cost")
                costs.append(cost if isinstance(cost, (int, float)) and cost >= 0 else None)
                if completion.get("fallback_from"):
                    warnings.append(f"Для проверки использована резервная бесплатная модель {completion['model']}.")
                try:
                    valid, checked, invalid = self.inspect_parts(completion["content"], pending, schema)
                except InvalidLLMResponse as error:
                    valid, checked, invalid = [], [], [(s, str(error)) for s in pending]
                issues.extend(valid); checked_ids.extend(checked)
                validation_errors.extend(dict(sentence_id=s.sentence_id, attempt=attempt+1, problem=problem) for s, problem in invalid)
                if not invalid:
                    break
                pending = [s for s, _ in invalid]
                if attempt:
                    unchecked.extend(pending)
                else:
                    feedback = [dict(sentence_id=s.sentence_id, problem=problem,
                        instruction="Исправь только это предложение. Используй точный token_id и весь текст words[token_id]. Сохрани цифры и ссылки. Если ошибок нет, верни исходное предложение без правок.")
                        for s, problem in invalid]
        if not checked_ids:
            raise InvalidLLMResponse("Proofreading could not validate any sentence after a targeted retry: "+validation_errors[-1]["problem"])
        if unchecked:
            identifiers = ", ".join(str(s.sentence_id) for s in unchecked)
            warnings.append(f"Проверка выполнена частично: не удалось проверить предложения {identifiers}. Правки к остальным предложениям доступны. Повторите проверку для оставшихся.")
        usage["cost"] = sum(costs) if all(c is not None for c in costs) else None
        issues.sort(key=lambda issue: (issue["start"], issue["end"], issue["id"]))
        return dict(issues=issues, warnings=list(dict.fromkeys(warnings)),
                    unchecked_sentences=[dict(sentence_id=s.sentence_id, start=s.start, end=s.end) for s in unchecked],
                    context=dict(calls=calls, usage=usage, sentence_count=len(sentences), batches=len(batches),
                                 checked_sentence_count=len(checked_ids), validation_errors=validation_errors))
