"""Conservative prompt sizing independent of the model vendor's tokenizer."""

import json
from dataclasses import dataclass

from .errors import ContextBudgetError


def token_upper_bound(text: str) -> int:
    """UTF-8 byte count: deliberately conservative, not an exact token count."""
    return len(text.encode("utf-8"))


def request_token_bound(messages, schema) -> int:
    # Include response schema, role wrappers and room for provider formatting.
    return token_upper_bound(json.dumps(dict(messages=messages, schema=schema), ensure_ascii=False))+1024


def sentence_batches(sentences, output_tokens, batch_size):
    batches, batch, size = [], [], 0
    for sentence in sentences:
        # Both original and rewrite, plus concise recommendations and JSON keys.
        cost = 2*token_upper_bound(sentence.original)+768
        if cost > output_tokens:
            raise ContextBudgetError("A sentence is too long for the output budget; increase OPENROUTER_MAX_OUTPUT_TOKENS")
        if batch and (len(batch) >= batch_size or size+cost > output_tokens):
            batches.append(batch)
            batch, size = [], 0
        batch.append(sentence)
        size += cost
    if batch:
        batches.append(batch)
    return batches


@dataclass(frozen=True)
class PreparedBatch:
    sentences: list
    messages: list[dict]
    schema: dict
    estimated_input_tokens: int


@dataclass(frozen=True)
class AnalysisPlan:
    text: str
    topic: str
    model: str
    references: list[dict]
    batches: list[PreparedBatch]
    context: dict
    output_tokens: int

    def to_dict(self):
        def response_format(batch):
            if self.context["response_format"] == "json_object":
                return dict(type="json_object")
            return dict(type="json_schema", json_schema=dict(name="sentence_corrections", strict=True, schema=batch.schema))
        return dict(topic=self.topic, model=self.model, references=self.references, context=self.context,
            requests=[dict(model=self.model, messages=b.messages, max_tokens=self.output_tokens,
                response_format=response_format(b)) for b in self.batches])
