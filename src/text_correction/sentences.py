"""Russian sentence boundaries with stable IDs and Python string offsets."""

from dataclasses import dataclass
from razdel import sentenize


@dataclass(frozen=True)
class Sentence:
    sentence_id: int
    start: int
    end: int
    original: str


def split_sentences(text: str) -> list[Sentence]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be nonempty")
    result = [Sentence(i, span.start, span.stop, span.text) for i, span in enumerate(sentenize(text), 1)]
    if not result:
        raise ValueError("No sentences found")
    # Ensure no content was lost by the segmenter, including standalone headings.
    cursor = 0
    for sentence in result:
        if text[cursor:sentence.start].strip():
            raise ValueError("Sentence segmentation left uncovered text")
        cursor = sentence.end
    if text[cursor:].strip():
        raise ValueError("Sentence segmentation left uncovered text")
    return result


def assemble_rewrite(text: str, sentences: list[dict]) -> str:
    pieces, cursor = [], 0
    for sentence in sentences:
        pieces.extend([text[cursor:sentence["start"]], sentence["rewrite"]])
        cursor = sentence["end"]
    pieces.append(text[cursor:])
    return "".join(pieces)
