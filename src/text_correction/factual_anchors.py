"""Reject observed rewriting failures involving numbers, bounds and URLs."""

import re
from .errors import InvalidLLMResponse

NUMBER = r"\d+(?:[.,]\d+)?"


def _numbers(text):
    return {n.replace(",", ".") for n in re.findall(NUMBER, text)}


def _bounds(text):
    result = set()
    pattern = r"\b(не\s+менее|не\s+более|до|от|менее|более|максимум|минимум)\s+("+NUMBER+")"
    for qualifier, number in re.findall(pattern, text.casefold()):
        qualifier = " ".join(qualifier.split())
        if qualifier in ("менее", "более"):
            kind = "upper_strict" if qualifier == "менее" else "lower_strict"
        else:
            kind = "upper" if qualifier in ("до", "не более", "максимум") else "lower"
        result.add((kind, number.replace(",", ".")))
    return result


def _urls(text):
    return {url.rstrip(".,;!?)»\"'") for url in re.findall(r"https?://[^\s<>]+", text)}


def validate_rewrite_anchors(original, rewrite):
    if _numbers(original) != _numbers(rewrite):
        raise InvalidLLMResponse("Rewrite changed, removed or invented numeric facts")
    if _bounds(original) != _bounds(rewrite):
        raise InvalidLLMResponse("Rewrite changed a numerical upper/lower bound")
    if _urls(original) != _urls(rewrite):
        raise InvalidLLMResponse("Rewrite changed an original URL")
    if (re.search(r"обеща|заявля|утвержда", original, re.I)
            and not re.search(r"гарантир", original, re.I) and re.search(r"гарантир", rewrite, re.I)):
        raise InvalidLLMResponse("Rewrite turned a reported claim into a guarantee")
