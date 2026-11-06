"""Stable internal IDs built from each source's own identifiers."""

import json
from uuid import NAMESPACE_URL, UUID, uuid5


def _id(kind: str, source: str, *external_ids: str | int) -> UUID:
    if not source or not external_ids or any(str(value) == "" for value in external_ids):
        raise ValueError("source and external IDs must be nonempty")
    key = json.dumps([kind, source, *map(str, external_ids)], separators=(",", ":"))
    return uuid5(NAMESPACE_URL, f"coursework:{key}")


def resource_id_for(source: str, external_resource_id: str | int) -> UUID:
    return _id("resource", source, external_resource_id)


def article_id_for(
    source: str, external_resource_id: str | int, external_article_id: str | int
) -> UUID:
    return _id("article", source, external_resource_id, external_article_id)
