"""Platform-independent resource identities and audience metadata."""

from dataclasses import dataclass
import math


def normalize_source(source: str) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Source must be a nonempty string")
    source = source.strip().upper()
    return {"TELEGRAM": "TG", "ВКОНТАКТЕ": "VK", "TWITTER": "X"}.get(source, source)


@dataclass(frozen=True)
class ResourceInfo:
    id: str
    name: str
    audience: int
    source: str = "TG"

    def __post_init__(self):
        if not self.id or not self.name or isinstance(self.audience, bool):
            raise ValueError("A resource requires an identity, name and positive audience")
        try:
            valid = math.isfinite(self.audience) and self.audience > 0 and int(self.audience) == self.audience
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("Resource audience must be a positive integer")
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "source", normalize_source(self.source))

    @classmethod
    def from_database(cls, row):
        return cls(str(row["id"]), row["name"], row["reader_count"], row["source"])


def resolve_resource(resources, identifier, source):
    source = normalize_source(source)
    if isinstance(identifier, ResourceInfo):
        if identifier.source != source:
            raise ValueError("Resource source does not match the requested source")
        return identifier
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Resource must be a UUID, exact name or ResourceInfo")
    key = identifier.strip().lstrip("@").casefold()
    matches = [r for r in resources if r.source == source
               and (r.id.casefold() == key or r.name.strip().casefold() == key)]
    if len(matches) != 1:
        raise ValueError("Resource is unknown or ambiguous; use its UUID or supply ResourceInfo")
    return matches[0]
