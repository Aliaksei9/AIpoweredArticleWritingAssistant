"""One embedding-space configuration, loaded consistently at process startup."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MODEL = 'intfloat/multilingual-e5-small'


@dataclass(frozen=True)
class EmbeddingSettings:
    model: str = DEFAULT_MODEL
    dimensions: int = 384
    document_prefix: str = 'passage'
    query_prefix: str = 'query'

    def __post_init__(self):
        if not self.model.strip():raise ValueError('EMBEDDING_MODEL must not be empty')
        if isinstance(self.dimensions,bool) or not isinstance(self.dimensions,int) or self.dimensions<1:
            raise ValueError('EMBEDDING_DIMENSIONS must be a positive integer')

    @classmethod
    def from_environment(cls,environ):
        try:dimensions=int(environ.get('EMBEDDING_DIMENSIONS','384'))
        except ValueError:raise ValueError('EMBEDDING_DIMENSIONS must be a positive integer') from None
        return cls(environ.get('EMBEDDING_MODEL',DEFAULT_MODEL).strip(),dimensions,
            environ.get('EMBEDDING_DOCUMENT_PREFIX','passage').strip(),
            environ.get('EMBEDDING_QUERY_PREFIX','query').strip())

    @property
    def signature(self):
        # Preserve compatibility with existing default-E5 artifacts and vectors.
        if self.dimensions==384 and self.document_prefix=='passage' and self.query_prefix=='query':
            return self.model
        payload=json.dumps([self.model,self.dimensions,self.document_prefix,self.query_prefix],ensure_ascii=False)
        return self.model+'#profile='+hashlib.sha256(payload.encode()).hexdigest()[:20]


load_dotenv(Path(__file__).resolve().parents[2]/'.env')
SETTINGS = EmbeddingSettings.from_environment(os.environ)
MODEL_NAME = SETTINGS.model
EMBEDDING_DIMENSIONS = SETTINGS.dimensions
EMBEDDING_SIGNATURE = SETTINGS.signature
