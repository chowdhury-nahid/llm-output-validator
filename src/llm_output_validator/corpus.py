from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import Citation


@dataclass
class DocumentCorpus:
    _docs: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> DocumentCorpus:
        instance = cls()
        instance._docs = dict(data)
        return instance

    @classmethod
    def from_json_file(cls, path: Path) -> DocumentCorpus:
        raw = json.loads(Path(path).read_text())
        return cls.from_dict(raw.get("documents", raw))

    def contains(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def authority_score(self, citations: list[Citation]) -> float:
        if not citations:
            return 0.0
        weights = [self._docs[c.document_id] for c in citations if c.document_id in self._docs]
        if not weights:
            return 0.0
        return sum(weights) / len(weights)

    def missing_citations(self, citations: list[Citation]) -> list[str]:
        return [c.document_id for c in citations if c.document_id not in self._docs]
