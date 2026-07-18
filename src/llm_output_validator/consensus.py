from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ConsensusClaim:
    consensus_value: float | str
    agreement_count: int
    total_sources: int
    critical: bool = True
    tolerance: float = 0.001

    def matches(self, value: float | str) -> bool:
        if isinstance(self.consensus_value, (int, float)) and isinstance(value, (int, float)):
            return abs(float(value) - float(self.consensus_value)) <= self.tolerance
        return str(value) == str(self.consensus_value)


@dataclass
class ConsensusReference:
    question_id: str
    jurisdiction: str
    sources: list[dict[str, object]] = field(default_factory=list)
    consensus: dict[str, ConsensusClaim] = field(default_factory=dict)

    @classmethod
    def from_json_file(cls, path: Path) -> ConsensusReference:
        raw = json.loads(Path(path).read_text())
        consensus: dict[str, ConsensusClaim] = {}
        for key, entry in raw.get("consensus", {}).items():
            consensus[key] = ConsensusClaim(
                consensus_value=entry["consensus_value"],
                agreement_count=entry["agreement_count"],
                total_sources=entry["total_sources"],
                critical=entry.get("critical", True),
                tolerance=entry.get("tolerance", 0.001),
            )
        return cls(
            question_id=raw["question_id"],
            jurisdiction=raw["jurisdiction"],
            sources=raw.get("sources", []),
            consensus=consensus,
        )

    @classmethod
    def load_from_dir(cls, directory: Path) -> list[ConsensusReference]:
        refs: list[ConsensusReference] = []
        for path in sorted(Path(directory).glob("*.json")):
            refs.append(cls.from_json_file(path))
        return refs
