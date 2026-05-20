from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class JurisdictionRule:
    lo: float
    hi: float
    superseded_after: date | None = None
    superseded_by: str | None = None


@dataclass
class RangeTable:
    _rules: dict[str, JurisdictionRule]

    @classmethod
    def from_dict(cls, data: dict) -> RangeTable:
        rules: dict[str, JurisdictionRule] = {}
        for jur, entry in data.items():
            superseded_after = None
            if entry.get("superseded_after"):
                superseded_after = date.fromisoformat(entry["superseded_after"])
            rules[jur] = JurisdictionRule(
                lo=entry["lo"],
                hi=entry["hi"],
                superseded_after=superseded_after,
                superseded_by=entry.get("superseded_by"),
            )
        return cls(_rules=rules)

    @classmethod
    def from_json_file(cls, path: Path) -> RangeTable:
        raw = json.loads(Path(path).read_text())
        return cls.from_dict(raw)

    def get_range(self, jurisdiction: str) -> tuple[float, float]:
        rule = self._rules[jurisdiction]
        return rule.lo, rule.hi

    def has_superseding_rule(self, jurisdiction: str, after: date, before: date) -> bool:
        rule = self._rules.get(jurisdiction)
        if rule is None or rule.superseded_after is None:
            return False
        return after < rule.superseded_after <= before

    def known_jurisdictions(self) -> frozenset[str]:
        return frozenset(self._rules.keys())
