from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json

CASES_DIR = Path(__file__).resolve().parents[1] / "cases" / "ukgbc"


@dataclass
class UKGBCCase:
    id: str
    title: str
    source_url: str
    building_type: str
    location: str
    construction_year: str | None
    floor_area_m2: float | None
    goals: List[str]
    constraints: List[str]
    measures: List[str]
    notes: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UKGBCCase":
        return cls(
            id=data["id"],
            title=data.get("title", ""),
            source_url=data.get("source_url", ""),
            building_type=data.get("building_type", ""),
            location=data.get("location", ""),
            construction_year=data.get("construction_year"),
            floor_area_m2=data.get("floor_area_m2"),
            goals=data.get("goals", []),
            constraints=data.get("constraints", []),
            measures=data.get("measures", []),
            notes=data.get("notes", []),
        )


def load_all_ukgbc_cases() -> List[UKGBCCase]:
    cases: List[UKGBCCase] = []
    if not CASES_DIR.exists():
        return cases
    for path in sorted(CASES_DIR.glob("*.json")):
        if path.name.startswith("example_"):
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        cases.append(UKGBCCase.from_dict(data))
    return cases


def demo_print_cases() -> None:
    cases = load_all_ukgbc_cases()
    print(f"Loaded {len(cases)} UKGBC cases")
    for c in cases:
        print(f"\n- {c.id}: {c.title}")
        print(f"  Location: {c.location}")
        print(f"  Goals: {', '.join(c.goals) if c.goals else 'none'}")
        print(f"  Measures: {', '.join(c.measures) if c.measures else 'none'}")


if __name__ == "__main__":
    demo_print_cases()
