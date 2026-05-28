from __future__ import annotations

import json
from pathlib import Path


QA_PATH = Path("evaluation/qa/standards_questions.jsonl")

REQUIRED_FIELDS = [
    "id",
    "domain",
    "question_type",
    "difficulty",
    "question",
    "expected_facts",
    "forbidden_facts",
    "source_doc_id",
    "source_pages",
    "source_note",
    "scoring_note",
    "evidence_status",
]

VALID_DOMAINS = {"breeam", "wlca", "energy", "cost"}
VALID_QUESTION_TYPES = {
    "definition",
    "enumeration",
    "scope",
    "process",
    "tradeoff",
    "compliance",
    "application",
    "concept",
}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_EVIDENCE_STATUS = {
    "pilot_unverified",
    "source_verified",
    "expert_checked",
}


def main() -> None:
    if not QA_PATH.exists():
        raise FileNotFoundError(f"Cannot find {QA_PATH}")

    seen_ids = set()
    total = 0
    warnings = 0

    with QA_PATH.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            total += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            missing = [field for field in REQUIRED_FIELDS if field not in item]
            if missing:
                raise ValueError(
                    f"Line {line_number}, item {item.get('id', '<no id>')} "
                    f"is missing fields: {missing}"
                )

            item_id = item["id"]

            if item_id in seen_ids:
                raise ValueError(f"Duplicate id found: {item_id}")

            seen_ids.add(item_id)

            if item["domain"] not in VALID_DOMAINS:
                raise ValueError(f"{item_id}: invalid domain: {item['domain']}")

            if item["question_type"] not in VALID_QUESTION_TYPES:
                raise ValueError(
                    f"{item_id}: invalid question_type: {item['question_type']}"
                )

            if item["difficulty"] not in VALID_DIFFICULTIES:
                raise ValueError(
                    f"{item_id}: invalid difficulty: {item['difficulty']}"
                )

            if item["evidence_status"] not in VALID_EVIDENCE_STATUS:
                raise ValueError(
                    f"{item_id}: invalid evidence_status: {item['evidence_status']}"
                )

            if not isinstance(item["expected_facts"], list) or not item["expected_facts"]:
                raise ValueError(f"{item_id}: expected_facts must be a non-empty list")

            if not isinstance(item["forbidden_facts"], list):
                raise ValueError(f"{item_id}: forbidden_facts must be a list")

            if item["source_pages"] == "TO_VERIFY":
                print(f"WARNING: {item_id} has source_pages=TO_VERIFY")
                warnings += 1

            if item["evidence_status"] == "pilot_unverified":
                print(f"WARNING: {item_id} is still pilot_unverified")
                warnings += 1

    print(f"\nValidated {total} QA items.")
    print(f"Warnings: {warnings}")
    print("Schema check completed.")


if __name__ == "__main__":
    main()