from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from .config import OPENAI_MODEL


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("logs/qa_full_outputs_run_01.jsonl")
OUT_DIR = Path("logs")

# Set JUDGE_MODEL in .env or the shell if you want a separate
# evaluation model. Otherwise it uses the configured model.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", OPENAI_MODEL)

# Deterministic judge setting.
JUDGE_TEMPERATURE = 0.0


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI()


# ============================================================
# JSON / TEXT HELPERS
# ============================================================

def extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a model response."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError(
            "Judge response did not contain a valid JSON object."
        )

    return json.loads(match.group(0))


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load generated benchmark outputs."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input JSONL not found: {path}"
        )

    records: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {e}"
                ) from e

    return records


def validate_records(
    records: List[Dict[str, Any]],
) -> None:
    """Validate the expected 48-question × 3-system structure."""
    if not records:
        raise ValueError("No evaluation records found.")

    if len(records) != 48:
        raise ValueError(
            f"Expected 48 question records, found {len(records)}."
        )

    ids = [r.get("id") for r in records]

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs found.")

    expected_systems = {
        "generic_llm_no_rag",
        "single_agent_rag",
        "great_multi_agent_rag",
    }

    for record in records:
        systems = set(record.get("systems", {}).keys())

        if systems != expected_systems:
            raise ValueError(
                f"Unexpected systems for {record.get('id')}: "
                f"{sorted(systems)}"
            )


# ============================================================
# SEMANTIC FACT JUDGE
# ============================================================

JUDGE_SYSTEM_PROMPT = """
You are an independent evaluator of answers to a technical
green-retrofit knowledge benchmark.

Score each expected fact against the candidate answer.

Use this strict scale for EACH expected fact:

0 = absent, contradicted, or materially incorrect.
1 = partially expressed, incomplete, or only weakly supported.
2 = clearly and correctly expressed, including valid semantic
    paraphrases. Do not require exact wording.

Important rules:
- Judge meaning, not keyword overlap.
- Do not penalise valid paraphrases.
- Do not give credit merely because a related topic is mentioned.
- Do not infer a fact that the answer does not actually communicate.
- If the answer contradicts an expected fact, score 0.
- Do not require extra details beyond the expected fact itself.
- Do not invent additional required facts.
- The benchmark expected_facts are the scoring criteria.
- Return ONLY valid JSON.
"""


def judge_facts(
    question: str,
    expected_facts: List[str],
    forbidden_facts: List[str],
    answer: str,
) -> Dict[str, Any]:
    """Score expected facts and forbidden claims."""
    user_prompt = f"""
QUESTION:
{question}

EXPECTED FACTS:
{json.dumps(expected_facts, ensure_ascii=False, indent=2)}

FORBIDDEN FACTS:
{json.dumps(forbidden_facts, ensure_ascii=False, indent=2)}

CANDIDATE ANSWER:
{answer}

Return exactly this JSON structure:
{{
  "fact_scores": [
    {{
      "fact": "exact expected fact string",
      "score": 0,
      "reason": "brief reason"
    }}
  ],
  "forbidden_violations": [
    {{
      "fact": "exact forbidden fact string",
      "violated": false,
      "reason": "brief reason"
    }}
  ]
}}

Every expected fact must appear exactly once in fact_scores.
Every forbidden fact must appear exactly once in
forbidden_violations.
"""

    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": JUDGE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=JUDGE_TEMPERATURE,
        response_format={"type": "json_object"},
    )

    return extract_json(
        completion.choices[0].message.content or ""
    )


# ============================================================
# JUDGEMENT VALIDATION
# ============================================================

def validate_judgement(
    judgement: Dict[str, Any],
    expected_facts: List[str],
    forbidden_facts: List[str],
) -> None:
    """Check that the judge returned complete valid scoring."""
    fact_scores = judgement.get("fact_scores")

    if not isinstance(fact_scores, list):
        raise ValueError(
            "Judge output missing fact_scores list."
        )

    if len(fact_scores) != len(expected_facts):
        raise ValueError(
            "Judge returned the wrong number of fact scores: "
            f"expected {len(expected_facts)}, "
            f"got {len(fact_scores)}."
        )

    returned_facts = [
        item.get("fact")
        for item in fact_scores
    ]

    if returned_facts != expected_facts:
        raise ValueError(
            "Judge fact order/content does not exactly match "
            "benchmark expected_facts."
        )

    for item in fact_scores:
        if item.get("score") not in {0, 1, 2}:
            raise ValueError(
                f"Invalid semantic fact score: "
                f"{item.get('score')}"
            )

    violations = judgement.get(
        "forbidden_violations"
    )

    if not isinstance(violations, list):
        raise ValueError(
            "Judge output missing forbidden_violations list."
        )

    if len(violations) != len(forbidden_facts):
        raise ValueError(
            "Judge returned the wrong number of "
            "forbidden-fact checks."
        )

    returned_forbidden = [
        item.get("fact")
        for item in violations
    ]

    if returned_forbidden != forbidden_facts:
        raise ValueError(
            "Judge forbidden-fact order/content does not exactly "
            "match benchmark forbidden_facts."
        )

    for item in violations:
        if not isinstance(
            item.get("violated"),
            bool,
        ):
            raise ValueError(
                "Forbidden-fact 'violated' must be boolean."
            )


# ============================================================
# SEMANTIC METRICS
# ============================================================

def calculate_semantic_metrics(
    judgement: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate transparent semantic expected-fact metrics.

    fact_score:
        sum of 0/1/2 fact scores divided by 2N.

    complete_fact_rate:
        proportion of expected facts scored 2.

    partial_fact_rate:
        proportion scored 1.

    zero_fact_rate:
        proportion scored 0.

    These metrics are NOT claim-level precision and do not
    measure evidence groundedness.
    """
    scores = [
        int(item["score"])
        for item in judgement["fact_scores"]
    ]

    n = len(scores)

    if n == 0:
        fact_score = 0.0
        complete_rate = 0.0
        partial_rate = 0.0
        zero_rate = 0.0
    else:
        fact_score = sum(scores) / (2.0 * n)
        complete_rate = scores.count(2) / n
        partial_rate = scores.count(1) / n
        zero_rate = scores.count(0) / n

    violations = [
        item["fact"]
        for item in judgement[
            "forbidden_violations"
        ]
        if item["violated"]
    ]

    return {
        "fact_score": fact_score,
        "complete_fact_rate": complete_rate,
        "partial_fact_rate": partial_rate,
        "zero_fact_rate": zero_rate,
        "forbidden_violation_count": len(
            violations
        ),
        "forbidden_violations": violations,
        "fact_scores": scores,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate() -> None:
    """Semantically score the generated outputs from one run."""
    print()
    print("=" * 70)
    print("GREAT SEMANTIC FACT EVALUATION")
    print("=" * 70)
    print(f"Input       : {INPUT_PATH}")
    print(f"Judge model : {JUDGE_MODEL}")
    print(f"Temperature : {JUDGE_TEMPERATURE}")
    print()

    records = load_records(INPUT_PATH)
    validate_records(records)

    print("Input validation: PASS")
    print(f"Questions: {len(records)}")
    print("Systems per question: 3")
    print("Answers to score: 144")
    print()

    output_records: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for question_index, record in enumerate(
        records,
        start=1,
    ):
        qid = record["id"]
        question = record["question"]
        expected_facts = record.get(
            "expected_facts",
            [],
        )
        forbidden_facts = record.get(
            "forbidden_facts",
            [],
        )

        print(
            f"[{question_index:02d}/48] "
            f"{qid}: scoring 3 systems..."
        )

        output_record = {
            "run_id": record.get("run_id"),
            "question_index": question_index,
            "id": qid,
            "domain": record.get("domain"),
            "question_type": record.get(
                "question_type"
            ),
            "difficulty": record.get(
                "difficulty"
            ),
            "question": question,
            "expected_facts": expected_facts,
            "forbidden_facts": forbidden_facts,
            "systems": {},
        }

        for system_name, system_data in (
            record["systems"].items()
        ):
            answer = system_data.get(
                "answer",
                "",
            )

            judgement = judge_facts(
                question=question,
                expected_facts=expected_facts,
                forbidden_facts=forbidden_facts,
                answer=answer,
            )

            validate_judgement(
                judgement,
                expected_facts,
                forbidden_facts,
            )

            metrics = calculate_semantic_metrics(
                judgement
            )

            output_record[
                "systems"
            ][system_name] = {
                "answer": answer,
                "judgement": judgement,
                "metrics": metrics,
            }

            rows.append(
                {
                    "run_id": record.get("run_id"),
                    "question_index": question_index,
                    "id": qid,
                    "domain": record.get(
                        "domain"
                    ),
                    "question_type": record.get(
                        "question_type"
                    ),
                    "difficulty": record.get(
                        "difficulty"
                    ),
                    "system": system_name,
                    "fact_score": metrics[
                        "fact_score"
                    ],
                    "complete_fact_rate": metrics[
                        "complete_fact_rate"
                    ],
                    "partial_fact_rate": metrics[
                        "partial_fact_rate"
                    ],
                    "zero_fact_rate": metrics[
                        "zero_fact_rate"
                    ],
                    "forbidden_violation_count": metrics[
                        "forbidden_violation_count"
                    ],
                    "forbidden_violations": "; ".join(
                        metrics[
                            "forbidden_violations"
                        ]
                    ),
                    "fact_scores": ";".join(
                        str(x)
                        for x in metrics[
                            "fact_scores"
                        ]
                    ),
                }
            )

        output_records.append(
            output_record
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_id = records[0].get(
        "run_id",
        "unknown",
    )

    csv_path = (
        OUT_DIR
        / f"qa_semantic_metrics_{run_id}.csv"
    )

    jsonl_path = (
        OUT_DIR
        / f"qa_semantic_outputs_{run_id}.jsonl"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fieldnames = [
            "run_id",
            "question_index",
            "id",
            "domain",
            "question_type",
            "difficulty",
            "system",
            "fact_score",
            "complete_fact_rate",
            "partial_fact_rate",
            "zero_fact_rate",
            "forbidden_violation_count",
            "forbidden_violations",
            "fact_scores",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in output_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 70)
    print("SEMANTIC EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Saved CSV   : {csv_path}")
    print(f"Saved JSONL : {jsonl_path}")
    print(f"Rows        : {len(rows)}")
    print()
    print(
        "fact_score = semantic expected-fact coverage."
    )
    print(
        "It is NOT claim-level precision or evidence "
        "groundedness."
    )


if __name__ == "__main__":
    evaluate()
