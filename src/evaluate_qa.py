from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .agents import (
    UserProfile,
    PlannerAgent,
    build_agents,
    generic_llm_no_rag,
    single_agent_rag_with_trace,
)
from .rag_store import RAGStore


# ============================================================
# CONFIGURATION
# ============================================================

QA_PATH = Path(
    "evaluation/qa/standards_questions_48.jsonl"
)

OUT_DIR = Path("logs")

RUN_ID = "run_01_trace"

# Keep this TRUE for the first validation run.
# It evaluates only the three pilot questions.
# Change to FALSE only after the pilot completes cleanly.
PILOT_MODE = True

PILOT_IDS = [
    "breeam_001",
    "wlca_008",
    "energy_010",
]


# ============================================================
# LOAD BENCHMARK
# ============================================================

def load_questions(
    path: Path = QA_PATH,
) -> List[Dict[str, Any]]:
    """Load the frozen 48-question JSONL benchmark."""

    if not path.exists():
        raise FileNotFoundError(
            f"QA benchmark not found: {path}"
        )

    questions: List[Dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                questions.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {e}"
                ) from e

    return questions


# ============================================================
# BENCHMARK VALIDATION
# ============================================================

def validate_benchmark(
    questions: List[Dict[str, Any]],
) -> None:
    """Validate the frozen benchmark before API calls."""

    if len(questions) != 48:
        raise ValueError(
            "Expected exactly 48 benchmark questions, "
            f"but found {len(questions)}."
        )

    required_fields = [
        "id",
        "domain",
        "question_type",
        "difficulty",
        "question",
        "expected_facts",
        "forbidden_facts",
    ]

    seen_ids = set()

    for index, question in enumerate(
        questions,
        start=1,
    ):

        for field in required_fields:
            if field not in question:
                raise ValueError(
                    f"Question {index} "
                    f"({question.get('id', '?')}) "
                    f"is missing field: {field}"
                )

        qid = question["id"]

        if qid in seen_ids:
            raise ValueError(
                f"Duplicate question ID: {qid}"
            )

        seen_ids.add(qid)

    domains = {
        question["domain"]
        for question in questions
    }

    expected_domains = {
        "breeam",
        "cost",
        "energy",
        "wlca",
    }

    if not expected_domains.issubset(domains):
        raise ValueError(
            "Benchmark is missing one or more expected "
            f"domains: {expected_domains - domains}"
        )


def select_questions(
    questions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select the pilot subset or the full benchmark."""

    if not PILOT_MODE:
        return questions

    by_id = {
        question["id"]: question
        for question in questions
    }

    missing = [
        qid
        for qid in PILOT_IDS
        if qid not in by_id
    ]

    if missing:
        raise ValueError(
            f"Pilot question IDs not found: {missing}"
        )

    return [
        by_id[qid]
        for qid in PILOT_IDS
    ]


# ============================================================
# FIXED USER PROFILE
# ============================================================

def make_default_profile() -> UserProfile:
    """Use one fixed profile across all systems."""

    return UserProfile(
        profession="Area Chief Engineer",
        organisation="",
        primary_concern="energy and carbon savings",
        scope="single office building",
        location="London, UK",
        timeframe="net-zero by 2050",
        time_horizon_years=20,
        standards_target=(
            "Building Research Establishment Environmental "
            "Assessment Method (BREEAM) Excellent, "
            "Whole Life Carbon Assessment (WLCA)-aligned"
        ),
        budget="medium",
        risk_appetite="moderate",
        extra_constraints=(
            "1960s concrete office, occupied during works"
        ),
    )


# ============================================================
# TRANSPARENT BASELINE METRICS
# ============================================================

def normalise_text(text: str) -> str:
    """Normalise text for transparent lexical matching."""

    return " ".join(
        text.lower()
        .replace("-", " ")
        .split()
    )


def fact_coverage(
    answer: str,
    expected_facts: List[str],
) -> Dict[str, Any]:
    """
    Transparent lexical baseline only.

    This is NOT the final semantic evaluation.
    It is retained so the old pilot results remain comparable.
    """

    answer_norm = normalise_text(answer)

    found: List[str] = []
    missing: List[str] = []

    for fact in expected_facts:
        fact_norm = normalise_text(fact)

        if fact_norm and fact_norm in answer_norm:
            found.append(fact)
        else:
            missing.append(fact)

    if expected_facts:
        recall = (
            len(found) / len(expected_facts)
        )
    else:
        recall = 0.0

    precision = recall

    if precision + recall > 0:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    accuracy = (
        1.0
        if expected_facts
        and len(missing) == 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "found_facts": found,
        "missing_facts": missing,
        "scoring_method": "lexical_baseline",
    }


def forbidden_fact_check(
    answer: str,
    forbidden_facts: List[str],
) -> Dict[str, Any]:
    """Detect explicit lexical matches to forbidden facts."""

    answer_norm = normalise_text(answer)

    violations: List[str] = []

    for fact in forbidden_facts:
        fact_norm = normalise_text(fact)

        if fact_norm and fact_norm in answer_norm:
            violations.append(fact)

    return {
        "violations": violations,
        "violation_count": len(violations),
    }


# ============================================================
# RETRIEVAL TRACE HELPERS
# ============================================================

def retrieval_summary(
    retrieved: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create compact retrieval metadata for CSV/analysis."""

    sources = []

    for item in retrieved:
        source = item.get("source")

        if source and source not in sources:
            sources.append(source)

    return {
        "retrieved_count": len(retrieved),
        "unique_sources": sources,
    }


def run_great_with_trace(
    question: str,
    profile: UserProfile,
    store: RAGStore,
) -> Dict[str, Any]:
    """
    Run the true GREAT condition.

    Every question is sent to all four specialists.
    The Planner then synthesises their answers.
    """

    agents = build_agents(store)

    specialist_traces: Dict[str, Any] = {}
    specialist_answers: Dict[str, str] = {}

    for name, agent in agents.items():

        print(
            f"      Specialist agent: {name}"
        )

        trace = agent.answer_with_trace(
            question,
            profile,
        )

        specialist_traces[name] = trace
        specialist_answers[name] = trace["answer"]

    print(
        "      Planner: synthesising "
        "specialist outputs"
    )

    planner = PlannerAgent()

    final_answer = planner.consolidate(
        question,
        profile,
        specialist_answers,
    )

    return {
        "answer": final_answer,
        "specialists": specialist_traces,
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate() -> None:
    """Run the controlled three-system evaluation."""

    print("=" * 70)
    print("GREAT Evaluation - Trace Logging")
    print("=" * 70)
    print(f"Benchmark : {QA_PATH}")
    print(f"Run ID    : {RUN_ID}")
    print(
        f"Pilot mode: {PILOT_MODE}"
    )
    print()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_questions = load_questions()

    validate_benchmark(
        all_questions
    )

    questions = select_questions(
        all_questions
    )

    print(
        f"Frozen benchmark : "
        f"{len(all_questions)} questions"
    )

    print(
        f"Questions this run: "
        f"{len(questions)}"
    )

    profile = make_default_profile()
    store = RAGStore()

    rows: List[Dict[str, Any]] = []
    full_records: List[Dict[str, Any]] = []

    for question_index, q in enumerate(
        questions,
        start=1,
    ):

        qid = q["id"]
        domain = q["domain"]
        question_type = q["question_type"]
        difficulty = q["difficulty"]
        question = q["question"]

        expected_facts = q["expected_facts"]
        forbidden_facts = q["forbidden_facts"]

        source_doc_id = q.get(
            "source_doc_id",
            "",
        )

        source_doc_ids = q.get(
            "source_doc_ids",
            [],
        )

        source_pages = q.get(
            "source_pages",
            "",
        )

        print()
        print("-" * 70)
        print(
            f"[{question_index}/{len(questions)}] "
            f"{qid} | {domain} | {difficulty}"
        )
        print(
            f"Question: {question}"
        )

        # ----------------------------------------------------
        # SYSTEM A: generic LLM without RAG
        # ----------------------------------------------------

        print(
            "   -> Generic LLM without RAG"
        )

        generic_answer = generic_llm_no_rag(
            question,
            profile,
        )

        # ----------------------------------------------------
        # SYSTEM B: single-agent RAG
        # ----------------------------------------------------

        print(
            "   -> Single-agent RAG"
        )

        single_trace = single_agent_rag_with_trace(
            question,
            profile,
            store,
        )

        # ----------------------------------------------------
        # SYSTEM C: GREAT multi-agent RAG
        # ----------------------------------------------------

        print(
            "   -> GREAT multi-agent RAG"
        )

        great_trace = run_great_with_trace(
            question,
            profile,
            store,
        )

        system_traces: Dict[str, Dict[str, Any]] = {
            "generic_llm_no_rag": {
                "answer": generic_answer,
                "retrieved": [],
            },
            "single_agent_rag": {
                "answer": single_trace["answer"],
                "retrieved": single_trace[
                    "retrieved"
                ],
            },
            "great_multi_agent_rag": {
                "answer": great_trace["answer"],
                "specialists": great_trace[
                    "specialists"
                ],
            },
        }

        record: Dict[str, Any] = {
            "run_id": RUN_ID,
            "question_index": question_index,
            "id": qid,
            "domain": domain,
            "question_type": question_type,
            "difficulty": difficulty,
            "question": question,
            "expected_facts": expected_facts,
            "forbidden_facts": forbidden_facts,
            "source_doc_id": source_doc_id,
            "source_doc_ids": source_doc_ids,
            "source_pages": source_pages,
            "systems": {},
        }

        # ----------------------------------------------------
        # Score and save each system
        # ----------------------------------------------------

        for system_name, trace in system_traces.items():

            answer = trace["answer"]

            coverage = fact_coverage(
                answer,
                expected_facts,
            )

            forbidden = forbidden_fact_check(
                answer,
                forbidden_facts,
            )

            if system_name == "single_agent_rag":

                retrieval = trace[
                    "retrieved"
                ]

                retrieval_meta = retrieval_summary(
                    retrieval
                )

            elif system_name == "great_multi_agent_rag":

                retrieval = []

                for specialist in trace[
                    "specialists"
                ].values():

                    retrieval.extend(
                        specialist.get(
                            "retrieved",
                            [],
                        )
                    )

                retrieval_meta = retrieval_summary(
                    retrieval
                )

            else:

                retrieval = []

                retrieval_meta = {
                    "retrieved_count": 0,
                    "unique_sources": [],
                }

            rows.append(
                {
                    "run_id": RUN_ID,
                    "question_index": question_index,
                    "id": qid,
                    "domain": domain,
                    "question_type": question_type,
                    "difficulty": difficulty,
                    "system": system_name,
                    "accuracy": coverage[
                        "accuracy"
                    ],
                    "precision": coverage[
                        "precision"
                    ],
                    "recall": coverage[
                        "recall"
                    ],
                    "f1": coverage[
                        "f1"
                    ],
                    "forbidden_violation_count": (
                        forbidden[
                            "violation_count"
                        ]
                    ),
                    "retrieved_count": (
                        retrieval_meta[
                            "retrieved_count"
                        ]
                    ),
                    "unique_source_count": len(
                        retrieval_meta[
                            "unique_sources"
                        ]
                    ),
                    "found_facts": "; ".join(
                        coverage[
                            "found_facts"
                        ]
                    ),
                    "missing_facts": "; ".join(
                        coverage[
                            "missing_facts"
                        ]
                    ),
                    "forbidden_violations": (
                        "; ".join(
                            forbidden[
                                "violations"
                            ]
                        )
                    ),
                }
            )

            record["systems"][system_name] = {
                "answer": answer,
                "metrics": {
                    "fact_coverage": coverage,
                    "forbidden_facts": forbidden,
                    "retrieval_summary": retrieval_meta,
                },
            }

            if system_name == "single_agent_rag":
                record["systems"][
                    system_name
                ]["retrieval_trace"] = retrieval

            if system_name == "great_multi_agent_rag":
                record["systems"][
                    system_name
                ]["specialist_traces"] = trace[
                    "specialists"
                ]

        full_records.append(
            record
        )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    csv_path = (
        OUT_DIR
        / f"qa_metrics_{RUN_ID}.csv"
    )

    jsonl_path = (
        OUT_DIR
        / f"qa_full_outputs_{RUN_ID}.jsonl"
    )

    fieldnames = [
        "run_id",
        "question_index",
        "id",
        "domain",
        "question_type",
        "difficulty",
        "system",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "forbidden_violation_count",
        "retrieved_count",
        "unique_source_count",
        "found_facts",
        "missing_facts",
        "forbidden_violations",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

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

        for record in full_records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(
        f"Frozen questions     : "
        f"{len(all_questions)}"
    )
    print(
        f"Questions evaluated  : "
        f"{len(questions)}"
    )
    print(
        "Systems evaluated    : 3"
    )
    print(
        "Responses generated  : "
        f"{len(questions) * 3}"
    )
    print(
        f"Run ID               : {RUN_ID}"
    )
    print()
    print(
        f"Saved metrics to:\n  {csv_path}"
    )
    print(
        f"Saved full traces to:\n  {jsonl_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    evaluate()