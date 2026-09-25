from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .agents import (
    UserProfile,
    build_agents,
    PlannerAgent,
    generic_llm_no_rag,
    single_agent_rag,
)
from .rag_store import RAGStore


# ============================================================
# FINAL EXPERIMENT CONFIGURATION
# ============================================================

QA_PATH = Path("evaluation/qa/standards_questions_48.jsonl")
OUT_DIR = Path("logs")

# One execution = one experimental run.
# Use run_01 ... run_05 for the five repeated runs.
RUN_ID = "run_01"

# Smoke test first. Set to False only after the 9-response test passes.
PILOT_MODE = False
PILOT_IDS = [
    "breeam_001",
    "wlca_008",
    "energy_010",
]

SYSTEMS = [
    "generic_llm_no_rag",
    "single_agent_rag",
    "great_multi_agent_rag",
]


# ============================================================
# BENCHMARK LOADING AND VALIDATION
# ============================================================

def load_questions(path: Path = QA_PATH) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []

    if not path.exists():
        raise FileNotFoundError(f"QA benchmark not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

    return questions


def validate_benchmark(questions: List[Dict[str, Any]]) -> None:
    if len(questions) != 48:
        raise ValueError(
            f"Expected exactly 48 benchmark questions; found {len(questions)}."
        )

    required = [
        "id",
        "domain",
        "question_type",
        "difficulty",
        "question",
        "expected_facts",
        "forbidden_facts",
        "source_doc_id",
        "source_pages",
        "scoring_note",
    ]

    ids = set()
    for index, question in enumerate(questions, start=1):
        for field in required:
            if field not in question:
                raise ValueError(
                    f"Question {index} ({question.get('id', '?')}) "
                    f"is missing required field: {field}"
                )

        qid = question["id"]
        if qid in ids:
            raise ValueError(f"Duplicate question ID: {qid}")
        ids.add(qid)

        if not isinstance(question["expected_facts"], list):
            raise ValueError(f"expected_facts must be a list: {qid}")
        if not isinstance(question["forbidden_facts"], list):
            raise ValueError(f"forbidden_facts must be a list: {qid}")


def select_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not PILOT_MODE:
        return questions

    lookup = {q["id"]: q for q in questions}
    missing = [qid for qid in PILOT_IDS if qid not in lookup]
    if missing:
        raise ValueError(f"Pilot question IDs not found: {missing}")

    return [lookup[qid] for qid in PILOT_IDS]


# ============================================================
# FIXED USER CONTEXT
# ============================================================

def make_default_profile() -> UserProfile:
    return UserProfile(
        profession="Area Chief Engineer",
        organisation="",
        primary_concern="energy and carbon savings",
        scope="single office building",
        location="London, UK",
        timeframe="net-zero by 2050",
        time_horizon_years=20,
        standards_target=(
            "Building Research Establishment Environmental Assessment "
            "Method (BREEAM) Excellent, Whole Life Carbon Assessment "
            "(WLCA)-aligned"
        ),
        budget="medium",
        risk_appetite="moderate",
        extra_constraints="1960s concrete office, occupied during works",
    )


# ============================================================
# DETERMINISTIC SEMANTIC-LITE FACT SCORING
# ============================================================
#
# The benchmark's expected_facts are the scoring units. We do not
# use an additional LLM judge in the 720-response experiment.
# Instead, this scorer normalises terminology and checks whether
# the answer contains the core concepts required by each fact.
#
# Scores:
#   0 = absent / incorrect
#   1 = partial or incomplete support
#   2 = clear support
#
# FactScore = sum(scores) / (2 * number of expected facts)
#
# This is intentionally transparent and reproducible. A small
# human-checked subset can later be reported as a validation step.

SYNONYMS: Dict[str, List[str]] = {
    "whole life carbon": [
        "whole life carbon",
        "whole-life carbon",
        "whole lifecycle carbon",
        "whole life-cycle carbon",
        "wlc",
        "wlca",
    ],
    "operational carbon": [
        "operational carbon",
        "operational emissions",
        "use stage carbon",
        "carbon from operation",
    ],
    "embodied carbon": [
        "embodied carbon",
        "embodied emissions",
        "carbon embodied in materials",
        "material-related carbon",
    ],
    "life cycle": [
        "life cycle",
        "lifecycle",
        "life-cycle",
        "whole life",
    ],
    "like-for-like": [
        "like for like",
        "like-for-like",
        "common basis",
        "same basis",
        "consistent basis",
        "equivalent basis",
    ],
    "consistent assumptions": [
        "consistent assumptions",
        "same assumptions",
        "equivalent assumptions",
        "common assumptions",
        "consistent methodology",
    ],
    "consistent data": [
        "consistent data",
        "same data basis",
        "comparable data",
        "common data",
    ],
    "equivalent scope": [
        "equivalent scope",
        "same scope",
        "consistent scope",
        "same boundary",
        "consistent boundary",
        "common boundary",
    ],
    "energy performance": [
        "energy performance",
        "energy efficiency",
        "energy use",
        "energy consumption",
    ],
    "operational costs": [
        "operational costs",
        "running costs",
        "energy costs",
        "operating costs",
    ],
    "carbon benefits": [
        "carbon benefits",
        "carbon reduction",
        "lower carbon",
        "emissions reduction",
    ],
    "breeam sustainability objectives": [
        "breeam sustainability objectives",
        "breeam objectives",
        "breeam requirements",
        "breeam credits",
        "breeam performance",
    ],
    "evidence supports assessment": [
        "evidence supports assessment",
        "evidence supports the assessment",
        "supports assessment",
        "assessment evidence",
    ],
    "evidence supports verification": [
        "evidence supports verification",
        "supports verification",
        "verification evidence",
        "verification records",
    ],
    "evidence demonstrates compliance": [
        "evidence demonstrates compliance",
        "demonstrates compliance",
        "compliance evidence",
        "proof of compliance",
    ],
    "life cycle cost": [
        "life cycle cost",
        "life-cycle cost",
        "lifecycle cost",
        "life cycle costing",
        "lcc",
    ],
    "capital cost": [
        "capital cost",
        "upfront cost",
        "initial cost",
        "initial capital",
    ],
    "professional involvement": [
        "relevant professionals",
        "professional involvement",
        "specialists involved",
        "multidisciplinary team",
    ],
    "design flexibility": [
        "design flexibility",
        "flexibility in design",
        "design options",
        "keep options open",
    ],
    "value engineering": [
        "value engineering",
        "value management",
    ],
    "project quantities": [
        "project quantities",
        "quantities",
        "quantity information",
        "quantity take-offs",
        "quantity takeoffs",
    ],
    "quantity surveyor": [
        "quantity surveyor",
        "qs",
    ],
    "post-construction": [
        "post-construction",
        "post construction",
        "as-built",
        "after construction",
    ],
    "operational carbon reduction": [
        "reduce operational carbon",
        "operational carbon reduction",
        "lower operational emissions",
    ],
    "additional materials": [
        "additional materials",
        "additional products",
        "more material",
        "material use",
    ],
    "long-term period": [
        "long-term",
        "long term",
        "over 20 years",
        "20 years",
        "40 years",
        "60 years",
    ],
}


def normalise(text: str) -> str:
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9%\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def phrase_variants(fact: str) -> List[str]:
    fact_norm = normalise(fact)
    variants = [fact_norm]

    for key, values in SYNONYMS.items():
        if normalise(key) in fact_norm:
            variants.extend(normalise(value) for value in values)

    return list(dict.fromkeys(variants))


def token_overlap(answer: str, fact: str) -> float:
    answer_tokens = set(normalise(answer).split())
    fact_tokens = set(normalise(fact).split())
    if not fact_tokens:
        return 0.0
    return len(answer_tokens & fact_tokens) / len(fact_tokens)


def score_fact(
    answer: str,
    fact: str,
    scoring_note: str = "",
) -> Tuple[int, str]:
    answer_norm = normalise(answer)
    variants = phrase_variants(fact)

    if any(variant and variant in answer_norm for variant in variants):
        return 2, "clear phrase or recognised equivalent"

    overlap = token_overlap(answer, fact)
    if overlap >= 0.60:
        return 2, "strong concept overlap"
    if overlap >= 0.30:
        return 1, "partial concept overlap"

    # A scoring note often contains explicit semantic equivalents.
    note_norm = normalise(scoring_note)
    fact_terms = [
        term for term in normalise(fact).split()
        if len(term) > 3
    ]
    if fact_terms and sum(term in note_norm for term in fact_terms) >= max(1, len(fact_terms) // 2):
        if any(term in answer_norm for term in fact_terms):
            return 1, "partial support consistent with scoring note"

    return 0, "no sufficient support detected"


def semantic_fact_score(
    answer: str,
    expected_facts: List[str],
    scoring_note: str = "",
) -> Dict[str, Any]:
    fact_scores = []
    total = 0

    for fact in expected_facts:
        score, reason = score_fact(answer, fact, scoring_note)
        fact_scores.append({
            "fact": fact,
            "score": score,
            "reason": reason,
        })
        total += score

    denominator = 2 * len(expected_facts)
    fact_score = total / denominator if denominator else 0.0

    clear = sum(item["score"] == 2 for item in fact_scores)
    partial = sum(item["score"] == 1 for item in fact_scores)
    missing = sum(item["score"] == 0 for item in fact_scores)

    return {
        "fact_score": fact_score,
        "fact_scores": fact_scores,
        "clear_facts": clear,
        "partial_facts": partial,
        "missing_fact_count": missing,
    }


def forbidden_fact_check(
    answer: str,
    forbidden_facts: List[str],
) -> Dict[str, Any]:
    answer_norm = normalise(answer)
    violations = []

    for fact in forbidden_facts:
        variants = phrase_variants(fact)
        violated = any(v and v in answer_norm for v in variants)
        if violated:
            violations.append({
                "fact": fact,
                "violated": True,
                "reason": "forbidden phrase detected",
            })

    return {
        "forbidden_violation_count": len(violations),
        "forbidden_violations": violations,
    }


# ============================================================
# SYSTEM EXECUTION
# ============================================================

def run_generic(
    question: str,
    profile: UserProfile,
) -> Dict[str, Any]:
    answer = generic_llm_no_rag(question, profile)
    return {
        "answer": answer,
        "retrieved": [],
        "collection": None,
    }


def run_single(
    question: str,
    profile: UserProfile,
    store: RAGStore,
) -> Dict[str, Any]:
    try:
        from .agents import single_agent_rag_with_trace

        result = single_agent_rag_with_trace(
            question,
            profile,
            store,
        )
        return result

    except ImportError:
        answer = single_agent_rag(
            question,
            profile,
            store,
        )
        return {
            "answer": answer,
            "retrieved": [],
            "collection": "ALL",
        }


def run_great(
    question: str,
    profile: UserProfile,
    store: RAGStore,
) -> Dict[str, Any]:
    agents = build_agents(store)
    specialist_answers: Dict[str, str] = {}
    specialist_traces: Dict[str, Any] = {}

    for name, agent in agents.items():
        if hasattr(agent, "answer_with_trace"):
            result = agent.answer_with_trace(
                question,
                profile,
            )
            specialist_answers[name] = result["answer"]
            specialist_traces[name] = result
        else:
            specialist_answers[name] = agent.answer(
                question,
                profile,
            )
            specialist_traces[name] = {
                "agent": name,
                "answer": specialist_answers[name],
                "retrieved": [],
            }

    planner = PlannerAgent()
    final_answer = planner.consolidate(
        question,
        profile,
        specialist_answers,
    )

    # Flatten specialist retrieval traces for evaluation.
    # Each specialist contributes SPECIALIST_TOP_K retrieved chunks.
    retrieved = []

    for name, trace in specialist_traces.items():
        for item in trace.get("retrieved", []):
            item_with_agent = dict(item)
            item_with_agent["agent"] = name
            retrieved.append(item_with_agent)

    return {
        "answer": final_answer,
        "specialist_answers": specialist_answers,
        "specialist_traces": specialist_traces,
        "retrieved": retrieved,
        "collection": "SPECIALIST_AGENTS",
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate() -> None:
    print("=" * 72)
    print("GREAT FINAL EVALUATION RUNNER")
    print("=" * 72)
    print(f"Benchmark : {QA_PATH}")
    print(f"Run ID    : {RUN_ID}")
    print(f"Mode      : {'PILOT' if PILOT_MODE else 'FULL'}")
    print()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions = load_questions()
    validate_benchmark(questions)
    selected = select_questions(questions)

    print(
        f"Benchmark validation: PASS ({len(questions)} questions)"
    )
    print(
        f"Questions selected : {len(selected)}"
    )
    print(
        f"Responses expected : {len(selected) * len(SYSTEMS)}"
    )
    print()

    profile = make_default_profile()
    store = RAGStore()

    csv_rows: List[Dict[str, Any]] = []
    full_records: List[Dict[str, Any]] = []

    for question_index, q in enumerate(
        selected,
        start=1,
    ):
        qid = q["id"]
        question = q["question"]
        expected_facts = q["expected_facts"]
        forbidden_facts = q["forbidden_facts"]
        scoring_note = q.get("scoring_note", "")

        print("-" * 72)
        print(
            f"[{question_index}/{len(selected)}] {qid}"
        )
        print(question)

        system_results: Dict[str, Any] = {}

        print("  1/3 Generic LLM")
        system_results["generic_llm_no_rag"] = run_generic(
            question,
            profile,
        )

        print("  2/3 Single-agent RAG")
        system_results["single_agent_rag"] = run_single(
            question,
            profile,
            store,
        )

        print("  3/3 GREAT multi-agent RAG")
        system_results["great_multi_agent_rag"] = run_great(
            question,
            profile,
            store,
        )

        record = {
            "run_id": RUN_ID,
            "question_index": question_index,
            "id": qid,
            "domain": q["domain"],
            "question_type": q["question_type"],
            "difficulty": q["difficulty"],
            "question": question,
            "expected_facts": expected_facts,
            "forbidden_facts": forbidden_facts,
            "source_doc_id": q["source_doc_id"],
            "source_pages": q["source_pages"],
            "scoring_note": scoring_note,
            "systems": {},
        }

        for system_name, result in system_results.items():
            answer = result.get(
                "answer",
                "",
            )

            semantic = semantic_fact_score(
                answer,
                expected_facts,
                scoring_note,
            )

            forbidden = forbidden_fact_check(
                answer,
                forbidden_facts,
            )

            retrieval = result.get(
                "retrieved",
                [],
            ) or []

            unique_sources = set()

            for item in retrieval:
                source = (
                    item.get("source")
                    or item.get("source_doc_id")
                )

                if source:
                    unique_sources.add(
                        str(source)
                    )

            row = {
                "run_id": RUN_ID,
                "question_index": question_index,
                "id": qid,
                "domain": q["domain"],
                "question_type": q["question_type"],
                "difficulty": q["difficulty"],
                "system": system_name,
                "fact_score": semantic["fact_score"],
                "clear_facts": semantic["clear_facts"],
                "partial_facts": semantic["partial_facts"],
                "missing_fact_count": semantic["missing_fact_count"],
                "forbidden_violation_count": (
                    forbidden[
                        "forbidden_violation_count"
                    ]
                ),
                "retrieved_count": len(retrieval),
                "unique_source_count": len(
                    unique_sources
                ),
            }

            csv_rows.append(row)

            record["systems"][system_name] = {
                "answer": answer,
                "semantic_score": semantic,
                "forbidden_check": forbidden,
                "retrieved": retrieval,
                "collection": result.get(
                    "collection"
                ),
                "specialist_answers": result.get(
                    "specialist_answers"
                ),
                "specialist_traces": result.get(
                    "specialist_traces"
                ),
            }

            print(
                f"      {system_name}: "
                f"FactScore="
                f"{semantic['fact_score']:.3f}, "
                f"forbidden="
                f"{forbidden['forbidden_violation_count']}"
            )

        full_records.append(record)

    metrics_path = (
        OUT_DIR
        / f"qa_semantic_metrics_{RUN_ID}.csv"
    )

    outputs_path = (
        OUT_DIR
        / f"qa_semantic_outputs_{RUN_ID}.jsonl"
    )

    fieldnames = [
        "run_id",
        "question_index",
        "id",
        "domain",
        "question_type",
        "difficulty",
        "system",
        "fact_score",
        "clear_facts",
        "partial_facts",
        "missing_fact_count",
        "forbidden_violation_count",
        "retrieved_count",
        "unique_source_count",
    ]

    with metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    with outputs_path.open(
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

    print()
    print("=" * 72)
    print("RUN COMPLETE")
    print("=" * 72)
    print(
        f"Responses generated : {len(csv_rows)}"
    )
    print(
        f"Metrics             : {metrics_path}"
    )
    print(
        f"Full outputs        : {outputs_path}"
    )
    print()


if __name__ == "__main__":
    evaluate()