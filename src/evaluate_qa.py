from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any

from .agents import (
    UserProfile,
    build_agents,
    PlannerAgent,
    generic_llm_no_rag,
    single_agent_rag,
)
from .rag_store import RAGStore


QA_PATH = Path("evaluation/qa/standards_questions.jsonl")
OUT_DIR = Path("logs")


def load_questions(path: Path = QA_PATH) -> List[Dict[str, Any]]:
    questions = []

    if not path.exists():
        raise FileNotFoundError(f"QA file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            questions.append(json.loads(line))

    return questions


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
            "Building Research Establishment Environmental Assessment Method "
            "(BREEAM) Excellent, Whole Life Carbon Assessment (WLCA)-aligned"
        ),
        budget="medium",
        risk_appetite="moderate",
        extra_constraints="1960s concrete office, occupied during works",
    )


def normalise_text(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def fact_coverage(answer: str, expected_facts: List[str]) -> Dict[str, Any]:
    """
    Simple expected-fact matching benchmark.

    This is a transparent baseline:
    - recall = expected facts found / expected facts
    - precision currently approximated as recall
    - F1 from precision + recall
    - accuracy = 1 if all expected facts found
    """

    answer_norm = normalise_text(answer)
    expected_norm = [normalise_text(f) for f in expected_facts]

    found = []
    missing = []

    for raw, norm in zip(expected_facts, expected_norm):
        if norm in answer_norm:
            found.append(raw)
        else:
            missing.append(raw)

    recall = len(found) / len(expected_facts) if expected_facts else 0.0

    # Temporary simplified precision proxy
    precision = recall

    f1 = 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    accuracy = 1.0 if len(missing) == 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "found_facts": found,
        "missing_facts": missing,
    }


def answer_with_multi_agent(
    question: str,
    domain: str,
    profile: UserProfile,
    store: RAGStore,
) -> str:

    agents = build_agents(store)
    planner = PlannerAgent()

    # If question belongs to a specific domain, ask the relevant agent only
    if domain in agents:
        return agents[domain].answer(question, profile)

    # Otherwise ask all agents and consolidate
    per_agent = {
        name: agent.answer(question, profile)
        for name, agent in agents.items()
    }

    return planner.consolidate(question, profile, per_agent)


def evaluate() -> None:

    OUT_DIR.mkdir(exist_ok=True)

    questions = load_questions()

    profile = make_default_profile()

    store = RAGStore()

    rows = []

    full_records = []

    for q in questions:

        qid = q["id"]

        domain = q.get("domain", "general")

        question = q["question"]

        expected_facts = q.get("expected_facts", [])

        print(f"Evaluating {qid}: {question}")

        systems = {}

        # 1. Generic LLM
        systems["generic_llm_no_rag"] = generic_llm_no_rag(
            question,
            profile,
        )

        # 2. Single-agent RAG
        systems["single_agent_rag"] = single_agent_rag(
            question,
            profile,
            store,
        )

        # 3. GREAT multi-agent RAG
        systems["great_multi_agent_rag"] = answer_with_multi_agent(
            question=question,
            domain=domain,
            profile=profile,
            store=store,
        )

        record = {
            "id": qid,
            "domain": domain,
            "question": question,
            "expected_facts": expected_facts,
            "systems": {},
        }

        for system_name, answer in systems.items():

            metrics = fact_coverage(answer, expected_facts)

            rows.append(
                {
                    "id": qid,
                    "domain": domain,
                    "system": system_name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "found_facts": "; ".join(metrics["found_facts"]),
                    "missing_facts": "; ".join(metrics["missing_facts"]),
                }
            )

            record["systems"][system_name] = {
                "answer": answer,
                "metrics": metrics,
            }

        full_records.append(record)

    csv_path = OUT_DIR / "qa_metrics.csv"

    jsonl_path = OUT_DIR / "qa_full_outputs.jsonl"

    with csv_path.open("w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "domain",
                "system",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "found_facts",
                "missing_facts",
            ],
        )

        writer.writeheader()

        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as f:

        for record in full_records:

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved metrics to: {csv_path}")

    print(f"Saved full outputs to: {jsonl_path}")


if __name__ == "__main__":
    evaluate()