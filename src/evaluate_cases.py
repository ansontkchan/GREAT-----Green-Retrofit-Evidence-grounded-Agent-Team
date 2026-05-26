from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import csv
import json
import re

from .agents import UserProfile, build_agents, PlannerAgent, generic_llm_no_rag, single_agent_rag
from .case_loader import UKGBCCase, load_all_ukgbc_cases
from .metrics import compute_overlap_metrics
from .rag_store import RAGStore


def build_profile_from_case(case: UKGBCCase) -> UserProfile:
    """
    Synthetic, controlled profile for automated case-alignment evaluation.
    Real human profiles are collected in CLI/Streamlit, not here.
    """
    return UserProfile(
        profession="asset manager",
        organisation="",
        primary_concern="energy and carbon",
        scope=f"{case.building_type} retrofit",
        location=case.location,
        timeframe="net-zero by 2040",
        time_horizon_years=20,
        standards_target=", ".join(case.goals) if case.goals else "BREEAM / WLCA aligned",
        budget="medium",
        risk_appetite="moderate",
        extra_constraints="; ".join(case.constraints) if case.constraints else "",
    )


def make_case_prompt(case: UKGBCCase) -> str:
    return (
        f"Propose a strategic green retrofit plan for this {case.building_type} in {case.location}.\n"
        f"Known goals: {', '.join(case.goals) if case.goals else 'not specified'}\n"
        f"Known constraints: {', '.join(case.constraints) if case.constraints else 'none stated'}\n\n"
        "Include key retrofit measures, phasing, and how the strategy relates to BREEAM and WLCA."
    )


def heuristic_extract_measures(text: str) -> List[str]:
    """
    MVP heuristic. For publication-quality metrics, use this as first pass then manually clean.
    """
    measures: List[str] = []
    keywords = [
        "insulation", "glazing", "window", "heat pump", "pv", "solar", "led", "lighting",
        "bms", "controls", "ventilation", "hvac", "airtightness", "fabric", "metering",
        "water", "rainwater", "reuse", "recycled", "embodied carbon", "monitoring",
    ]
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        bullet = s.startswith(("-", "*", "•")) or re.match(r"^\d+[.)]\s+", s)
        has_keyword = any(k in s.lower() for k in keywords)
        if bullet and has_keyword:
            s = re.sub(r"^[-*•]\s*", "", s)
            s = re.sub(r"^\d+[.)]\s*", "", s)
            measures.append(s[:180])
        elif has_keyword and any(v in s.lower() for v in ["install", "upgrade", "replace", "improve", "implement", "retrofit"]):
            measures.append(s[:180])

    seen = set()
    out: List[str] = []
    for m in measures:
        key = " ".join(m.lower().split())
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def run_case(case: UKGBCCase, store: RAGStore) -> Dict[str, Any]:
    profile = build_profile_from_case(case)
    question = make_case_prompt(case)
    agents = build_agents(store)
    planner = PlannerAgent()

    generic_text = generic_llm_no_rag(question, profile)
    single_text = single_agent_rag(question, profile, store)

    per_agent = {name: agent.answer(question, profile) for name, agent in agents.items()}
    planner_text = planner.consolidate(question, profile, per_agent)

    systems = {
        "generic_no_rag": generic_text,
        "single_agent_rag": single_text,
        "multi_agent_planner": planner_text,
    }

    outputs: Dict[str, Any] = {}
    for system_name, text in systems.items():
        predicted = heuristic_extract_measures(text)
        outputs[system_name] = {
            "text": text,
            "predicted_measures_raw": predicted,
            "metrics": compute_overlap_metrics(case.measures, predicted),
        }

    return {
        "case": asdict(case),
        "profile": asdict(profile),
        "question": question,
        "outputs": outputs,
        "multi_agent_per_agent": per_agent,
    }


def main() -> None:
    cases = load_all_ukgbc_cases()
    if not cases:
        print("No UKGBC cases found in cases/ukgbc/. Add JSON files first.")
        return

    store = RAGStore()
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    jsonl_path = logs_dir / f"case_eval_{ts}.jsonl"
    csv_path = logs_dir / f"case_eval_{ts}.csv"

    rows: List[List[Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for case in cases:
            print(f"\n=== Evaluating case: {case.id} | {case.title} ===")
            result = run_case(case, store)
            jf.write(json.dumps(result) + "\n")
            for system_name, payload in result["outputs"].items():
                m = payload["metrics"]
                rows.append([
                    case.id, system_name, m["precision"], m["recall"], m["f1"], m["jaccard"],
                    m["ref_count"], m["pred_count"], m["overlap_count"],
                ])
                print(
                    f"{system_name:22s} P={m['precision']:.2f} R={m['recall']:.2f} "
                    f"F1={m['f1']:.2f} J={m['jaccard']:.2f} "
                    f"(ref={m['ref_count']}, pred={m['pred_count']}, overlap={m['overlap_count']})"
                )

    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["case_id", "system", "precision", "recall", "f1", "jaccard", "ref_count", "pred_count", "overlap_count"])
        writer.writerows(rows)

    print(f"\nSaved detailed JSONL: {jsonl_path}")
    print(f"Saved summary CSV:   {csv_path}")
    print("Note: heuristic measure extraction is for MVP debugging. Manually clean extracted measures before final paper metrics.")


if __name__ == "__main__":
    main()
