from __future__ import annotations

from .rag_store import RAGStore
from .agents import build_agents, PlannerAgent, generic_llm_no_rag, single_agent_rag
from .profile_assistant import collect_profile_cli


def main() -> None:
    profile = collect_profile_cli()
    query = (
        "Propose a strategic green retrofit plan for this building. Include key measures, phasing, "
        "and how they relate to BREEAM and WLCA."
    )

    store = RAGStore()
    agents = build_agents(store)
    planner = PlannerAgent()

    print("=" * 80)
    print("USER PROFILE")
    print("=" * 80)
    print(profile.to_text())

    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(query)

    print("=" * 80)
    print("BASELINE 1: Generic LLM (no RAG)")
    print("=" * 80)
    generic_answer = generic_llm_no_rag(query, profile)
    print(generic_answer)

    print("=" * 80)
    print("BASELINE 2: Single-agent RAG")
    print("=" * 80)
    single_answer = single_agent_rag(query, profile, store)
    print(single_answer)

    print("=" * 80)
    print("MULTI-AGENT RAG: specialised agents + planner")
    print("=" * 80)
    agent_answers = {}
    for name, agent in agents.items():
        print(f"\n--- Asking {name.upper()} agent ---")
        answer = agent.answer(query, profile)
        agent_answers[name] = answer
        print(answer[:900])
        print("-" * 40)

    print("\n" + "=" * 80)
    print("PLANNER / ORCHESTRATOR CONSOLIDATED PLAN")
    print("=" * 80)
    final_plan = planner.consolidate(query, profile, agent_answers)
    print(final_plan)


if __name__ == "__main__":
    main()
