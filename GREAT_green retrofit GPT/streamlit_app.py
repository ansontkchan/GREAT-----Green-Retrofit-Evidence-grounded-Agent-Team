from __future__ import annotations

import streamlit as st

from src.agents import UserProfile, build_agents, PlannerAgent, generic_llm_no_rag, single_agent_rag
from src.rag_store import RAGStore


@st.cache_resource
def get_runtime():
    store = RAGStore()
    agents = build_agents(store)
    planner = PlannerAgent()
    return store, agents, planner


def main() -> None:
    st.set_page_config(page_title="GREAT", layout="wide")
    st.title("GREAT: Green Retrofit Evidence-grounded Agent Team")
    st.caption("Benchmarking Multi-agent RAG System for Green Retrofit Planning in 3 LLM-based systems: (1) generic LLM without RAG, (2) single-agent with RAG, and (3) multi-agent with RAG.")

    with st.sidebar:
        st.header("User/context profile")
        profession = st.text_input("Profession / Role", "Area Chief Engineer")
        location = st.text_input("Location", "London, UK")
        primary_concern = st.text_input("Primary concern", "energy and carbon savings")
        scope = st.text_input("Scope", "Single office building")
        timeframe = st.text_input("Timeframe", "net-zero by 2050")
        time_horizon_years = st.number_input("Planning horizon (years)", min_value=1, max_value=80, value=20)
        breeam_rating = st.selectbox(
            "BREEAM rating appetite",
            ["Outstanding", "Excellent", "Very Good", "Good", "Pass", "Unclassified", "No specific rating"],
            index=1,
        )
        standards_target = "No specific BREEAM rating target" if breeam_rating == "No specific rating" else f"BREEAM {breeam_rating}, WLCA-aligned"
        budget = st.selectbox("Budget", ["low", "medium", "high"], index=1)
        risk_appetite = st.selectbox("Risk appetite", ["low", "moderate", "high"], index=1)
        extra_constraints = st.text_area("Extra constraints", "1960s concrete office, occupied during works")

    profile = UserProfile(
        profession=profession,
        organisation="",
        primary_concern=primary_concern,
        scope=scope,
        location=location,
        timeframe=timeframe,
        time_horizon_years=int(time_horizon_years),
        standards_target=standards_target,
        budget=budget,
        risk_appetite=risk_appetite,
        extra_constraints=extra_constraints,
    )

    st.subheader("Question")
    question = st.text_area(
        "Ask GREAT",
        value="Propose a strategic green retrofit plan for this building. Include key retrofit measures, phasing. Give justifications on the proposed measure relate to sustainable building certification principles outlined in BREEAM and the whole life carbon assessment (WLCA) framework.",
        height=130,
    )

    mode = st.radio(
        "System mode",
        ["All three", "Generic LLM (no RAG)", "Single-agent with RAG", "Multi-agent with RAG"],
        horizontal=True,
    )

    with st.expander("Current profile sent to the agents"):
        st.code(profile.to_text())

    if st.button("Run", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        store, agents, planner = get_runtime()
        show_generic = mode in {"All three", "Generic LLM (no RAG)"}
        show_single = mode in {"All three", "Single-agent with RAG"}
        show_multi = mode in {"All three", "Multi-agent with RAG"}

        if show_generic:
            with st.spinner("Running generic LLM..."):
                ans = generic_llm_no_rag(question, profile)
            st.markdown("## Baseline 1: Generic LLM (no RAG)")
            st.write(ans)

        if show_single:
            with st.spinner("Running single-agent RAG..."):
                ans = single_agent_rag(question, profile, store)
            st.markdown("## Baseline 2: Single-agent with RAG")
            st.write(ans)

        if show_multi:
            with st.spinner("Running multi-agent with RAG..."):
                per_agent = {name: agent.answer(question, profile) for name, agent in agents.items()}
                final = planner.consolidate(question, profile, per_agent)
            st.markdown("## Multi-agent RAG: output")
            st.write(final)
            with st.expander("Show specialised agent outputs"):
                for name, text in per_agent.items():
                    st.markdown(f"### {name.capitalize()} agent")
                    st.write(text)


if __name__ == "__main__":
    main()
