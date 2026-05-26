from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from openai import OpenAI

from .config import OPENAI_MODEL
from .rag_store import RAGStore

client = OpenAI()


@dataclass
class UserProfile:
    """Privacy-safe user/context profile. We do not collect the user's name."""
    profession: str = "Area Chief Engineer"
    organisation: str = ""
    primary_concern: str = "energy and carbon savings"
    scope: str = "single office building"
    location: str = "UK"
    timeframe: str = "net-zero by 2050"
    time_horizon_years: int = 20
    standards_target: str = "BREEAM Excellent, WLCA-aligned"
    budget: str = "medium"      # low / medium / high
    risk_appetite: str = "moderate"  # low / moderate / high
    extra_constraints: str = ""

    def to_text(self) -> str:
        lines = [
            "User/context profile:",
            f"- Profession/role: {self.profession}",
            f"- Organisation: {self.organisation or 'not specified'}",
            f"- Primary concern: {self.primary_concern}",
            f"- Scope: {self.scope}",
            f"- Location: {self.location}",
            f"- Timeframe: {self.timeframe}",
            f"- Time horizon years: {self.time_horizon_years}",
            f"- Standards target: {self.standards_target}",
            f"- Budget: {self.budget}",
            f"- Risk appetite: {self.risk_appetite}",
            f"- Extra constraints: {self.extra_constraints or 'none'}",
        ]
        return "\n".join(lines)


class RAGAgent:
    """Domain-specific retrieve-then-generate agent."""

    def __init__(self, name: str, system_prompt: str, collection_name: str, store: RAGStore):
        self.name = name
        self.system_prompt = system_prompt
        self.collection_name = collection_name
        self.store = store

    def retrieve_context(self, query: str, k: int = 6) -> str:
        return self.store.query(self.collection_name, query, n_results=k)

    def answer(self, query: str, profile: UserProfile, extra_system: str = "") -> str:
        context = self.retrieve_context(query)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": profile.to_text()},
            {
                "role": "system",
                "content": (
                    "Use the retrieved context as your primary evidence. Do not invent standard clauses. "
                    "If the retrieved context is insufficient, say what is missing and recommend human review. "
                    "When possible, mention the source document and page shown in the context."
                ),
            },
        ]
        if extra_system:
            messages.append({"role": "system", "content": extra_system})
        messages.append({"role": "user", "content": f"Question:\n{query}\n\nRetrieved context:\n{context}"})

        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return completion.choices[0].message.content.strip()


def build_agents(store: RAGStore) -> Dict[str, RAGAgent]:
    return {
        "energy": RAGAgent(
            name="energy",
            collection_name="energy",
            store=store,
            system_prompt=(
                "You are the Energy Efficiency Agent for GREAT. Focus on operational energy, "
                "fabric-first measures, HVAC, controls, renewables, occupancy disruption, and phasing."
            ),
        ),
        "breeam": RAGAgent(
            name="breeam",
            collection_name="breeam",
            store=store,
            system_prompt=(
                "You are the BREEAM Agent. Focus on BREEAM Refurbishment and Fit-Out logic, credit pathways, "
                "prerequisites, evidence requirements, risks, and likely routes to the target rating."
            ),
        ),
        "wlca": RAGAgent(
            name="wlca",
            collection_name="wlca",
            store=store,
            system_prompt=(
                "You are the Whole Life Carbon Assessment Agent. Focus on RICS WLCA principles, modules, "
                "embodied vs operational carbon trade-offs, data quality, uncertainty, and reporting."
            ),
        ),
        "cost": RAGAgent(
            name="cost",
            collection_name="cost",
            store=store,
            system_prompt=(
                "You are the Cost and Feasibility Agent. Focus on rough order-of-magnitude feasibility, "
                "phasing, quick wins versus deep retrofit, risks, and items needing QS validation."
            ),
        ),
    }


class PlannerAgent:
    """Planner/Orchestrator that synthesises specialised agent outputs into one strategy."""

    def __init__(self):
        self.system_prompt = (
            "You are the Planner/Orchestrator Agent for GREAT, the Green Retrofit Evidence-grounded Agent Team. You receive outputs from "
            "specialised agents: [energy], [breeam], [wlca], [cost]. Your tasks are to identify conflicts, "
            "integrate trade-offs, produce one coherent phased retrofit strategy, mark key ideas with agent tags, "
            "and state open issues for expert review."
        )

    def consolidate(self, query: str, profile: UserProfile, agent_answers: Dict[str, str]) -> str:
        answers_block = "\n\n".join(f"=== Agent: {name} ===\n{text}" for name, text in agent_answers.items())
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": profile.to_text()},
            {
                "role": "user",
                "content": (
                    f"User question:\n{query}\n\nAgent answers:\n{answers_block}\n\n"
                    "Now provide:\n"
                    "1) Executive summary.\n"
                    "2) Phased retrofit strategy.\n"
                    "3) BREEAM and WLCA alignment notes.\n"
                    "4) Key trade-offs and conflicts.\n"
                    "5) Open issues for human expert review."
                ),
            },
        ]
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()


def generic_llm_no_rag(query: str, profile: UserProfile) -> str:
    """Baseline 1: generic LLM without retrieval."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a generic green retrofit consultant. You do not have access to project standards or "
                "retrieved documents in this run. Be cautious and say when details require checking."
            ),
        },
        {"role": "system", "content": profile.to_text()},
        {"role": "user", "content": query},
    ]
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.7,
    )
    return completion.choices[0].message.content.strip()


def single_agent_rag(query: str, profile: UserProfile, store: RAGStore) -> str:
    """Baseline 2: single RAG agent over the global ALL collection."""
    context = store.query("ALL", query, n_results=10)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a single-agent RAG consultant for green retrofit planning. Use only the retrieved context "
                "as evidence. Cite source document/page when possible. If context is insufficient, say so."
            ),
        },
        {"role": "system", "content": profile.to_text()},
        {"role": "user", "content": f"Question:\n{query}\n\nRetrieved context:\n{context}"},
    ]
    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return completion.choices[0].message.content.strip()
