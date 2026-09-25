from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from openai import OpenAI

from .config import OPENAI_MODEL
from .rag_store import RAGStore


# ============================================================
# CONFIGURATION
# ============================================================

# Fixed temperature for all experimental conditions.
#
# IMPORTANT:
# Keep this unchanged throughout the benchmark.
EVAL_TEMPERATURE = 0.2

# Retrieval settings.
#
# These are part of the frozen experimental configuration.
SPECIALIST_TOP_K = 6
SINGLE_AGENT_TOP_K = 8


# ============================================================
# OPENAI CLIENT
# ============================================================

client = OpenAI()


# ============================================================
# USER PROFILE
# ============================================================

@dataclass
class UserProfile:
    """
    Fixed contextual profile used consistently across
    experimental conditions.
    """

    profession: str = "asset manager"
    organisation: str = ""
    primary_concern: str = "energy and carbon"
    scope: str = "single office building"
    location: str = "UK"
    timeframe: str = "net-zero by 2040"
    time_horizon_years: int = 20
    standards_target: str = (
        "BREEAM Excellent, WLCA-aligned"
    )
    budget: str = "medium"
    risk_appetite: str = "moderate"
    extra_constraints: str = ""

    def to_text(self) -> str:

        lines = [
            "User/context profile:",
            f"- Profession/role: {self.profession}",
            f"- Organisation: "
            f"{self.organisation or 'not specified'}",
            f"- Primary concern: {self.primary_concern}",
            f"- Scope: {self.scope}",
            f"- Location: {self.location}",
            f"- Timeframe: {self.timeframe}",
            f"- Time horizon years: "
            f"{self.time_horizon_years}",
            f"- Standards target: "
            f"{self.standards_target}",
            f"- Budget: {self.budget}",
            f"- Risk appetite: {self.risk_appetite}",
            f"- Extra constraints: "
            f"{self.extra_constraints or 'none'}",
        ]

        return "\n".join(lines)


# ============================================================
# DOMAIN-SPECIFIC RAG AGENT
# ============================================================

class RAGAgent:
    """
    Domain-specific retrieve-then-generate agent.

    Evaluation v2 provides two interfaces:

        answer(...)
            Returns only the final answer string.

        answer_with_trace(...)
            Returns the answer together with the exact
            retrieved evidence used to generate it.

    The latter is used by the controlled evaluation pipeline.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        collection_name: str,
        store: RAGStore,
    ):

        self.name = name
        self.system_prompt = system_prompt
        self.collection_name = collection_name
        self.store = store

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve_trace(
        self,
        query: str,
        k: int = SPECIALIST_TOP_K,
    ):
        """
        Return structured retrieval metadata for evaluation.
        """
        return self.store.query_with_metadata(
            self.collection_name,
            query,
            n_results=k,
        )

    def retrieve_context(
        self,
        query: str,
        k: int = SPECIALIST_TOP_K,
    ) -> list:

        return self.store.query_with_metadata(
            self.collection_name,
            query,
            n_results=k,
        )

    # ========================================================
    # GENERATION WITH TRACE
    # ========================================================

    def answer_with_trace(
        self,
        query: str,
        profile: UserProfile,
        extra_system: str = "",
    ) -> Dict[str, Any]:
        """
        Generate an answer while preserving the retrieval trace.

        Returned structure:

            {
                "agent": ...,
                "collection": ...,
                "retrieved": [...],
                "answer": ...
            }
        """

        retrieved = self.retrieve_context(
            query
        )

        context = (
            self.store.format_retrieval_context(
                retrieved
            )
        )

        messages = [

            {
                "role": "system",
                "content": self.system_prompt,
            },

            {
                "role": "system",
                "content": profile.to_text(),
            },

            {
                "role": "system",
                "content": (
                    "Use the retrieved context as your "
                    "primary evidence. "
                    "Do not invent standard clauses. "
                    "If the retrieved context is insufficient, "
                    "say what is missing and recommend human "
                    "review. "
                    "When possible, mention the source "
                    "document and page shown in the context."
                ),
            },
        ]

        if extra_system:

            messages.append(
                {
                    "role": "system",
                    "content": extra_system,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Retrieved context:\n{context}"
                ),
            }
        )

        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=EVAL_TEMPERATURE,
        )

        answer = (
            completion
            .choices[0]
            .message
            .content
            .strip()
        )

        return {
            "agent": self.name,
            "collection": self.collection_name,
            "retrieved": retrieved,
            "answer": answer,
        }

    # ========================================================
    # BACKWARD-COMPATIBLE ANSWER
    # ========================================================

    def answer(
        self,
        query: str,
        profile: UserProfile,
        extra_system: str = "",
    ) -> str:
        """
        Backward-compatible interface.

        Returns only the generated answer.
        """

        result = self.answer_with_trace(
            query,
            profile,
            extra_system,
        )

        return result["answer"]


# ============================================================
# BUILD SPECIALIST AGENTS
# ============================================================

def build_agents(
    store: RAGStore,
) -> Dict[str, RAGAgent]:
    """
    Build the four fixed domain-specialist agents.

    The domain decomposition is part of the GREAT experimental
    architecture and should remain unchanged during evaluation.
    """

    return {

        # ----------------------------------------------------
        # ENERGY
        # ----------------------------------------------------

        "energy": RAGAgent(
            name="energy",
            collection_name="energy",
            store=store,
            system_prompt=(
                "You are the Energy Efficiency Agent for GREAT. "
                "Focus on operational energy, fabric-first "
                "measures, HVAC, controls, renewables, "
                "occupancy disruption, and phasing."
            ),
        ),

        # ----------------------------------------------------
        # BREEAM
        # ----------------------------------------------------

        "breeam": RAGAgent(
            name="breeam",
            collection_name="breeam",
            store=store,
            system_prompt=(
                "You are the BREEAM Agent. "
                "Focus on BREEAM Refurbishment and Fit-Out "
                "logic, credit pathways, prerequisites, "
                "evidence requirements, risks, and likely "
                "routes to the target rating."
            ),
        ),

        # ----------------------------------------------------
        # WLCA
        # ----------------------------------------------------

        "wlca": RAGAgent(
            name="wlca",
            collection_name="wlca",
            store=store,
            system_prompt=(
                "You are the Whole Life Carbon Assessment Agent. "
                "Focus on RICS WLCA principles, modules, "
                "embodied versus operational carbon trade-offs, "
                "data quality, uncertainty, and reporting."
            ),
        ),

        # ----------------------------------------------------
        # COST
        # ----------------------------------------------------

        "cost": RAGAgent(
            name="cost",
            collection_name="cost",
            store=store,
            system_prompt=(
                "You are the Cost and Feasibility Agent. "
                "Focus on rough order-of-magnitude feasibility, "
                "phasing, quick wins versus deep retrofit, "
                "risks, and items needing QS validation."
            ),
        ),
    }


# ============================================================
# PLANNER / ORCHESTRATOR
# ============================================================

class PlannerAgent:
    """
    Planner/Orchestrator that synthesises specialised
    agent outputs into one final answer.
    """

    def __init__(self):

        self.system_prompt = (
            "You are the Planner/Orchestrator Agent for GREAT, "
            "the Green Retrofit Evidence-grounded Agent Team. "

            "You receive outputs from specialised agents: "
            "[energy], [breeam], [wlca], and [cost]. "

            "Your task is to synthesise the relevant specialist "
            "evidence into one accurate answer to the user's "
            "question. "

            "Integrate information across agents when useful, "
            "but do not add unrelated retrofit recommendations "
            "merely to fill a template. "

            "Preserve important disagreements or uncertainty "
            "between agents. "

            "Do not invent information that is absent from the "
            "specialist outputs. "

            "Where evidence is insufficient, explicitly say so."
        )

    def consolidate(
        self,
        query: str,
        profile: UserProfile,
        agent_answers: Dict[str, str],
    ) -> str:
        """
        Generate the final Planner synthesis.

        The Planner receives specialist answers rather than
        directly querying the vector store.
        """

        answers_block = "\n\n".join(
            f"=== Agent: {name} ===\n{text}"
            for name, text in agent_answers.items()
        )

        messages = [

            {
                "role": "system",
                "content": self.system_prompt,
            },

            {
                "role": "system",
                "content": profile.to_text(),
            },

            {
                "role": "user",
                "content": (
                    f"User question:\n"
                    f"{query}\n\n"

                    f"Agent answers:\n"
                    f"{answers_block}\n\n"

                    "Answer the user's question directly "
                    "using the specialist outputs. "

                    "Integrate relevant information across "
                    "agents when useful. "

                    "Do not add unrelated retrofit "
                    "recommendations merely to fill a template. "

                    "Preserve important disagreements or "
                    "uncertainty between agents. "

                    "Do not invent information that is absent "
                    "from the specialist outputs. "

                    "Where evidence is insufficient, explicitly "
                    "say so."
                ),
            },
        ]

        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=EVAL_TEMPERATURE,
        )

        return (
            completion
            .choices[0]
            .message
            .content
            .strip()
        )


# ============================================================
# GENERIC LLM BASELINE
# ============================================================

def generic_llm_no_rag(
    query: str,
    profile: UserProfile,
) -> str:
    """
    Baseline 1:
    Generic LLM without retrieval.
    """

    messages = [

        {
            "role": "system",
            "content": (
                "You are a generic green retrofit consultant. "

                "You do not have access to project standards "
                "or retrieved documents in this run. "

                "Be cautious and say when details require "
                "checking."
            ),
        },

        {
            "role": "system",
            "content": profile.to_text(),
        },

        {
            "role": "user",
            "content": query,
        },
    ]

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=EVAL_TEMPERATURE,
    )

    return (
        completion
        .choices[0]
        .message
        .content
        .strip()
    )


# ============================================================
# SINGLE-AGENT RAG BASELINE
# ============================================================

def single_agent_rag_with_trace(
    query: str,
    profile: UserProfile,
    store: RAGStore,
) -> Dict[str, Any]:
    """
    Baseline 2:
    Single-agent RAG over the global ALL collection.

    Returns both:
        - generated answer
        - exact retrieved evidence
    """

    retrieved = store.query_with_metadata(
        "ALL",
        query,
        n_results=SINGLE_AGENT_TOP_K,
    )

    context = (
        store.format_retrieval_context(
            retrieved
        )
    )

    messages = [

        {
            "role": "system",
            "content": (
                "You are a single-agent RAG consultant "
                "for green retrofit planning. "

                "Use only the retrieved context as evidence. "

                "Cite source document/page when possible. "

                "If context is insufficient, say so."
            ),
        },

        {
            "role": "system",
            "content": profile.to_text(),
        },

        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                f"Retrieved context:\n{context}"
            ),
        },
    ]

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=EVAL_TEMPERATURE,
    )

    answer = (
        completion
        .choices[0]
        .message
        .content
        .strip()
    )

    return {
        "answer": answer,
        "retrieved": retrieved,
        "collection": "ALL",
    }


def single_agent_rag(
    query: str,
    profile: UserProfile,
    store: RAGStore,
) -> str:
    """
    Backward-compatible single-agent RAG interface.

    Returns only the answer.
    """

    result = single_agent_rag_with_trace(
        query,
        profile,
        store,
    )

    return result["answer"]