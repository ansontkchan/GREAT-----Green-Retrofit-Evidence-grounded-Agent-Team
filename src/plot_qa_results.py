from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS_PATH = Path("logs/qa_backend_metrics.csv")
FIGURE_DIR = Path("reports/figures")


SYSTEM_LABELS = {
    "generic_llm_no_rag": "Generic LLM\n(no RAG)",
    "single_agent_rag": "Single-agent\nRAG",
    "great_multi_agent_rag": "GREAT\nmulti-agent RAG",
}


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")

    return pd.read_csv(path)


def plot_overall_metrics(df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        "answer_accuracy",
        "answer_precision",
        "answer_recall",
        "answer_f1",
    ]

    summary = df.groupby("system")[metric_cols].mean()
    summary = summary.rename(index=SYSTEM_LABELS)

    ax = summary.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Standards QA Performance by System")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("System")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()

    out = FIGURE_DIR / "qa_overall_metrics.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved: {out}")


def plot_domain_f1(df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    pivot = df.pivot_table(
        index="domain",
        columns="system",
        values="answer_f1",
        aggfunc="mean",
    )

    pivot = pivot.rename(columns=SYSTEM_LABELS)

    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Standards QA F1 by Domain")
    ax.set_ylabel("Mean F1")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Domain")
    ax.legend(title="System", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()

    out = FIGURE_DIR / "qa_domain_f1.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved: {out}")


def plot_context_recall(df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rag_df = df[df["system"] != "generic_llm_no_rag"].copy()

    if "context_recall" not in rag_df.columns:
        print("No context_recall column found.")
        return

    summary = rag_df.groupby("system")["context_recall"].mean()
    summary = summary.rename(index=SYSTEM_LABELS)

    ax = summary.plot(kind="bar", figsize=(7, 5))
    ax.set_title("Retrieved Context Recall by RAG System")
    ax.set_ylabel("Mean context recall")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("System")
    plt.xticks(rotation=0)
    plt.tight_layout()

    out = FIGURE_DIR / "qa_context_recall.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved: {out}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_METRICS_PATH

    df = load_metrics(path)

    plot_overall_metrics(df)
    plot_domain_f1(df)
    plot_context_recall(df)


if __name__ == "__main__":
    main()