from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS_PATH = Path("logs/qa_metrics.csv")
FIGURE_DIR = Path("reports/figures")


SYSTEM_LABELS = {
    "generic_llm_no_rag": "Generic LLM\n(no RAG)",
    "single_agent_rag": "Single-agent\nRAG",
    "great_multi_agent_rag": "GREAT\nmulti-agent RAG",
}


def load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Cannot find metrics file: {METRICS_PATH}")

    df = pd.read_csv(METRICS_PATH)

    print("Loaded columns:")
    print(list(df.columns))

    required = {"id", "domain", "system", "accuracy", "precision", "recall", "f1"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Metrics CSV is missing required columns: {missing}")

    return df


def plot_overall_metrics(df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metric_cols = ["accuracy", "precision", "recall", "f1"]

    summary = df.groupby("system")[metric_cols].mean()
    summary = summary.rename(index=SYSTEM_LABELS)

    ax = summary.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Backend Standards-QA Performance by System")
    ax.set_ylabel("Mean score")
    ax.set_xlabel("System")
    ax.set_ylim(0, 1.05)
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
        values="f1",
        aggfunc="mean",
    )

    pivot = pivot.rename(columns=SYSTEM_LABELS)

    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Backend Standards-QA F1 by Domain")
    ax.set_ylabel("Mean F1")
    ax.set_xlabel("Domain")
    ax.set_ylim(0, 1.05)
    ax.legend(title="System", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()

    out = FIGURE_DIR / "qa_domain_f1.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved: {out}")


def plot_accuracy_heatmap(df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    pivot = df.pivot_table(
        index="id",
        columns="system",
        values="accuracy",
        aggfunc="mean",
    )

    pivot = pivot.rename(columns=SYSTEM_LABELS)

    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.3)))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)

    ax.set_title("Question-Level Success / Failure by System")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, label="Accuracy")
    plt.xticks(rotation=0)
    plt.tight_layout()

    out = FIGURE_DIR / "qa_accuracy_heatmap.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved: {out}")


def main() -> None:
    df = load_metrics()

    plot_overall_metrics(df)
    plot_domain_f1(df)
    plot_accuracy_heatmap(df)


if __name__ == "__main__":
    main()