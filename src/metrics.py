from __future__ import annotations

from typing import Any, Dict, List


def normalise_measure(measure: str) -> str:
    return " ".join(measure.lower().strip().replace("/", " ").replace("-", " ").split())


def compute_overlap_metrics(reference_measures: List[str], predicted_measures: List[str]) -> Dict[str, Any]:
    """
    Measure-set alignment. Interpret as similarity to a reference strategy, not absolute truth.
    """
    ref_set = {normalise_measure(m) for m in reference_measures if m.strip()}
    pred_set = {normalise_measure(m) for m in predicted_measures if m.strip()}

    overlap = ref_set & pred_set
    precision = len(overlap) / len(pred_set) if pred_set else 0.0
    recall = len(overlap) / len(ref_set) if ref_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    union = ref_set | pred_set
    jaccard = len(overlap) / len(union) if union else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "ref_count": len(ref_set),
        "pred_count": len(pred_set),
        "overlap_count": len(overlap),
        "overlap_measures": sorted(overlap),
    }
