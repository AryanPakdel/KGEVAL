"""Metrics for comparing pipeline predictions against benchmark labels.

Convention follows GraphEval / GraphEval+: positive class = 1 (hallucination),
so `balanced_accuracy` measures our ability to detect hallucinations.
"""

from __future__ import annotations

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    if not y_true:
        return {
            "n": 0,
            "balanced_accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "label_distribution": {"consistent": 0, "hallucination": 0},
            "prediction_distribution": {"consistent": 0, "hallucination": 0},
        }

    ba = balanced_accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "n": len(y_true),
        "balanced_accuracy": float(ba),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "label_distribution": {
            "consistent": int(sum(1 for y in y_true if y == 0)),
            "hallucination": int(sum(1 for y in y_true if y == 1)),
        },
        "prediction_distribution": {
            "consistent": int(sum(1 for y in y_pred if y == 0)),
            "hallucination": int(sum(1 for y in y_pred if y == 1)),
        },
    }


def format_metrics_table(results: dict[str, dict]) -> str:
    """Pretty-print a table indexed by {benchmark: metrics_dict}."""
    header = f"{'benchmark':<12} {'n':>5} {'bal_acc':>8} {'prec':>6} {'rec':>6} {'f1':>6}"
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        ba = f"{m['balanced_accuracy']*100:6.2f}" if m["balanced_accuracy"] is not None else "  n/a"
        pr = f"{m['precision']*100:5.2f}" if m["precision"] is not None else " n/a"
        rc = f"{m['recall']*100:5.2f}" if m["recall"] is not None else " n/a"
        f1 = f"{m['f1']*100:5.2f}" if m["f1"] is not None else " n/a"
        lines.append(f"{name:<12} {m['n']:>5} {ba:>8} {pr:>6} {rc:>6} {f1:>6}")
    return "\n".join(lines)
