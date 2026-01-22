"""
Minimal baseline for Purpose–Behavior Alignment (v0.1)

Reads JSONL examples, prints label distributions, and runs a simple baseline:
- Purpose baseline: always predict the most frequent purpose label in the dataset.
- Behavior baseline: always predict the most frequent behavior label in the dataset.

This is intentionally simple and fully interpretable.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_DATA_PATH = Path("annotations/examples/examples_v0_1.jsonl")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find data file: {path.as_posix()}")
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
    return data


def get_labels(examples: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    purposes: List[str] = []
    behaviors: List[str] = []
    alignments: List[str] = []
    for ex in examples:
        purposes.append(ex["purpose"]["label"])
        behaviors.append(ex["behavior"]["label"])
        alignments.append(ex["alignment"]["label"])
    return purposes, behaviors, alignments



def majority_label(labels: List[str]) -> str:
    if not labels:
        raise ValueError("Empty label list.")
    return Counter(labels).most_common(1)[0][0]


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    """
    Macro-F1 over labels present in y_true.

    For each label:
      precision = tp / (tp + fp)
      recall    = tp / (tp + fn)
      f1        = 2pr / (p + r)

    Then average across labels.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        return 0.0

    labels = sorted(set(y_true))
    f1s: List[float] = []

    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s)



def main(data_path: Path = DEFAULT_DATA_PATH) -> None:
    examples = load_jsonl(data_path)
    purposes, behaviors, alignments = get_labels(examples)

    print(f"Loaded {len(examples)} examples from {data_path.as_posix()}\n")

    purpose_counts = Counter(purposes)
    behavior_counts = Counter(behaviors)
    alignment_counts = Counter(alignments)

    print("Purpose label distribution:")
    for label, cnt in purpose_counts.most_common():
        print(f"  {label:12s}  {cnt}")

    print("\nBehavior label distribution:")
    for label, cnt in behavior_counts.most_common():
        print(f"  {label:12s}  {cnt}")

    print("\nAlignment label distribution:")
    for label, cnt in alignment_counts.most_common():
        print(f"  {label:12s}  {cnt}")
    maj_purpose = majority_label(purposes)
    maj_behavior = majority_label(behaviors)
    maj_alignment = majority_label(alignments)

    pred_purpose = [maj_purpose] * len(purposes)
    pred_behavior = [maj_behavior] * len(behaviors)

    pred_alignment_majority = [maj_alignment] * len(alignments)
    pred_alignment_rule = [("misaligned" if p == "PUR_PROTEST" else "aligned") for p in purposes]

    print("\nBaselines:")
    print(
        f"  Majority purpose:   {maj_purpose:12s} | acc = {accuracy(purposes, pred_purpose):.3f} | macro-F1 = {macro_f1(purposes, pred_purpose):.3f}"
    )
    print(
        f"  Majority behavior:  {maj_behavior:12s} | acc = {accuracy(behaviors, pred_behavior):.3f} | macro-F1 = {macro_f1(behaviors, pred_behavior):.3f}"
    )
    print(
        f"  Majority alignment: {maj_alignment:12s} | acc = {accuracy(alignments, pred_alignment_majority):.3f} | macro-F1 = {macro_f1(alignments, pred_alignment_majority):.3f}"
    )
    print(
        f"  Rule alignment:     {'PUR_PROTEST->misaligned':12s} | acc = {accuracy(alignments, pred_alignment_rule):.3f} | macro-F1 = {macro_f1(alignments, pred_alignment_rule):.3f}"
    )

if __name__ == "__main__": main()

