"""Pure functions for diarization accuracy metrics.

No IO, no app imports. Metrics for speaker diarization quality assessment.
"""

from collections import Counter
from typing import Iterable


def speaker_count(labels: Iterable[str]) -> int:
    """Count distinct speaker labels.

    Args:
        labels: Iterable of speaker label strings (e.g., 'SPEAKER_00', 'SPEAKER_01').

    Returns:
        Number of distinct labels.
    """
    return len(set(labels))


def count_speaker_label(labels: Iterable[str], target: str) -> int:
    """Count exact matches for a target speaker label.

    Args:
        labels: Iterable of speaker label strings.
        target: Target label to count (e.g., 'SPEAKER_02').

    Returns:
        Count of exact matches.
    """
    return sum(1 for label in labels if label == target)


def word_level_speaker_accuracy(
    words: list[dict], reference: list[dict], collar: float = 0.25
) -> float:
    """Compute label-invariant word-level speaker accuracy.

    Aligns hypothesis words to reference words by timestamp midpoint (within collar).
    Returns the best 2-speaker label permutation accuracy.

    Args:
        words: List of dicts with keys 'start', 'end', 'speaker'.
        reference: List of dicts with same structure (ground truth).
        collar: Tolerance in seconds for midpoint alignment.

    Returns:
        Accuracy in [0.0, 1.0]. If no words, returns 0.0.
    """
    if not words or not reference:
        return 0.0

    aligned_pairs = []

    for word in words:
        hyp_mid = (word["start"] + word["end"]) / 2.0
        hyp_label = word["speaker"]

        best_ref = None
        best_dist = float("inf")

        for ref in reference:
            ref_mid = (ref["start"] + ref["end"]) / 2.0
            dist = abs(hyp_mid - ref_mid)

            if dist < collar and dist < best_dist:
                best_dist = dist
                best_ref = ref

        if best_ref is not None:
            aligned_pairs.append((hyp_label, best_ref["speaker"]))

    if not aligned_pairs:
        return 0.0

    unique_hyp_labels = set(label for label, _ in aligned_pairs)
    unique_ref_labels = set(label for _, label in aligned_pairs)

    permutations = []

    if len(unique_hyp_labels) <= 1 or len(unique_ref_labels) <= 1:
        permutations.append({})
    else:
        hyp_labels_list = sorted(unique_hyp_labels)
        ref_labels_list = sorted(unique_ref_labels)

        for hyp_a in hyp_labels_list[:2]:
            for ref_a in ref_labels_list[:2]:
                hyp_b = next((l for l in hyp_labels_list if l != hyp_a), None)
                ref_b = next((l for l in ref_labels_list if l != ref_a), None)

                mapping = {hyp_a: ref_a}
                if hyp_b is not None and ref_b is not None:
                    mapping[hyp_b] = ref_b

                permutations.append(mapping)

    best_accuracy = 0.0

    for mapping in permutations:
        matches = 0
        for hyp_label, ref_label in aligned_pairs:
            mapped_hyp = mapping.get(hyp_label, hyp_label)
            if mapped_hyp == ref_label:
                matches += 1

        accuracy = matches / len(aligned_pairs)
        best_accuracy = max(best_accuracy, accuracy)

    return best_accuracy
