"""Tests for speaker reclustering and relabeling (Phase 6).

Tests validate recluster_speakers():
- Pure-numpy AHC (2-cluster agglomerative hierarchical clustering)
- Confidence scoring (intra vs inter-cluster margin)
- Label canonicalization (cluster ids → SPEAKER_00/SPEAKER_01)
- Low-confidence fallback (returns original labels unchanged)
- Speaker phantom cleanup (SPEAKER_02+ → reassigned to nearest cluster)
- Deterministic output (no randomness in label assignment)
- Immutability (input dicts not mutated, output is new dicts in tuple)
- Edge cases (N < 2, N == 2, empty inputs, unmapped words)
"""

import numpy as np
import pytest


@pytest.mark.unit
class TestTwoWellSeparatedClusters:
    """Tests for clearly separable embeddings → split into 2 clusters."""

    def test_two_well_separated_clusters_split_correctly(self):
        """4 embeddings: [1,0,0], [0.98,0.02,0] vs [0,1,0], [0.02,0.98,0].
        All words currently SPEAKER_00 → after reclustering, at least 2 distinct labels.
        """
        from app.services.speaker_recluster import recluster_speakers

        # Create clearly separable embeddings: cluster A ≈ [1,0,0], cluster B ≈ [0,1,0]
        # 2 windows with distinct embeddings
        # L2-normalize each embedding
        embeddings = np.array([
            [1.0, 0.0, 0.0],    # Window 0: Cluster A
            [0.0, 1.0, 0.0],    # Window 1: Cluster B
        ], dtype=np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Input words all labeled SPEAKER_00
        words = [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_00", "text": "b"},
            {"start": 2.0, "end": 3.0, "speaker_id": "SPEAKER_00", "text": "c"},
            {"start": 3.0, "end": 4.0, "speaker_id": "SPEAKER_00", "text": "d"},
        ]

        # Window mapping: words 0,1 → window 0; words 2,3 → window 1
        window_indices_per_word = [0, 0, 1, 1]

        updated_words, confidence, changed = recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
            confidence_threshold=0.0,  # Low threshold to force clustering
        )

        # Should have found 2 clusters and changed labels
        assert changed is True
        assert isinstance(updated_words, tuple)
        assert len(updated_words) == 4

        # Extract unique labels from output
        labels = {w["speaker_id"] for w in updated_words}
        # Should have at least 2 distinct labels (could be >2 if some words unmapped)
        assert len(labels) >= 2

        # All labels should be canonical SPEAKER_NN format
        for label in labels:
            assert label in ("SPEAKER_00", "SPEAKER_01"), f"Non-canonical label: {label}"

        # Confidence should be in [0, 1]
        assert 0.0 <= confidence <= 1.0

    def test_phantom_speaker_02_reassigned_to_nearest_cluster(self):
        """4 embeddings, input includes SPEAKER_02 phantom.
        After reclustering, all labels in {SPEAKER_00, SPEAKER_01}.
        SPEAKER_02 word reassigned to nearest cluster.
        """
        from app.services.speaker_recluster import recluster_speakers

        # 2 windows with distinct embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],        # Window 0: Cluster A
            [0.0, 1.0, 0.0],        # Window 1: Cluster B
        ], dtype=np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        words = [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_00", "text": "b"},
            {"start": 2.0, "end": 3.0, "speaker_id": "SPEAKER_02", "text": "c"},  # Phantom
            {"start": 3.0, "end": 4.0, "speaker_id": "SPEAKER_01", "text": "d"},
        ]

        window_indices_per_word = [0, 0, 1, 1]

        updated_words, confidence, changed = recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
            confidence_threshold=0.0,
        )

        # Extract unique output labels
        labels = {w["speaker_id"] for w in updated_words}
        assert labels == {"SPEAKER_00", "SPEAKER_01"}, f"Got labels: {labels}"

        # SPEAKER_02 should be reassigned
        assert not any(w["speaker_id"] == "SPEAKER_02" for w in updated_words)

    def test_low_confidence_returns_original_labels_unchanged(self):
        """4 nearly-identical embeddings (all ~[1,0,0]).
        Confidence below threshold → changed=False, labels byte-identical to input.
        """
        from app.services.speaker_recluster import recluster_speakers

        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.98, 0.02, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)

        words = [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_00", "text": "b"},
            {"start": 2.0, "end": 3.0, "speaker_id": "SPEAKER_00", "text": "c"},
            {"start": 3.0, "end": 4.0, "speaker_id": "SPEAKER_00", "text": "d"},
        ]

        window_indices_per_word = [0, 0, 1, 1]

        # High threshold → low confidence → fallback
        updated_words, confidence, changed = recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
            confidence_threshold=0.99,  # Very high threshold
        )

        # Should not have changed
        assert changed is False
        assert confidence < 0.99

        # Labels should be identical to input
        for i, updated_word in enumerate(updated_words):
            assert updated_word["speaker_id"] == words[i]["speaker_id"]


@pytest.mark.unit
class TestCanonicalLabeling:
    """Tests for label canonicalization (cluster ids → SPEAKER_NN)."""

    def test_canonical_label_names_only(self):
        """Returned labels never contain raw sklearn cluster ids (0, 1)."""
        from app.services.speaker_recluster import recluster_speakers

        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
        ], dtype=np.float32)

        words = [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_00", "text": "b"},
            {"start": 2.0, "end": 3.0, "speaker_id": "SPEAKER_00", "text": "c"},
            {"start": 3.0, "end": 4.0, "speaker_id": "SPEAKER_00", "text": "d"},
        ]

        window_indices_per_word = [0, 0, 1, 1]

        updated_words, _, _ = recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
            confidence_threshold=0.0,
        )

        for word in updated_words:
            label = word["speaker_id"]
            # No raw sklearn ids
            assert label not in ("0", "1"), f"Raw cluster id detected: {label}"
            # Must be canonical format
            assert label in ("SPEAKER_00", "SPEAKER_01"), f"Non-canonical: {label}"

    def test_label_mapping_is_deterministic(self):
        """Run reclustering twice on same input → identical results."""
        from app.services.speaker_recluster import recluster_speakers

        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
        ], dtype=np.float32)

        words = [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_00", "text": "b"},
            {"start": 2.0, "end": 3.0, "speaker_id": "SPEAKER_00", "text": "c"},
            {"start": 3.0, "end": 4.0, "speaker_id": "SPEAKER_00", "text": "d"},
        ]

        window_indices_per_word = [0, 0, 1, 1]

        result1 = recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
            confidence_threshold=0.0,
        )

        result2 = recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
            confidence_threshold=0.0,
        )

        # Results should be identical (same labels in same order)
        assert result1[0] == result2[0], "Labels differ between runs"
        assert result1[1] == result2[1], "Confidence differs between runs"
        assert result1[2] == result2[2], "Changed flag differs between runs"


@pytest.mark.unit
class TestWindowMapping:
    """Tests for word-to-window mapping and relabeling."""

    def test_word_indices_to_window_mapping(self):
        """6 words mapped to 2 windows (3 each).
        Windows have clearly separable embeddings.
        All words from window 0 get same label; all from window 1 get other label.
        """
        from app.services.speaker_recluster import recluster_speakers

        embeddings = np.array([
            [1.0, 0.0, 0.0],        # Window 0 embedding
            [0.99, 0.01, 0.0],      # Window 0 embedding
            [0.0, 1.0, 0.0],        # Window 1 embedding
            [0.01, 0.99, 0.0],      # Window 1 embedding
        ], dtype=np.float32)

        words = [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_00", "text": "b"},
            {"start": 2.0, "end": 3.0, "speaker_id": "SPEAKER_00", "text": "c"},
            {"start": 3.0, "end": 4.0, "speaker_id": "SPEAKER_00", "text": "d"},
            {"start": 4.0, "end": 5.0, "speaker_id": "SPEAKER_00", "text": "e"},
            {"start": 5.0, "end": 6.0, "speaker_id": "SPEAKER_00", "text": "f"},
        ]

        # Window 0 covers words 0,1,2; window 1 covers words 3,4,5
        window_indices_per_word = [0, 0, 0, 1, 1, 1]

        updated_words, _, _ = recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
            confidence_threshold=0.0,
        )

        # Words from window 0 should have same label
        label_0 = updated_words[0]["speaker_id"]
        assert updated_words[1]["speaker_id"] == label_0
        assert updated_words[2]["speaker_id"] == label_0

        # Words from window 1 should have same label (possibly different from 0)
        label_1 = updated_words[3]["speaker_id"]
        assert updated_words[4]["speaker_id"] == label_1
        assert updated_words[5]["speaker_id"] == label_1

    def test_unmapped_words_keep_original_label(self):
        """Word with window index None or out-of-range → keep original label."""
        from app.services.speaker_recluster import recluster_speakers

        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
        ], dtype=np.float32)

        words = [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_02", "text": "b"},  # Will be unmapped
            {"start": 2.0, "end": 3.0, "speaker_id": "SPEAKER_00", "text": "c"},
            {"start": 3.0, "end": 4.0, "speaker_id": "SPEAKER_00", "text": "d"},
        ]

        window_indices_per_word = [0, None, 1, 1]  # Word 1 unmapped

        updated_words, _, _ = recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
            confidence_threshold=0.0,
        )

        # Unmapped word should keep original label
        assert updated_words[1]["speaker_id"] == "SPEAKER_02"


@pytest.mark.unit
class TestImmutability:
    """Tests for immutability of input and output."""

    def test_input_words_not_mutated(self):
        """Input word list/dicts should not be modified."""
        from app.services.speaker_recluster import recluster_speakers

        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
        ], dtype=np.float32)

        words = [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_00", "text": "b"},
            {"start": 2.0, "end": 3.0, "speaker_id": "SPEAKER_00", "text": "c"},
            {"start": 3.0, "end": 4.0, "speaker_id": "SPEAKER_00", "text": "d"},
        ]

        window_indices_per_word = [0, 0, 1, 1]

        # Store original ids
        original_ids = [id(w) for w in words]
        original_labels = [w["speaker_id"] for w in words]

        recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
            confidence_threshold=0.0,
        )

        # Input list/dicts should be unchanged
        assert [id(w) for w in words] == original_ids
        assert [w["speaker_id"] for w in words] == original_labels

    def test_returned_words_are_immutable_tuple_of_dicts(self):
        """Return type is tuple[dict, ...], dicts are NEW copies (not input refs)."""
        from app.services.speaker_recluster import recluster_speakers

        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
        ], dtype=np.float32)

        words = [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_00", "text": "b"},
            {"start": 2.0, "end": 3.0, "speaker_id": "SPEAKER_00", "text": "c"},
            {"start": 3.0, "end": 4.0, "speaker_id": "SPEAKER_00", "text": "d"},
        ]

        window_indices_per_word = [0, 0, 1, 1]

        updated_words, _, _ = recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
            confidence_threshold=0.0,
        )

        # Should be tuple
        assert isinstance(updated_words, tuple)

        # Dicts should be NEW, not references
        for i, updated_word in enumerate(updated_words):
            assert id(updated_word) != id(words[i]), f"Word {i} is same object!"

    def test_confidence_score_in_unit_range(self):
        """Confidence always in [0.0, 1.0]."""
        from app.services.speaker_recluster import recluster_speakers

        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
        ], dtype=np.float32)

        words = [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_00", "text": "b"},
            {"start": 2.0, "end": 3.0, "speaker_id": "SPEAKER_00", "text": "c"},
            {"start": 3.0, "end": 4.0, "speaker_id": "SPEAKER_00", "text": "d"},
        ]

        window_indices_per_word = [0, 0, 1, 1]

        _, confidence, _ = recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
            confidence_threshold=0.0,
        )

        assert 0.0 <= confidence <= 1.0


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases: empty inputs, N<2, N=2."""

    def test_empty_inputs_return_empty_unchanged(self):
        """recluster_speakers([], np.zeros((0, 2)), []) → ((), 0.0, False)."""
        from app.services.speaker_recluster import recluster_speakers

        embeddings = np.zeros((0, 2), dtype=np.float32)
        words = []
        window_indices_per_word = []

        updated_words, confidence, changed = recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
        )

        assert updated_words == ()
        assert confidence == 0.0
        assert changed is False

    def test_n_equals_2_edge_case(self):
        """Exactly 2 embeddings → each gets its own cluster."""
        from app.services.speaker_recluster import recluster_speakers

        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        words = [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "a"},
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_00", "text": "b"},
        ]

        window_indices_per_word = [0, 1]

        updated_words, _, changed = recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
            confidence_threshold=0.0,
        )

        # Should have 2 distinct labels (each embedding in its own cluster)
        labels = {w["speaker_id"] for w in updated_words}
        assert len(labels) == 2
        assert labels == {"SPEAKER_00", "SPEAKER_01"}
        assert changed is True

    def test_n_equals_1_edge_case(self):
        """Single embedding → single cluster, no change."""
        from app.services.speaker_recluster import recluster_speakers

        embeddings = np.array([
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)

        words = [
            {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00", "text": "a"},
        ]

        window_indices_per_word = [0]

        updated_words, _, changed = recluster_speakers(
            words,
            embeddings,
            window_indices_per_word,
            confidence_threshold=0.0,
        )

        # Single embedding → single cluster → changed=False
        assert changed is False
        assert updated_words[0]["speaker_id"] == "SPEAKER_00"
