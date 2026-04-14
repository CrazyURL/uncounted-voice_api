"""WeSpeaker ResNet34-LM ONNX embedding extraction (Phase 4: Option B).

Lazy-loads WeSpeaker ONNX model and extracts speaker embeddings.
Handles audio preprocessing: mono mixdown, resample, float32 normalization.
Outputs L2-normalized embeddings for use in speaker reclustering.

Immutable frozen dataclass for error representation.
No external dependencies beyond numpy + stdlib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Constants
_TARGET_SAMPLE_RATE = 16_000
_MIN_AUDIO_SAMPLES = _TARGET_SAMPLE_RATE // 2  # 0.5 seconds


@dataclass(frozen=True)
class EmbeddingUnavailable:
    """Typed result when embedding extraction fails.

    Attributes:
        reason: One of "model_missing", "import_failed", "audio_too_short", "extraction_failed"
    """
    reason: str


class SpeakerEmbeddingModel:
    """WeSpeaker ResNet34-LM ONNX runtime wrapper.

    Lazy-loads the ONNX model on first extract_embedding call.
    Handles audio preprocessing and output normalization.
    """

    def __init__(self) -> None:
        """Initialize model wrapper (lazy load, no model loading here)."""
        self._session = None
        self._loaded = False
        self._unavailable: EmbeddingUnavailable | None = None

    def _load(self) -> EmbeddingUnavailable | None:
        """Lazy load ONNX model.

        Returns:
            None on success, EmbeddingUnavailable on failure.
            Subsequent calls reuse the cached result.
        """
        if self._loaded:
            return self._unavailable

        model_path = os.environ.get("VOICE_DIARIZATION_WESPEAKER_MODEL_PATH")
        if not model_path:
            self._unavailable = EmbeddingUnavailable("model_missing")
            self._loaded = True
            return self._unavailable

        if not os.path.isfile(model_path):
            self._unavailable = EmbeddingUnavailable("model_missing")
            self._loaded = True
            return self._unavailable

        # Try to import and load onnxruntime
        try:
            import onnxruntime
        except ImportError:
            self._unavailable = EmbeddingUnavailable("import_failed")
            self._loaded = True
            return self._unavailable

        try:
            # Parse provider setting
            provider_str = os.environ.get("VOICE_DIARIZATION_EMBEDDING_PROVIDER", "cpu").lower()
            if provider_str == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            # Parse thread setting
            intra_op_threads = int(os.environ.get("VOICE_DIARIZATION_ONNX_INTRA_OP_THREADS", "2"))

            # Create session options
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = intra_op_threads

            # Load ONNX session
            self._session = onnxruntime.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )

            self._loaded = True
            return None

        except Exception:
            self._unavailable = EmbeddingUnavailable("extraction_failed")
            self._loaded = True
            return self._unavailable

    def extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray | EmbeddingUnavailable:
        """Extract speaker embedding from audio.

        Args:
            audio: Audio samples (mono float32 or any shape). Can be (N,) or (C, N).
            sample_rate: Sample rate in Hz.

        Returns:
            L2-normalized embedding (1D array) or EmbeddingUnavailable on failure.
        """
        # Lazy load model
        unavailable = self._load()
        if unavailable is not None:
            return unavailable

        # Validate audio length
        if audio.ndim == 1:
            num_samples = len(audio)
        else:
            num_samples = audio.shape[-1]

        if num_samples < _MIN_AUDIO_SAMPLES:
            return EmbeddingUnavailable("audio_too_short")

        # Preprocess audio: mono mixdown
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Normalize to float32 in range [-1, 1]
        if audio.dtype != np.float32:
            if np.issubdtype(audio.dtype, np.integer):
                # int16 or similar: normalize to [-1, 1]
                info = np.iinfo(audio.dtype)
                audio = audio.astype(np.float32) / max(abs(info.min), abs(info.max))
            else:
                # Other float types: just cast
                audio = audio.astype(np.float32)

        # Resample if needed (simple linear interpolation via np.interp)
        if sample_rate != _TARGET_SAMPLE_RATE:
            # Create resample indices
            ratio = _TARGET_SAMPLE_RATE / sample_rate
            num_samples_resampled = int(len(audio) * ratio)
            old_indices = np.arange(len(audio))
            new_indices = np.linspace(0, len(audio) - 1, num_samples_resampled)
            audio = np.interp(new_indices, old_indices, audio).astype(np.float32)

        # Run inference
        try:
            # Prepare input (model expects [1, N] for mono)
            input_name = self._session.get_inputs()[0].name
            output_name = self._session.get_outputs()[0].name
            input_array = audio.reshape(1, -1).astype(np.float32)

            output = self._session.run(
                [output_name],
                {input_name: input_array}
            )[0]

            # Infer output dimension from model metadata
            output_shape = self._session.get_outputs()[0].shape
            embedding_dim = output_shape[-1] if isinstance(output_shape, (list, tuple)) else output_shape

            # Reshape output to 1D (squeeze batch dimension)
            if output.ndim == 2 and output.shape[0] == 1:
                embedding = output.squeeze(axis=0).astype(np.float32)
            else:
                embedding = output.astype(np.float32)

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-12:
                embedding = embedding / norm

            return embedding

        except Exception:
            return EmbeddingUnavailable("extraction_failed")
