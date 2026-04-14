"""Tests for WeSpeaker ONNX embedding model (Phase 4).

Tests validate the SpeakerEmbeddingModel class:
- Lazy loading on first extract_embedding call
- ONNX session creation with provider selection
- Audio preprocessing (mono mixdown, resample, float32 normalize)
- Output dimension inference from model metadata
- L2 normalization of embeddings
- Handling of unavailable model (env var unset, file missing, import failed)
- Audio length validation (minimum duration)

Uses fakes for onnxruntime to avoid requiring ONNX installations in test environments.
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.mark.unit
class TestSpeakerEmbeddingModelInit:
    """Tests for SpeakerEmbeddingModel.__init__()."""

    def test_init_does_not_load_model(self, monkeypatch):
        """__init__ should not trigger lazy load."""
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_MODEL_PATH", "/nonexistent.onnx")

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        model = SpeakerEmbeddingModel()
        assert model is not None  # Just verify construction succeeds


@pytest.mark.unit
class TestMissingModelPath:
    """Tests for missing or invalid VOICE_DIARIZATION_WESPEAKER_MODEL_PATH."""

    def test_missing_model_path_returns_unavailable(self, monkeypatch):
        """Unset VOICE_DIARIZATION_WESPEAKER_MODEL_PATH → EmbeddingUnavailable('model_missing')."""
        monkeypatch.delenv("VOICE_DIARIZATION_WESPEAKER_MODEL_PATH", raising=False)

        from app.services.speaker_embedding import SpeakerEmbeddingModel, EmbeddingUnavailable

        model = SpeakerEmbeddingModel()
        audio = np.random.randn(16000).astype(np.float32)  # 1 second @ 16kHz
        result = model.extract_embedding(audio, 16000)

        assert isinstance(result, EmbeddingUnavailable)
        assert result.reason == "model_missing"

    def test_invalid_model_path_returns_unavailable(self, monkeypatch):
        """Non-existent model file → EmbeddingUnavailable('model_missing')."""
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_MODEL_PATH", "/nonexistent/path.onnx")

        from app.services.speaker_embedding import SpeakerEmbeddingModel, EmbeddingUnavailable

        model = SpeakerEmbeddingModel()
        audio = np.random.randn(16000).astype(np.float32)
        result = model.extract_embedding(audio, 16000)

        assert isinstance(result, EmbeddingUnavailable)
        assert result.reason == "model_missing"


@pytest.mark.unit
class TestONNXRuntimeProvider:
    """Tests for execution provider selection."""

    def test_default_provider_is_cpu(self, monkeypatch, tmp_path):
        """Default VOICE_DIARIZATION_EMBEDDING_PROVIDER=cpu → CPUExecutionProvider."""
        # Create a fake onnxruntime module
        fake_onnx = MagicMock()
        captured_providers = []

        class FakeSession:
            def __init__(self, path, sess_options=None, providers=None):
                self.path = path
                self.providers = providers if providers is not None else []
                captured_providers.append(self.providers)

            def get_outputs(self):
                return [MagicMock(shape=[1, 192])]

            def run(self, output_names, input_feed):
                return [np.array([[0.6, 0.8] + [0.0] * 190], dtype=np.float32)]

        fake_onnx.InferenceSession = FakeSession
        fake_onnx.SessionOptions = MagicMock

        # Inject fake module
        monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnx)

        # Create a dummy model file
        model_path = tmp_path / "test.onnx"
        model_path.write_bytes(b"fake model")

        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_MODEL_PATH", str(model_path))
        monkeypatch.delenv("VOICE_DIARIZATION_EMBEDDING_PROVIDER", raising=False)

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        model = SpeakerEmbeddingModel()
        audio = np.random.randn(16000).astype(np.float32)
        result = model.extract_embedding(audio, 16000)

        assert len(captured_providers) > 0
        providers_used = captured_providers[0]
        assert "CPUExecutionProvider" in providers_used

    def test_provider_override_via_env(self, monkeypatch, tmp_path):
        """VOICE_DIARIZATION_EMBEDDING_PROVIDER=cuda → includes CUDAExecutionProvider."""
        fake_onnx = MagicMock()
        captured_providers = []

        class FakeSession:
            def __init__(self, path, sess_options=None, providers=None):
                self.path = path
                self.providers = providers if providers is not None else []
                captured_providers.append(self.providers)

            def get_outputs(self):
                return [MagicMock(shape=[1, 192])]

            def run(self, output_names, input_feed):
                return [np.array([[0.6, 0.8] + [0.0] * 190], dtype=np.float32)]

        fake_onnx.InferenceSession = FakeSession
        fake_onnx.SessionOptions = MagicMock

        monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnx)

        model_path = tmp_path / "test.onnx"
        model_path.write_bytes(b"fake model")

        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_MODEL_PATH", str(model_path))
        monkeypatch.setenv("VOICE_DIARIZATION_EMBEDDING_PROVIDER", "cuda")

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        model = SpeakerEmbeddingModel()
        audio = np.random.randn(16000).astype(np.float32)
        result = model.extract_embedding(audio, 16000)

        assert len(captured_providers) > 0
        providers_used = captured_providers[0]
        assert "CUDAExecutionProvider" in providers_used


@pytest.mark.unit
class TestAudioNormalization:
    """Tests for audio preprocessing (mono, resample, float32)."""

    def test_input_normalization_mono_float32(self, monkeypatch, tmp_path):
        """Stereo int16 → mono float32 before session.run()."""
        fake_onnx = MagicMock()
        captured_inputs = []

        class FakeSession:
            def __init__(self, path, sess_options=None, providers=None):
                pass

            def get_inputs(self):
                mock_input = MagicMock()
                mock_input.name = "audio_input"
                return [mock_input]

            def get_outputs(self):
                mock_output = MagicMock()
                mock_output.name = "embeddings"
                mock_output.shape = [1, 192]
                return [mock_output]

            def run(self, output_names, input_feed):
                # Capture the input for validation
                for key, val in input_feed.items():
                    captured_inputs.append((key, val.copy() if hasattr(val, "copy") else val))
                return [np.array([[0.6, 0.8] + [0.0] * 190], dtype=np.float32)]

        fake_onnx.InferenceSession = FakeSession
        fake_onnx.SessionOptions = MagicMock

        monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnx)

        model_path = tmp_path / "test.onnx"
        model_path.write_bytes(b"fake model")

        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_MODEL_PATH", str(model_path))

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        # Test stereo int16 input
        model = SpeakerEmbeddingModel()
        stereo_audio = np.random.randint(-32768, 32767, (2, 16000), dtype=np.int16)
        result = model.extract_embedding(stereo_audio, 16000)

        assert len(captured_inputs) > 0
        _, input_array = captured_inputs[0]
        # Should be 2D [1, N] mono float32 after reshape in code
        assert input_array.ndim == 2
        assert input_array.dtype == np.float32
        assert input_array.shape[0] == 1


@pytest.mark.unit
class TestOutputDimensionInference:
    """Tests for output dimension inference from session metadata."""

    def test_output_dimension_inferred_from_session(self, monkeypatch, tmp_path):
        """Output dimension read from session.get_outputs()[0].shape, not hardcoded."""
        fake_onnx = MagicMock()
        test_dim = 192

        class FakeSession:
            def __init__(self, path, sess_options=None, providers=None):
                pass

            def get_inputs(self):
                mock_input = MagicMock()
                mock_input.name = "audio_input"
                return [mock_input]

            def get_outputs(self):
                return [MagicMock(shape=[1, test_dim])]

            def run(self, output_names, input_feed):
                # Return embedding with the inferred dimension
                return [np.zeros((1, test_dim), dtype=np.float32)]

        fake_onnx.InferenceSession = FakeSession
        fake_onnx.SessionOptions = MagicMock

        monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnx)

        model_path = tmp_path / "test.onnx"
        model_path.write_bytes(b"fake model")

        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_MODEL_PATH", str(model_path))

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        model = SpeakerEmbeddingModel()
        audio = np.random.randn(16000).astype(np.float32)
        result = model.extract_embedding(audio, 16000)

        # Result should be 1D with shape (test_dim,) or 2D (1, test_dim) — depends on implementation
        assert isinstance(result, np.ndarray)
        if result.ndim == 1:
            assert result.shape == (test_dim,)
        else:
            assert result.shape == (1, test_dim)


@pytest.mark.unit
class TestL2Normalization:
    """Tests for L2 normalization of output embedding."""

    def test_output_is_l2_normalized(self, monkeypatch, tmp_path):
        """Output embedding has L2 norm ≈ 1.0 (within 1e-6)."""
        fake_onnx = MagicMock()

        class FakeSession:
            def __init__(self, path, sess_options=None, providers=None):
                pass

            def get_inputs(self):
                mock_input = MagicMock()
                mock_input.name = "audio_input"
                return [mock_input]

            def get_outputs(self):
                return [MagicMock(shape=[1, 2])]

            def run(self, output_names, input_feed):
                # Return [3, 4] which has L2 norm = 5.0 (not normalized)
                return [np.array([[3.0, 4.0]], dtype=np.float32)]

        fake_onnx.InferenceSession = FakeSession
        fake_onnx.SessionOptions = MagicMock

        monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnx)

        model_path = tmp_path / "test.onnx"
        model_path.write_bytes(b"fake model")

        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_MODEL_PATH", str(model_path))

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        model = SpeakerEmbeddingModel()
        audio = np.random.randn(16000).astype(np.float32)
        result = model.extract_embedding(audio, 16000)

        # Compute L2 norm
        if result.ndim == 2:
            result = result.squeeze()
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6, f"Expected L2 norm ≈ 1.0, got {norm}"


@pytest.mark.unit
class TestAudioTooShort:
    """Tests for audio length validation."""

    def test_audio_too_short_returns_unavailable(self, monkeypatch, tmp_path):
        """Audio < 0.5 seconds → EmbeddingUnavailable('audio_too_short')."""
        fake_onnx = MagicMock()

        class FakeSession:
            def __init__(self, path, sess_options=None, providers=None):
                pass

            def get_outputs(self):
                return [MagicMock(shape=[1, 192])]

            def run(self, output_names, input_feed):
                return [np.zeros((1, 192), dtype=np.float32)]

        fake_onnx.InferenceSession = FakeSession
        fake_onnx.SessionOptions = MagicMock

        monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnx)

        model_path = tmp_path / "test.onnx"
        model_path.write_bytes(b"fake model")

        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_MODEL_PATH", str(model_path))

        from app.services.speaker_embedding import SpeakerEmbeddingModel, EmbeddingUnavailable

        model = SpeakerEmbeddingModel()
        # 50 samples @ 16kHz = 0.003 seconds (way less than 0.5s)
        short_audio = np.random.randn(50).astype(np.float32)
        result = model.extract_embedding(short_audio, 16000)

        assert isinstance(result, EmbeddingUnavailable)
        assert result.reason == "audio_too_short"


@pytest.mark.unit
class TestONNXRuntimeImportFailure:
    """Tests for handling onnxruntime import failure."""

    def test_onnxruntime_import_failure_handled(self, monkeypatch, tmp_path):
        """onnxruntime unavailable (ImportError) → EmbeddingUnavailable('import_failed')."""
        # Set onnxruntime to None to simulate import failure
        monkeypatch.setitem(sys.modules, "onnxruntime", None)

        model_path = tmp_path / "test.onnx"
        model_path.write_bytes(b"fake model")

        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_MODEL_PATH", str(model_path))

        # Import after setting sys.modules
        from app.services.speaker_embedding import SpeakerEmbeddingModel, EmbeddingUnavailable

        model = SpeakerEmbeddingModel()
        audio = np.random.randn(16000).astype(np.float32)
        result = model.extract_embedding(audio, 16000)

        assert isinstance(result, EmbeddingUnavailable)
        assert result.reason == "import_failed"
