"""Tests for pyannote-based speaker embedding model (Phase 4, Option B).

Tests validate the SpeakerEmbeddingModel class:
- Lazy loading on first extract_embedding call
- pyannote.audio.Model.from_pretrained 호출 + device 배치
- Audio preprocessing (mono mixdown, resample, float32 normalize)
- L2 normalization of embeddings
- Handling of unavailable model (from_pretrained 실패, import 실패, 오디오 짧음)

Uses fakes for pyannote.audio via sys.modules injection to avoid downloading
the real wespeaker model in test environments.
"""

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def _install_fake_pyannote(monkeypatch, fake_model_factory):
    """Inject a fake `pyannote.audio` module that exposes `Model.from_pretrained`.

    `fake_model_factory(repo, **kwargs)` should return an object that responds
    to `.eval()`, `.to(device)`, and `__call__(tensor)`.
    """
    fake_pkg = types.ModuleType("pyannote")
    fake_audio = types.ModuleType("pyannote.audio")

    class FakeModelClass:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            return fake_model_factory(repo, **kwargs)

    fake_audio.Model = FakeModelClass
    fake_pkg.audio = fake_audio
    monkeypatch.setitem(sys.modules, "pyannote", fake_pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_audio)


def _reload_module():
    """Drop the cached speaker_embedding module so the next import re-binds
    the freshly-faked `pyannote.audio`."""
    sys.modules.pop("app.services.speaker_embedding", None)


@pytest.fixture(autouse=True)
def _reset_embedding_module():
    yield
    sys.modules.pop("app.services.speaker_embedding", None)


@pytest.mark.unit
class TestSpeakerEmbeddingModelInit:
    def test_init_does_not_load_model(self, monkeypatch):
        load_calls = []

        def factory(repo, **kwargs):
            load_calls.append(repo)
            return MagicMock()

        _install_fake_pyannote(monkeypatch, factory)
        _reload_module()

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        SpeakerEmbeddingModel()
        assert load_calls == []  # lazy, not loaded yet


@pytest.mark.unit
class TestModelLoadFailure:
    def test_from_pretrained_failure_returns_unavailable(self, monkeypatch):
        def factory(repo, **kwargs):
            raise RuntimeError("no such repo")

        _install_fake_pyannote(monkeypatch, factory)
        _reload_module()

        from app.services.speaker_embedding import (
            EmbeddingUnavailable,
            SpeakerEmbeddingModel,
        )

        model = SpeakerEmbeddingModel()
        audio = np.zeros(16000, dtype=np.float32)
        result = model.extract_embedding(audio, 16000)
        assert isinstance(result, EmbeddingUnavailable)
        assert result.reason == "model_missing"

    def test_empty_repo_returns_unavailable(self, monkeypatch):
        def factory(repo, **kwargs):
            return MagicMock()

        _install_fake_pyannote(monkeypatch, factory)
        monkeypatch.setenv("VOICE_DIARIZATION_WESPEAKER_REPO", "")
        _reload_module()

        from app.services.speaker_embedding import (
            EmbeddingUnavailable,
            SpeakerEmbeddingModel,
        )

        model = SpeakerEmbeddingModel()
        audio = np.zeros(16000, dtype=np.float32)
        result = model.extract_embedding(audio, 16000)
        assert isinstance(result, EmbeddingUnavailable)
        assert result.reason == "model_missing"


@pytest.mark.unit
class TestDeviceSelection:
    def _make_fake_model(self, captured):
        fake = MagicMock()
        fake.eval = MagicMock(return_value=fake)

        def fake_to(device):
            captured["device"] = device
            return fake

        fake.to = fake_to

        def fake_call(tensor):
            return torch.zeros((1, 4), dtype=torch.float32)

        fake.side_effect = fake_call
        return fake

    def test_default_provider_is_cpu(self, monkeypatch):
        captured = {}

        def factory(repo, **kwargs):
            return self._make_fake_model(captured)

        _install_fake_pyannote(monkeypatch, factory)
        monkeypatch.delenv("VOICE_DIARIZATION_EMBEDDING_PROVIDER", raising=False)
        _reload_module()

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        model = SpeakerEmbeddingModel()
        audio = np.zeros(16000, dtype=np.float32)
        model.extract_embedding(audio, 16000)
        assert captured["device"] == "cpu"

    def test_cuda_provider_uses_cuda_when_available(self, monkeypatch):
        captured = {}

        def factory(repo, **kwargs):
            return self._make_fake_model(captured)

        _install_fake_pyannote(monkeypatch, factory)
        monkeypatch.setenv("VOICE_DIARIZATION_EMBEDDING_PROVIDER", "cuda")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        _reload_module()

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        model = SpeakerEmbeddingModel()
        audio = np.zeros(16000, dtype=np.float32)
        model.extract_embedding(audio, 16000)
        assert captured["device"] == "cuda"

    def test_cuda_provider_falls_back_when_unavailable(self, monkeypatch):
        captured = {}

        def factory(repo, **kwargs):
            return self._make_fake_model(captured)

        _install_fake_pyannote(monkeypatch, factory)
        monkeypatch.setenv("VOICE_DIARIZATION_EMBEDDING_PROVIDER", "cuda")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        _reload_module()

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        model = SpeakerEmbeddingModel()
        audio = np.zeros(16000, dtype=np.float32)
        model.extract_embedding(audio, 16000)
        assert captured["device"] == "cpu"


@pytest.mark.unit
class TestAudioPreprocessing:
    def test_stereo_int16_converted_to_mono_float32(self, monkeypatch):
        captured = {}

        def fake_call(tensor):
            captured["shape"] = tuple(tensor.shape)
            captured["dtype"] = tensor.dtype
            return torch.tensor([[0.6, 0.8]], dtype=torch.float32)

        def factory(repo, **kwargs):
            fake = MagicMock()
            fake.eval = MagicMock(return_value=fake)
            fake.to = MagicMock(return_value=fake)
            fake.side_effect = fake_call
            return fake

        _install_fake_pyannote(monkeypatch, factory)
        _reload_module()

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        model = SpeakerEmbeddingModel()
        stereo = np.random.randint(-32768, 32767, (2, 16000), dtype=np.int16)
        result = model.extract_embedding(stereo, 16000)

        assert isinstance(result, np.ndarray)
        # (batch=1, channels=1, samples)
        assert captured["shape"][0] == 1
        assert captured["shape"][1] == 1
        assert captured["dtype"] == torch.float32

    def test_resample_from_8k_to_16k(self, monkeypatch):
        captured = {}

        def fake_call(tensor):
            captured["samples"] = tensor.shape[-1]
            return torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        def factory(repo, **kwargs):
            fake = MagicMock()
            fake.eval = MagicMock(return_value=fake)
            fake.to = MagicMock(return_value=fake)
            fake.side_effect = fake_call
            return fake

        _install_fake_pyannote(monkeypatch, factory)
        _reload_module()

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        model = SpeakerEmbeddingModel()
        audio = np.zeros(8000, dtype=np.float32)  # 1s @ 8kHz
        model.extract_embedding(audio, 8000)
        # 1s @ 16kHz target → ~16000 samples
        assert 15000 <= captured["samples"] <= 17000


@pytest.mark.unit
class TestL2Normalization:
    def test_output_is_l2_normalized(self, monkeypatch):
        def fake_call(tensor):
            # Unnormalized [3, 4] → L2 norm = 5.0
            return torch.tensor([[3.0, 4.0]], dtype=torch.float32)

        def factory(repo, **kwargs):
            fake = MagicMock()
            fake.eval = MagicMock(return_value=fake)
            fake.to = MagicMock(return_value=fake)
            fake.side_effect = fake_call
            return fake

        _install_fake_pyannote(monkeypatch, factory)
        _reload_module()

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        model = SpeakerEmbeddingModel()
        audio = np.zeros(16000, dtype=np.float32)
        result = model.extract_embedding(audio, 16000)

        assert isinstance(result, np.ndarray)
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5


@pytest.mark.unit
class TestAudioTooShort:
    def test_audio_too_short_returns_unavailable(self, monkeypatch):
        def factory(repo, **kwargs):
            return MagicMock()

        _install_fake_pyannote(monkeypatch, factory)
        _reload_module()

        from app.services.speaker_embedding import (
            EmbeddingUnavailable,
            SpeakerEmbeddingModel,
        )

        model = SpeakerEmbeddingModel()
        # 50 samples @ 16kHz ≈ 3ms
        short_audio = np.random.randn(50).astype(np.float32)
        result = model.extract_embedding(short_audio, 16000)
        assert isinstance(result, EmbeddingUnavailable)
        assert result.reason == "audio_too_short"


@pytest.mark.unit
class TestImportFailure:
    def test_pyannote_import_failure_returns_unavailable(self, monkeypatch):
        # Force ImportError by setting pyannote.audio to None
        monkeypatch.setitem(sys.modules, "pyannote", None)
        monkeypatch.setitem(sys.modules, "pyannote.audio", None)
        _reload_module()

        from app.services.speaker_embedding import (
            EmbeddingUnavailable,
            SpeakerEmbeddingModel,
        )

        model = SpeakerEmbeddingModel()
        audio = np.zeros(16000, dtype=np.float32)
        result = model.extract_embedding(audio, 16000)
        assert isinstance(result, EmbeddingUnavailable)
        assert result.reason == "import_failed"


@pytest.mark.unit
class TestForwardFailure:
    def test_forward_exception_returns_extraction_failed(self, monkeypatch):
        def fake_call(tensor):
            raise RuntimeError("boom")

        def factory(repo, **kwargs):
            fake = MagicMock()
            fake.eval = MagicMock(return_value=fake)
            fake.to = MagicMock(return_value=fake)
            fake.side_effect = fake_call
            return fake

        _install_fake_pyannote(monkeypatch, factory)
        _reload_module()

        from app.services.speaker_embedding import (
            EmbeddingUnavailable,
            SpeakerEmbeddingModel,
        )

        model = SpeakerEmbeddingModel()
        audio = np.zeros(16000, dtype=np.float32)
        result = model.extract_embedding(audio, 16000)
        assert isinstance(result, EmbeddingUnavailable)
        assert result.reason == "extraction_failed"


@pytest.mark.unit
class TestLazyLoadSingleton:
    def test_model_loaded_once(self, monkeypatch):
        call_count = {"n": 0}

        def factory(repo, **kwargs):
            call_count["n"] += 1
            fake = MagicMock()
            fake.eval = MagicMock(return_value=fake)
            fake.to = MagicMock(return_value=fake)
            fake.side_effect = lambda t: torch.tensor([[1.0, 0.0]], dtype=torch.float32)
            return fake

        _install_fake_pyannote(monkeypatch, factory)
        _reload_module()

        from app.services.speaker_embedding import SpeakerEmbeddingModel

        model = SpeakerEmbeddingModel()
        audio = np.zeros(16000, dtype=np.float32)
        model.extract_embedding(audio, 16000)
        model.extract_embedding(audio, 16000)
        model.extract_embedding(audio, 16000)
        assert call_count["n"] == 1
