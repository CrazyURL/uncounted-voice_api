"""WeSpeaker speaker embedding via pyannote.audio (Phase 4: Option B).

Lazy-loads `pyannote.audio.Model.from_pretrained(repo)` and extracts speaker
embeddings via direct PyTorch inference. Outputs L2-normalized embeddings for
use in speaker reclustering.

Pivoted from `onnxruntime` to PyTorch direct calls — pyannote/wespeaker model's
forward() uses `torch.vmap(torchaudio.compliance.kaldi.fbank)` which is
incompatible with both legacy TorchScript and dynamo ONNX exporters as of
PyTorch 2.x. pyannote.audio is already in voice-api requirements; this avoids
ONNX conversion friction entirely.

Frozen interface (do not change):
    EmbeddingUnavailable(reason: str)
    SpeakerEmbeddingModel.extract_embedding(audio, sample_rate)
        -> np.ndarray | EmbeddingUnavailable
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

_TARGET_SAMPLE_RATE = 16_000
_MIN_AUDIO_SAMPLES = _TARGET_SAMPLE_RATE // 2  # 0.5초
_DEFAULT_REPO = "pyannote/wespeaker-voxceleb-resnet34-LM"


@dataclass(frozen=True)
class EmbeddingUnavailable:
    """Typed sentinel when embedding extraction is not possible.

    Attributes:
        reason: One of
            - "model_missing"     — repo 미설정 또는 from_pretrained 실패
            - "import_failed"     — torch 또는 pyannote.audio import 실패
            - "audio_too_short"   — 입력이 0.5초 미만
            - "extraction_failed" — forward 호출 중 예외
    """

    reason: str


class SpeakerEmbeddingModel:
    """pyannote/wespeaker 임베딩 추출 wrapper.

    `extract_embedding` 첫 호출 시 모델을 lazy-load한다. 이후 호출은 캐시된
    모델 또는 실패 sentinel을 재사용한다.
    """

    def __init__(self) -> None:
        self._model = None
        self._torch_module = None
        self._device = None
        self._loaded = False
        self._unavailable: EmbeddingUnavailable | None = None

    def _load(self) -> EmbeddingUnavailable | None:
        if self._loaded:
            return self._unavailable

        try:
            import torch
            from pyannote.audio import Model
        except (ImportError, ModuleNotFoundError):
            self._unavailable = EmbeddingUnavailable("import_failed")
            self._loaded = True
            return self._unavailable

        repo = os.environ.get("VOICE_DIARIZATION_WESPEAKER_REPO", _DEFAULT_REPO)
        if not repo:
            self._unavailable = EmbeddingUnavailable("model_missing")
            self._loaded = True
            return self._unavailable

        provider = os.environ.get("VOICE_DIARIZATION_EMBEDDING_PROVIDER", "cpu").lower()
        device_name = "cpu"
        if provider == "cuda":
            try:
                if torch.cuda.is_available():
                    device_name = "cuda"
            except Exception:
                device_name = "cpu"

        try:
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                model = Model.from_pretrained(repo, use_auth_token=hf_token)
            else:
                model = Model.from_pretrained(repo)
            model.eval()
            model.to(device_name)
        except Exception:
            self._unavailable = EmbeddingUnavailable("model_missing")
            self._loaded = True
            return self._unavailable

        self._model = model
        self._device = device_name
        self._torch_module = torch
        self._loaded = True
        return None

    def extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray | EmbeddingUnavailable:
        unavailable = self._load()
        if unavailable is not None:
            return unavailable

        if audio.ndim == 1:
            num_samples = len(audio)
        else:
            num_samples = audio.shape[-1]
        if num_samples < _MIN_AUDIO_SAMPLES:
            return EmbeddingUnavailable("audio_too_short")

        # mono mixdown
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # int → float32 in [-1, 1]
        if audio.dtype != np.float32:
            if np.issubdtype(audio.dtype, np.integer):
                info = np.iinfo(audio.dtype)
                audio = audio.astype(np.float32) / max(abs(info.min), abs(info.max))
            else:
                audio = audio.astype(np.float32)

        # 선형 보간 resample
        if sample_rate != _TARGET_SAMPLE_RATE:
            ratio = _TARGET_SAMPLE_RATE / sample_rate
            num_resampled = int(len(audio) * ratio)
            old_indices = np.arange(len(audio))
            new_indices = np.linspace(0, len(audio) - 1, num_resampled)
            audio = np.interp(new_indices, old_indices, audio).astype(np.float32)

        torch = self._torch_module
        try:
            tensor = torch.from_numpy(audio).reshape(1, 1, -1).to(self._device)
            with torch.no_grad():
                output = self._model(tensor)
        except Exception:
            return EmbeddingUnavailable("extraction_failed")

        if hasattr(output, "detach"):
            embedding = output.detach().cpu().numpy().astype(np.float32)
        else:
            embedding = np.asarray(output, dtype=np.float32)

        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(axis=0)

        norm = float(np.linalg.norm(embedding))
        if norm > 1e-12:
            embedding = embedding / norm

        return embedding
