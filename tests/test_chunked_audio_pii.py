import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path
from app.stt_processor import _transcribe_chunked

@pytest.fixture
def mock_chunk_deps():
    with patch("app.stt_processor._detect_silence_points") as mock_silence, \
         patch("app.stt_processor._extract_chunk") as mock_extract, \
         patch("whisperx.load_audio") as mock_load, \
         patch("app.stt_processor.preprocess") as mock_preprocess, \
         patch("app.stt_processor._transcribe_chunk") as mock_trans_chunk, \
         patch("app.stt_processor.emit_chunk_utterances") as mock_emit:
        
        # 10s audio, split at 5.0s
        mock_silence.return_value = [5.0]
        mock_load.return_value = np.zeros(16000 * 5)
        mock_preprocess.side_effect = lambda x, sr: x
        
        # Each chunk has same segment for simplicity
        mock_trans_chunk.return_value = [{
            "text": "전화번호는 010-1111-2222입니다.",
            "start": 1.0,
            "end": 4.0,
            "words": [
                {"word": "전화번호는", "start": 1.0, "end": 1.5},
                {"word": "010-1111-2222입니다.", "start": 1.5, "end": 4.0}
            ]
        }]
        
        mock_emit.return_value = ([], {}, 0)
        
        yield {
            "trans_chunk": mock_trans_chunk
        }

def test_transcribe_chunked_with_audio_pii_masking(mock_chunk_deps, tmp_path):
    # Total duration 10s
    file_path = Path("dummy.wav")
    
    # We use wraps to let it actually return something but still track calls
    from app.services.audio_pii_masker import mask_audio_ranges
    with patch("app.stt_processor.mask_audio_ranges", wraps=mask_audio_ranges) as mock_mask, \
         patch("app.config.CHUNK_DURATION_SEC", 5), \
         patch("app.config.CHUNK_MARGIN_SEC", 2):
        
        segments, utterances, audio_files, pii_ranges = _transcribe_chunked(
            file_path,
            task_id="chunk_test",
            total_duration=10.0,
            enable_diarize=False,
            mask_audio_pii=True
        )
        
    print(f"\nDetected PII Ranges: {pii_ranges}")
    print(f"Mask Call Count: {mock_mask.call_count}")
    print(f"Trans Chunk Call Count: {mock_chunk_deps['trans_chunk'].call_count}")
    
    assert len(pii_ranges) == 2 # One per chunk
    # Chunk 1 (0-5s): PII word starts at 1.5s -> global 1.5s
    # Chunk 2 (5-10s): PII word starts at 1.5s -> global 5.0 + 1.5 = 6.5s
    # Default pad 0.15s
    assert pii_ranges[0][0] == pytest.approx(1.35)
    assert pii_ranges[1][0] == pytest.approx(6.35)
    assert mock_mask.call_count == 2
