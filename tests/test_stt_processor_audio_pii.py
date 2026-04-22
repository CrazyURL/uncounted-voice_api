import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path
from app.stt_processor import transcribe

@pytest.fixture
def mock_whisperx_deps():
    with patch("whisperx.load_audio") as mock_load, \
         patch("subprocess.run") as mock_run, \
         patch("app.stt_processor._model") as mock_model, \
         patch("app.stt_processor._align_model") as mock_align_model, \
         patch("whisperx.align") as mock_align, \
         patch("app.stt_processor.job_store") as mock_job_store, \
         patch("app.stt_processor.preprocess") as mock_preprocess:
        
        mock_load.return_value = np.zeros(16000 * 5)
        mock_run.return_value = MagicMock(stdout="5.0")
        mock_preprocess.side_effect = lambda x, sr: x
        
        # Mock _model.transcribe
        mock_model.transcribe.return_value = {
            "segments": [{
                "text": "제 번호는 010-1234-5678입니다.",
                "start": 1.0,
                "end": 3.0
            }]
        }
        
        # Mock whisperx.align
        mock_align.return_value = {
            "segments": [{
                "text": "제 번호는 010-1234-5678입니다.",
                "start": 1.0,
                "end": 3.0,
                "words": [
                    {"word": "제", "start": 1.0, "end": 1.2},
                    {"word": "번호는", "start": 1.2, "end": 1.5},
                    {"word": "010-1234-5678입니다.", "start": 1.5, "end": 3.0}
                ]
            }]
        }
        
        yield {
            "model": mock_model,
            "align": mock_align
        }

def test_transcribe_with_audio_pii_masking(mock_whisperx_deps, tmp_path):
    dummy_file = tmp_path / "test_audio.wav"
    dummy_file.write_text("dummy content")
    
    # We need to mock Path.exists and stat for the dummy file
    with patch.object(Path, "exists", return_value=True), \
         patch.object(Path, "stat") as mock_stat, \
         patch("os.unlink"):
        
        mock_stat.return_value.st_size = 1000
        
        result = transcribe(
            str(dummy_file),
            task_id="test_task_123",
            mask_pii=True,
            mask_audio_pii=True
        )
    
    assert "pii_summary" in result
    found_phone = False
    for item in result["pii_summary"]:
        if item["type"] == "전화번호":
            found_phone = True
            assert "time_ranges" in item
            assert len(item["time_ranges"]) > 0
            # Word "010-1234-5678입니다." starts at 1.5
            # Default pad is 0.15 -> 1.5 - 0.15 = 1.35
            assert item["time_ranges"][0]["start"] == pytest.approx(1.35)
    
    assert found_phone
