import numpy as np
import pytest
from app.services.audio_pii_masker import find_pii_word_ranges, mask_audio_ranges

def test_find_pii_word_ranges():
    segments = [{
        "text": "제 번호는 010-1234-5678입니다.",
        "words": [
            {"word": "제", "start": 0.0, "end": 0.5},
            {"word": "번호는", "start": 0.5, "end": 1.0},
            {"word": "010-1234-5678입니다.", "start": 1.0, "end": 2.5}
        ]
    }]
    
    # "010-1234-5678" span in text "제 번호는 010-1234-5678입니다."
    # 01234567890123456789012
    # 제 번호는 010-1234-5678입니다.
    #        ^6           ^19
    
    # word_offsets will be:
    # "제" [0, 1]
    # "번호는" [2, 5]
    # "010-1234-5678입니다." [6, 23]
    
    ranges = find_pii_word_ranges(segments, pad_sec=0.0)
    assert len(ranges) == 1
    assert ranges[0][0] == 1.0 # start
    assert ranges[0][1] == 2.5 # end
    assert ranges[0][2] == "전화번호"

def test_mask_audio_ranges_beep():
    sr = 16000
    audio = np.zeros(sr * 3) # 3 seconds of silence
    ranges = [(1.0, 2.0, "PII")] # 1 second beep
    
    masked = mask_audio_ranges(audio, ranges, sr)
    
    # 마스킹된 구간이 0이 아니어야 함
    assert np.any(masked[sr:2*sr] != 0)
    # 마스킹되지 않은 구간은 0이어야 함 (fade 구간 제외 여유 있게)
    assert np.all(masked[0:sr-200] == 0)
    assert np.all(masked[2*sr+200:] == 0)
    
    # 1kHz 비프음인지 FFT로 검증
    # fade 구간을 피해서 중간 샘플 채취
    start_sample = sr + sr//4
    end_sample = 2*sr - sr//4
    beep_part = masked[start_sample:end_sample]
    
    fft = np.fft.fft(beep_part)
    freqs = np.fft.fftfreq(len(beep_part), 1/sr)
    peak_freq = abs(freqs[np.argmax(abs(fft))])
    assert 990 <= peak_freq <= 1010


def test_find_pii_word_ranges_skips_unfound_words():
    """segment.text에 없는 word(정렬 실패 등)가 있어도 뒤따르는 word의
    offset이 오염되지 않아야 한다. H1 회귀 테스트.
    """
    segments = [{
        "text": "번호는 010-1234-5678 입니다",
        "words": [
            {"word": "번호는", "start": 0.0, "end": 0.5},
            {"word": "XXXMISSINGXXX", "start": 0.5, "end": 0.6},  # segment.text에 없음
            {"word": "010-1234-5678", "start": 0.6, "end": 2.0},
            {"word": "입니다", "start": 2.0, "end": 2.5},
        ],
    }]
    ranges = find_pii_word_ranges(segments, pad_sec=0.0)
    assert len(ranges) == 1
    # 010-1234-5678 word의 timestamp가 정확히 반영되어야 한다
    assert ranges[0][0] == 0.6
    assert ranges[0][1] == 2.0
    assert ranges[0][2] == "전화번호"


def test_find_pii_word_ranges_empty_words():
    """words 배열이 없거나 비어 있으면 PII 매핑 스킵."""
    segments = [
        {"text": "010-1234-5678 전화", "words": []},
        {"text": "010-1234-5678 전화"},  # words 키 없음
    ]
    assert find_pii_word_ranges(segments) == []


def test_mask_audio_ranges_empty_ranges_returns_input():
    """빈 ranges면 원본을 그대로 반환 (copy 여부는 구현 선택)."""
    sr = 16000
    audio = np.ones(sr, dtype=np.float32)
    masked = mask_audio_ranges(audio, [], sr)
    assert np.array_equal(masked, audio)


def test_mask_audio_ranges_out_of_bounds_clamped():
    """오디오 길이를 벗어난 range는 경계 내로 clamp."""
    sr = 16000
    audio = np.zeros(sr * 2, dtype=np.float32)
    ranges = [(1.5, 10.0, "PII")]  # end가 오디오 끝(2s) 초과
    masked = mask_audio_ranges(audio, ranges, sr)
    # 앞 1.5초는 여전히 0
    assert np.all(masked[:int(1.5 * sr) - 100] == 0)
    # 1.5초 이후는 비프로 치환됨
    assert np.any(masked[int(1.5 * sr) + 200:] != 0)
