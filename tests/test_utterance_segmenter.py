"""Tests for utterance segmentation logic."""
import pytest
from app.services.utterance_segmenter import segment, UtteranceBoundary


def _word(text, start, end, speaker="SPEAKER_0"):
    return {"word": text, "start": start, "end": end, "speaker": speaker}


class TestSegmentBasic:
    def test_empty_words_returns_empty(self):
        assert segment([], 10.0) == []

    def test_single_word(self):
        words = [_word("hello", 0.0, 1.0)]
        result = segment(words, 5.0)
        assert len(result) == 1
        assert result[0].transcript_text == "hello"

    def test_continuous_speech_same_speaker(self):
        words = [
            _word("hello", 0.0, 0.5),
            _word("world", 0.6, 1.1),
            _word("test", 1.2, 1.7),
        ]
        result = segment(words, 5.0)
        assert len(result) == 1


class TestSpeakerChange:
    def test_splits_on_speaker_change(self):
        words = [
            _word("hello", 0.0, 1.0, "SPEAKER_0"),
            _word("hi", 1.1, 2.0, "SPEAKER_0"),
            _word("bye", 2.1, 3.0, "SPEAKER_1"),
            _word("later", 3.1, 4.0, "SPEAKER_1"),
        ]
        result = segment(words, 10.0)
        speakers = [u.speaker_id for u in result]
        assert "SPEAKER_0" in speakers
        assert "SPEAKER_1" in speakers


class TestSilenceGap:
    def test_splits_on_silence_gap(self):
        words = [
            _word("hello", 0.0, 1.0),
            _word("world", 1.1, 2.0),
            _word("keeps", 2.1, 3.0),
            _word("going", 3.1, 4.0),
            _word("more", 4.1, 5.5),
            _word("after", 6.5, 7.5),  # 1.0s gap > 0.5s
            _word("break", 7.6, 8.5),
            _word("here", 8.6, 9.5),
            _word("yes", 9.6, 10.5),
            _word("done", 10.6, 11.5),
        ]
        result = segment(words, 15.0)
        assert len(result) >= 2


class TestShortUtteranceMerge:
    def test_merges_short_same_speaker(self):
        words = [
            _word("a", 0.0, 0.5),
            _word("b", 1.5, 2.0),  # gap causes split, but both < 5s
        ]
        result = segment(words, 10.0)
        assert len(result) == 1

    def test_preserves_short_answer(self):
        words = [
            _word("hello", 0.0, 5.5, "SPEAKER_0"),
            _word("world", 5.6, 11.0, "SPEAKER_0"),
            _word("ok", 12.0, 12.5, "SPEAKER_1"),  # gap + speaker change
            _word("네", 13.0, 13.5, "SPEAKER_1"),  # short answer
        ]
        result = segment(words, 20.0)
        short_answers = [u for u in result if "네" in u.transcript_text]
        assert len(short_answers) >= 1


class TestLongUtteranceSplit:
    def test_splits_long_utterance(self):
        words = []
        for i in range(60):
            words.append(_word(f"word{i}", float(i), float(i) + 0.8))
        result = segment(words, 70.0)
        for u in result:
            assert u.duration_sec <= 35.0  # some tolerance


class TestCaseCBackchannelGuard:
    """Regression — Case C는 진짜 맞장구(SHORT_ANSWER_WORDS)만 병합해야 한다.

    원인 이슈: utterance_006 재현 케이스.
    SPEAKER_00의 긴 발화 중간에 SPEAKER_01이 0.86초짜리 짧은 질문 "같이 갈래요? 나랑?"을
    던졌을 때, 과거 Case C가 1초 미만 조건만 보고 무조건 이전 화자에 병합시켜
    화자 분리가 안 되던 문제.
    """

    def test_short_content_different_speaker_stays_independent(self):
        # Arrange — SPEAKER_00의 긴 발화 사이에 SPEAKER_01의 짧은 질문 (0.86s, 3단어)
        # 이후 다시 SPEAKER_00의 짧은 발화가 이어짐 (Case B도 안 탐)
        words = [
            # SPEAKER_00 (3.72s)
            _word("네", 36.60, 36.92, "SPEAKER_00"),
            _word("이쪽", 36.94, 37.20, "SPEAKER_00"),
            _word("부천", 37.30, 37.52, "SPEAKER_00"),
            _word("쪽으로", 37.62, 37.92, "SPEAKER_00"),
            _word("좀", 38.02, 38.10, "SPEAKER_00"),
            _word("알아봐야", 38.16, 38.54, "SPEAKER_00"),
            _word("될", 38.62, 38.66, "SPEAKER_00"),
            _word("것", 38.68, 38.70, "SPEAKER_00"),
            _word("같다", 38.76, 38.80, "SPEAKER_00"),
            _word("생각합니다", 38.82, 40.32, "SPEAKER_00"),
            # SPEAKER_01 짧은 질문 (0.86s, 3단어 — 콘텐츠, 맞장구 아님)
            _word("같이", 40.46, 40.70, "SPEAKER_01"),
            _word("갈래요", 40.76, 41.02, "SPEAKER_01"),
            _word("나랑", 41.10, 41.32, "SPEAKER_01"),
            # silence 2.9s
            # SPEAKER_00 짧은 발화 (1.24s)
            _word("그래주셔도", 44.33, 44.89, "SPEAKER_00"),
            _word("너무", 44.91, 44.97, "SPEAKER_00"),
            _word("감사하죠", 45.05, 45.57, "SPEAKER_00"),
        ]
        # Act
        result = segment(words, 60.0)

        # Assert — SPEAKER_01의 짧은 질문이 독립 utterance로 유지돼야 함
        spk01 = [u for u in result if u.speaker_id == "SPEAKER_01"]
        assert len(spk01) >= 1, "SPEAKER_01 utterance가 SPEAKER_00에 병합되면 안 됨"
        spk01_text = " ".join(u.transcript_text for u in spk01)
        assert "같이" in spk01_text
        assert "갈래요" in spk01_text

    def test_short_content_single_word_different_speaker_stays_independent(self):
        # Arrange — SPEAKER_01이 단일 콘텐츠 단어("와이프가")를 짧게 삽입 (0.4s)
        # 과거 Case C는 1초 미만 + 화자 다름 조건만 보고 SPEAKER_00에 병합했음
        words = [
            _word("저는", 0.0, 0.5, "SPEAKER_00"),
            _word("그냥", 0.6, 1.0, "SPEAKER_00"),
            _word("보통", 1.1, 1.5, "SPEAKER_00"),
            _word("이렇게", 1.6, 2.0, "SPEAKER_00"),
            _word("해요", 2.1, 2.5, "SPEAKER_00"),
            # SPEAKER_01 콘텐츠 단어 (0.4s, 1단어, SHORT_ANSWER_WORDS 아님)
            _word("와이프가", 2.70, 3.10, "SPEAKER_01"),
            # silence + SPEAKER_00
            _word("다시", 4.00, 4.40, "SPEAKER_00"),
            _word("이어집니다", 4.50, 5.20, "SPEAKER_00"),
        ]
        result = segment(words, 10.0)
        spk01 = [u for u in result if u.speaker_id == "SPEAKER_01"]
        assert len(spk01) >= 1, "콘텐츠 단어는 1단어여도 독립 utterance로 보존돼야 함"
        assert any("와이프가" in u.transcript_text for u in spk01)

    def test_short_answer_preserved_as_independent(self):
        # "네" 같은 backchannel이 SHORT_ANSWER_MIN_SEC(0.3s) 이상이면
        # 기존 _is_short_answer 경로로 독립 utterance 유지되는지 확인 (기존 동작 회귀 방지)
        words = [
            _word("긴", 0.0, 1.0, "SPEAKER_00"),
            _word("발화가", 1.1, 2.0, "SPEAKER_00"),
            _word("있어요", 2.1, 3.0, "SPEAKER_00"),
            _word("말이죠", 3.1, 4.0, "SPEAKER_00"),
            # SPEAKER_01 맞장구 (0.4s, 단독, SHORT_ANSWER_WORDS)
            _word("네", 4.20, 4.60, "SPEAKER_01"),
            _word("계속", 4.80, 5.30, "SPEAKER_00"),
            _word("이어집니다", 5.40, 6.00, "SPEAKER_00"),
        ]
        result = segment(words, 10.0)
        spk01 = [u for u in result if u.speaker_id == "SPEAKER_01"]
        assert len(spk01) >= 1, "SHORT_ANSWER_MIN_SEC 이상의 맞장구는 독립 유지 (기존 동작)"

    def test_micro_backchannel_below_short_answer_min_merged(self):
        # 초단기 맞장구 (< SHORT_ANSWER_MIN_SEC 0.3s) 는 _is_short_answer 탈락 후
        # _is_backchannel 경로로 이전 발화에 병합됨
        words = [
            _word("긴", 0.0, 1.0, "SPEAKER_00"),
            _word("발화가", 1.1, 2.0, "SPEAKER_00"),
            _word("있어요", 2.1, 3.0, "SPEAKER_00"),
            _word("말이죠", 3.1, 4.0, "SPEAKER_00"),
            # SPEAKER_01 초단기 맞장구 (0.2s, SHORT_ANSWER_MIN_SEC 미만)
            _word("네", 4.10, 4.30, "SPEAKER_01"),
            _word("계속", 4.50, 5.00, "SPEAKER_00"),
            _word("이어집니다", 5.10, 5.70, "SPEAKER_00"),
        ]
        result = segment(words, 10.0)
        spk01 = [u for u in result if u.speaker_id == "SPEAKER_01"]
        assert len(spk01) == 0, "0.3초 미만 초단기 맞장구는 이전 발화에 병합"

    def test_three_word_short_content_stays_independent(self):
        # 3단어 콘텐츠는 _is_short_answer(2단어 제한)도 _is_backchannel(2단어 제한)도
        # 모두 걸리지 않으므로 else: 경로로 독립 유지됨
        words = [
            _word("길게", 0.0, 2.0, "SPEAKER_00"),
            _word("말하고", 2.1, 4.0, "SPEAKER_00"),
            _word("있어요", 4.1, 5.5, "SPEAKER_00"),
            _word("같이", 5.70, 5.90, "SPEAKER_01"),
            _word("갈래요", 5.95, 6.20, "SPEAKER_01"),
            _word("나랑", 6.30, 6.50, "SPEAKER_01"),
            _word("그래서", 6.70, 7.20, "SPEAKER_00"),
            _word("말이에요", 7.30, 8.00, "SPEAKER_00"),
        ]
        result = segment(words, 10.0)
        spk01 = [u for u in result if u.speaker_id == "SPEAKER_01"]
        assert len(spk01) >= 1, "3단어 콘텐츠는 독립 유지돼야 함 (utterance_006 패턴)"
        assert any("같이" in u.transcript_text for u in spk01)


class TestPadding:
    def test_applies_padding(self):
        words = [_word("hello", 1.0, 2.0)]
        result = segment(words, 5.0)
        assert result[0].padded_start_sec < result[0].start_sec
        assert result[0].padded_end_sec > result[0].end_sec

    def test_padding_clamps_to_zero(self):
        words = [_word("hello", 0.05, 0.5)]
        result = segment(words, 5.0)
        assert result[0].padded_start_sec >= 0.0

    def test_padding_clamps_to_duration(self):
        words = [_word("hello", 4.8, 5.0)]
        result = segment(words, 5.0)
        assert result[0].padded_end_sec <= 5.0
