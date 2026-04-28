"""v3.0 PII 회귀 테스트 — 알파 샘플에서 노출됐던 한국어 이름 마스킹 실패 차단.

검증 대상:
- 김용철 케이스 ("김용철 형한테") — 3글자 이름 + 일상 호칭 "형"
- mask_segments 타입별 count 합산 (다중 segment에 같은 PII 유형)
- 일상 호칭(형/누나/언니/오빠/엄마/아빠 등) 인식
"""
from __future__ import annotations

import pytest

from app.pii_masker import (
    _HONORIFICS,
    detect_pii_spans,
    mask_pii,
    mask_segments,
)


# ─────────────────────────────────────────────────────────────────
# Bug B: 김용철 케이스 + 일상 호칭 인식
# ─────────────────────────────────────────────────────────────────


class TestNameMaskingWithInformalHonorifics:
    """알파 샘플에서 "김용철 형한테"가 마스킹 안 됐던 문제 회귀."""

    def test_kim_yongchul_with_hyung_honorific(self) -> None:
        """3글자 이름 + "형" 호칭 — 알파 샘플 회귀."""
        text = "그거 이제 저기 김용철 형한테 줬어?"
        result = mask_pii(text, enable_name_masking=True)
        # "김용철"이 "김OO"로 마스킹되어야 함
        assert "김용철" not in result["masked_text"], (
            f"김용철이 그대로 남음: {result['masked_text']!r}"
        )
        assert any(p["type"] == "이름" for p in result["pii_detected"]), (
            "PII detection에 이름 누락"
        )

    @pytest.mark.parametrize(
        "name,honorific",
        [
            ("김용철", "형"),
            ("이민호", "형님"),
            ("박지영", "누나"),
            ("최지원", "언니"),
            ("정현우", "오빠"),
            ("강예지", "동생"),
        ],
    )
    def test_3char_name_with_informal_honorific(
        self, name: str, honorific: str
    ) -> None:
        """3글자 한국 이름 + 일상 호칭 (가족·친밀 관계) — 모두 마스킹."""
        text = f"{name} {honorific}한테 전달해 주세요"
        result = mask_pii(text, enable_name_masking=True)
        assert name not in result["masked_text"], (
            f"{name}이 마스킹 안 됨: {result['masked_text']!r}"
        )

    @pytest.mark.parametrize(
        "honorific",
        ["형", "형님", "누나", "언니", "오빠", "동생",
         "어머니", "아버지", "엄마", "아빠", "할머니", "할아버지",
         "아주머니", "아저씨", "삼촌", "이모", "고모", "외삼촌"],
    )
    def test_informal_honorific_in_dictionary(self, honorific: str) -> None:
        """일상 호칭이 모두 사전에 등록됨."""
        assert honorific in _HONORIFICS, f"{honorific!r}가 _HONORIFICS에 없음"

    def test_2char_name_with_hyung_honorific(self) -> None:
        """2글자 이름 + 일상 호칭 (성+1글자)도 마스킹."""
        # "김철 형" — 김(성) + 철(1글자 이름) + 형(호칭)
        text = "김철 형이 왔어요"
        result = mask_pii(text, enable_name_masking=True)
        assert "김철" not in result["masked_text"], (
            f"김철이 마스킹 안 됨: {result['masked_text']!r}"
        )

    def test_3char_name_without_honorific_still_masked(self) -> None:
        """3글자 이름은 호칭 없어도 마스킹 (기존 동작 보장)."""
        text = "김용철이 와서 말했다"
        result = mask_pii(text, enable_name_masking=True)
        assert "김용철" not in result["masked_text"]

    def test_name_masking_disabled_keeps_name(self) -> None:
        """enable_name_masking=False면 이름 그대로 (기존 동작)."""
        text = "김용철 형한테 줬어"
        result = mask_pii(text, enable_name_masking=False)
        # 이름은 그대로 남음 (전화번호 등 다른 PII만 마스킹)
        assert "김용철" in result["masked_text"]


# ─────────────────────────────────────────────────────────────────
# Bug A: mask_segments 타입별 count 합산 (중복 dict 제거)
# ─────────────────────────────────────────────────────────────────


class TestMaskSegmentsTypeAggregation:
    """여러 segment에 같은 PII 유형이 있을 때 count 합산되는지 검증."""

    def test_same_type_in_multiple_segments_aggregates(self) -> None:
        """전화번호가 2개 segment에 각각 있으면 count: 2로 합산."""
        segments = [
            {"text": "제 번호는 010-1234-5678입니다."},
            {"text": "다른 번호는 010-9999-8888"},
        ]
        result = mask_segments(segments, enable_name_masking=False)
        phone_items = [r for r in result if r["type"] == "전화번호"]
        assert len(phone_items) == 1, (
            f"전화번호 항목이 중복됨: {result}"
        )
        assert phone_items[0]["count"] == 2

    def test_different_types_in_multiple_segments(self) -> None:
        """전화번호 + 이메일이 다른 segment에 있으면 각각 1건씩."""
        segments = [
            {"text": "전화 010-1234-5678"},
            {"text": "이메일 user@example.com"},
        ]
        result = mask_segments(segments, enable_name_masking=False)
        types = {r["type"]: r["count"] for r in result}
        assert types.get("전화번호") == 1
        assert types.get("이메일") == 1

    def test_segments_text_in_place_modified(self) -> None:
        """segment[].text가 in-place로 마스킹된 텍스트로 교체됨 (기존 동작)."""
        segments = [{"text": "전화 010-1234-5678"}]
        mask_segments(segments, enable_name_masking=False)
        assert "010-1234-5678" not in segments[0]["text"]

    def test_empty_segments(self) -> None:
        segments: list[dict] = []
        result = mask_segments(segments, enable_name_masking=False)
        assert result == []

    def test_no_pii_segments(self) -> None:
        segments = [{"text": "안녕하세요 반갑습니다"}]
        result = mask_segments(segments, enable_name_masking=False)
        assert result == []

    def test_response_schema_compliance(self) -> None:
        """응답 스키마 PIIDetectedItem (type + count) 계약 유지."""
        segments = [
            {"text": "전화 010-1111-2222"},
            {"text": "전화 010-3333-4444"},
            {"text": "이메일 a@b.com"},
        ]
        result = mask_segments(segments, enable_name_masking=False)
        # 각 항목은 type + count 키만 가짐
        for item in result:
            assert set(item.keys()) == {"type", "count"}, item
            assert isinstance(item["type"], str)
            assert isinstance(item["count"], int)
            assert item["count"] > 0
        # 동일 type 중복 없음
        types = [r["type"] for r in result]
        assert len(types) == len(set(types)), f"type 중복: {types}"


# ─────────────────────────────────────────────────────────────────
# 통합 — detect_pii_spans 정합성
# ─────────────────────────────────────────────────────────────────


class TestPiiSpanDetection:
    def test_kim_yongchul_span_detected(self) -> None:
        """detect_pii_spans 단계에서도 김용철 잡혀야 함."""
        spans = detect_pii_spans("김용철 형한테 줬어", enable_name_masking=True)
        name_spans = [s for s in spans if s.get("type") == "이름"]
        assert len(name_spans) >= 1, f"이름 span 없음: {spans}"
        # 김용철 위치 검증
        assert any(
            s["matched_text"] == "김용철" for s in name_spans
        ), f"김용철 매칭 없음: {name_spans}"
