from app.services.pii_service import pii_service


class TestPIIMasking:
    def test_phone_number(self):
        result = pii_service.mask_text("전화번호는 010-1234-5678입니다")
        assert "010-****-5678" in result["masked_text"]
        assert result["total_masked"] == 1

    def test_resident_number(self):
        result = pii_service.mask_text("주민번호 900101-1234567")
        assert "900101-*******" in result["masked_text"]

    def test_email(self):
        result = pii_service.mask_text("메일은 test@example.com입니다")
        assert "t***@example.com" in result["masked_text"]

    def test_card_number(self):
        result = pii_service.mask_text("카드 1234-5678-9012-3456")
        assert "1234-****-****-3456" in result["masked_text"]

    def test_name_masking_enabled(self):
        result = pii_service.mask_text("김철수씨가 왔습니다", enable_name_masking=True)
        assert "김OO" in result["masked_text"]

    def test_name_masking_disabled(self):
        result = pii_service.mask_text("김철수씨가 왔습니다", enable_name_masking=False)
        assert "김철수" in result["masked_text"]

    def test_exclude_common_words(self):
        result = pii_service.mask_text("이번에 최근 정보를 확인", enable_name_masking=True)
        assert "이번" in result["masked_text"]
        assert "최근" in result["masked_text"]

    def test_segments_masking(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "전화번호 010-9999-8888"},
            {"start": 1.0, "end": 2.0, "text": "이메일 abc@test.com"},
        ]
        pii_summary = pii_service.mask_segments(segments)
        assert len(pii_summary) == 2
        assert "010-****-8888" in segments[0]["text"]
