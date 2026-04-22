import pytest
from app.pii_masker import detect_pii_spans, mask_pii

def test_detect_pii_spans_basic():
    text = "제 번호는 010-1234-5678입니다."
    spans = detect_pii_spans(text)
    assert len(spans) == 1
    assert spans[0]["type"] == "전화번호"
    assert spans[0]["matched_text"] == "010-1234-5678"
    assert text[spans[0]["char_start"]:spans[0]["char_end"]] == "010-1234-5678"

def test_detect_pii_spans_multiple():
    text = "이메일은 test@example.com이고, 카드는 1234-5678-9012-3456입니다."
    spans = detect_pii_spans(text)
    # Sort by start for assertion
    spans.sort(key=lambda x: x["char_start"])
    assert len(spans) == 2
    assert spans[0]["type"] == "이메일"
    assert spans[0]["matched_text"] == "test@example.com"
    assert spans[1]["type"] == "카드번호"
    assert spans[1]["matched_text"] == "1234-5678-9012-3456"

def test_detect_pii_spans_names():
    text = "홍길동 씨와 김철수 님을 만났습니다."
    spans = detect_pii_spans(text, enable_name_masking=True)
    
    names = [s["matched_text"] for s in spans if s["type"] == "이름"]
    assert "홍길동" in names
    assert "김철수" in names
    assert len(names) == 2

def test_mask_pii_equivalence():
    text = "제 번호는 010-1234-5678이고 주소는 abc@def.com입니다. 홍길동 씨도요."
    # '이메일' -> '주소' to avoid false positive name match
    result = mask_pii(text, enable_name_masking=True)

    print(f"\nDetected PII: {result['pii_detected']}")
    assert "010-****-5678" in result["masked_text"]
    assert "a***@def.com" in result["masked_text"]
    assert "홍OO" in result["masked_text"]
    assert result["total_masked"] == 3


def test_pii_detected_order_follows_patterns():
    """pii_detected는 PII_PATTERNS 선언 순서를 따라야 한다.
    주민번호 → 운전면허 → 여권 → 카드 → 이메일 → 전화번호 → 계좌 → IP → 이름
    """
    text = (
        "전화 010-1234-5678, 주민 900101-1234567, "
        "이메일 a@b.com, 홍길동 씨"
    )
    result = mask_pii(text, enable_name_masking=True)
    types = [d["type"] for d in result["pii_detected"]]
    # 주민번호가 전화번호보다 먼저, 이름이 마지막
    assert types.index("주민등록번호") < types.index("전화번호")
    assert types.index("이메일") < types.index("이름")
    assert types[-1] == "이름"


def test_detect_pii_spans_empty_text():
    assert detect_pii_spans("") == []
    assert detect_pii_spans("   ") == []


def test_detect_pii_spans_no_duplicates_for_phone_patterns():
    """전화번호 패턴이 두 번(하이픈/붙여쓰기) 정의돼 있지만
    pii_detected에서는 한 타입으로만 집계되어야 한다."""
    text = "전화 010-1234-5678 또는 01099998888"
    result = mask_pii(text)
    phone_entries = [d for d in result["pii_detected"] if d["type"] == "전화번호"]
    assert len(phone_entries) == 1
    assert phone_entries[0]["count"] == 2
