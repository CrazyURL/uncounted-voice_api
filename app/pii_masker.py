import re
from typing import Optional


# 한국 성씨 목록 (상위 빈도)
KOREAN_SURNAMES = (
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
    "한", "오", "서", "신", "권", "황", "안", "송", "류", "전",
    "홍", "고", "문", "양", "손", "배", "백", "허", "유", "남",
    "심", "노", "하", "곽", "성", "차", "주", "우", "구", "민",
)

# 성씨와 겹치는 고빈도 일상어 제외 (2글자 접두사)
# 이 접두사로 시작하는 모든 매칭을 건너뛴다.
# 예: "정신" 등록 → "정신", "정신과", "정신적" 모두 건너뜀
_NAME_EXCLUDE_PREFIX = frozenset({
    # 이~
    "이런", "이제", "이거", "이건", "이게", "이걸", "이것", "이날", "이번",
    "이미", "이후", "이전", "이상", "이하", "이유", "이름", "이용", "이동",
    "이야", "이렇", "이리", "이래", "이때", "이내", "이틀", "이해",
    "이어", "이를", "이른", "이룬", "이뤄", "이끌", "이외", "이며", "이면",
    "이라", "이랑", "이요", "이에", "이든", "이니", "이나", "이다", "이도",
    # 정~
    "정말", "정도", "정리", "정보", "정신", "정상", "정확", "정식", "정기",
    "정해", "정한", "정하", "정작", "정오", "정문", "정답", "정비", "정산",
    # 강~
    "강한", "강해", "강화", "강조", "강력", "강당", "강의", "강물", "강변",
    "강남", "강북", "강서", "강동", "강원",
    # 하~
    "하는", "하고", "하면", "하게", "하지", "하자", "하다", "하며", "하니",
    "하여", "하루", "하반", "하나", "하늘", "하얀", "하물", "하필", "하소",
    "하던", "하더", "하도", "하긴", "하기", "하네", "하세",
    "하락", "하루", "하산", "하수", "하위", "하차", "하한",
    # 조~
    "조금", "조용", "조건", "조사", "조치", "조차", "조절", "조만", "조기",
    # 장~
    "장소", "장면", "장기", "장래", "장비", "장점", "장애", "장난", "장마",
    # 한~
    "한번", "한데", "한참", "한편", "한동", "한때", "한층", "한결", "한마",
    "한다", "한두", "한쪽", "한테",
    # 안~
    "안녕", "안전", "안내", "안정", "안쪽", "안에", "안과", "안개",
    "안되", "안돼", "안나", "안해",
    # 오~
    "오늘", "오전", "오후", "오래", "오히", "오직", "오른", "오면", "오고",
    # 서~
    "서로", "서울", "서쪽", "서비", "서류", "서둘",
    # 고~
    "고객", "고민", "고생", "고마", "고장", "고향", "고르", "고른",
    # 송~
    "송도", "송이", "송출",
    # 문~
    "문제", "문의", "문서", "문화", "문자", "문득", "문밖",
    # 남~
    "남자", "남편", "남쪽", "남는", "남기", "남은", "남녀", "남부",
    # 배~
    "배우", "배달", "배경", "배치",
    # 신~
    "신경", "신청", "신용", "신호", "신규", "신발", "신기", "신선",
    # 손~
    "손님", "손잡", "손으", "손해",
    # 백~
    "백만", "백원", "백화",
    # 최~
    "최근", "최고", "최대", "최소", "최종", "최선", "최초", "최저",
    # 윤~
    "윤리",
    # 임~
    "임시", "임대", "임금",
    # 권~
    "권리", "권한",
    # 양~
    "양쪽", "양해",
    # 유~
    "유지", "유리", "유일", "유사", "유명", "유효",
    # 차~
    "차이", "차라", "차량", "차로", "차원", "차례",
    # 주~
    "주로", "주요", "주의", "주변", "주민", "주간", "주말", "주소", "주어",
    # 민~
    "민간", "민원",
    # 성~
    "성격", "성과", "성공", "성장",
    # 구~
    "구체", "구간", "구매", "구역",
    # 전~
    "전화", "전혀", "전체", "전부", "전달", "전문", "전국", "전자", "전기",
    "전환", "전통", "전략", "전날", "전반", "전용", "전망", "전제", "전선",
    # 홍~
    "홍보",
    # 황~
    "황당",
    # 노~
    "노력", "노래",
    # 배~  (보강)
    "배고",
    # 허~
    "허락",
    # 곽~  (드물지만)
    # 우~
    "우리", "우선",
})

# 한국 성씨 정규식 (컴파일됨) — 40개 성씨를 | 로 연결
_SURNAME_PATTERN = re.compile(
    f"({'|'.join(re.escape(s) for s in KOREAN_SURNAMES)})([가-힣]{{1,2}})"
)

# 호칭/직함 (이름 뒤에 올 수 있는 단어)
_HONORIFICS = (
    "씨", "님", "선생", "교수", "박사", "사장", "대표", "이사",
    "부장", "차장", "과장", "대리", "사원", "주임", "팀장", "실장",
    "원장", "국장", "처장", "위원", "총장", "학장", "선배", "후배",
)

# PII 패턴 정의 (순서 중요: 구체적인 패턴이 먼저)
PII_PATTERNS = [
    # 주민등록번호: 900101-1234567
    (
        re.compile(r"(\d{6})\s*[-]\s*([1-4]\d{6})"),
        lambda m: f"{m.group(1)}-*******",
        "주민등록번호",
    ),
    # 운전면허번호: 12-34-567890-12
    (
        re.compile(r"(\d{2})-(\d{2})-(\d{6})-(\d{2})"),
        lambda m: "**-**-******-**",
        "운전면허번호",
    ),
    # 여권번호: M12345678
    (
        re.compile(r"([A-Z])(\d{8})"),
        lambda m: f"{m.group(1)}********",
        "여권번호",
    ),
    # 카드번호: 1234-5678-9012-3456 또는 공백 구분
    (
        re.compile(r"(\d{4})[\s-](\d{4})[\s-](\d{4})[\s-](\d{4})"),
        lambda m: f"{m.group(1)}-****-****-{m.group(4)}",
        "카드번호",
    ),
    # 이메일
    (
        re.compile(r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"),
        lambda m: f"{m.group(1)[0]}***@{m.group(2)}",
        "이메일",
    ),
    # 전화번호: 010-1234-5678, 02-123-4567 등
    (
        re.compile(r"(0\d{1,2})[\s.-](\d{3,4})[\s.-](\d{4})"),
        lambda m: f"{m.group(1)}-****-{m.group(3)}",
        "전화번호",
    ),
    # 전화번호 (붙여쓰기): 01012345678
    (
        re.compile(r"(010)(\d{4})(\d{4})"),
        lambda m: f"{m.group(1)}****{m.group(3)}",
        "전화번호",
    ),
    # 계좌번호: 11~14자리 연속 숫자 (단어 경계, 최소 11자리로 상향)
    (
        re.compile(r"\b(\d{3})(\d{8,11})\b"),
        lambda m: f"{m.group(1)}{'*' * len(m.group(2))}",
        "계좌번호",
    ),
    # IP주소
    (
        re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b"),
        lambda m: "***.***.***.***",
        "IP주소",
    ),
]


def _matches_exclude_prefix(surname: str, given: str) -> bool:
    """성+이름이 제외 목록의 접두사와 일치하는지 확인한다.

    "정신"이 제외 목록에 있으면 "정신", "정신과", "정신적" 모두 제외된다.
    """
    full = surname + given
    # 정확히 일치 (2글자 매칭: 성+1글자)
    if full in _NAME_EXCLUDE_PREFIX:
        return True
    # 접두사 일치 (3글자 매칭: 성+2글자 중 성+첫글자가 제외 목록에 있으면)
    if len(given) >= 2 and (surname + given[0]) in _NAME_EXCLUDE_PREFIX:
        return True
    return False


def _is_likely_name_with_context(
    surname: str, given: str, before: str, after: str
) -> bool:
    """앞뒤 문맥을 포함해 이름 여부를 판단한다."""

    # 1. 제외 목록 (접두사 매칭)
    if _matches_exclude_prefix(surname, given):
        return False

    # 2. 단어 중간에서 매칭된 경우 제외
    #    이름은 공백/문장부호 뒤에 오거나 문장 시작이어야 함
    if before and before[-1] not in " \t\n,.:;!?()\"'·…—-~":
        return False

    # 3. 2글자 이름 (성+1글자): 뒤에 호칭이 올 때만 이름으로 판단
    if len(given) == 1:
        after_stripped = after.lstrip()
        for h in _HONORIFICS:
            if after_stripped.startswith(h):
                return True
        return False

    # 4. 3글자 이름 (성+2글자): 제외 목록을 통과하고 단어 시작이면 이름으로 간주
    return True


def detect_pii_spans(text: str, enable_name_masking: bool = False) -> list[dict]:
    """텍스트에서 PII의 위치(span)를 감지하고 리스트로 반환한다."""
    spans = []

    # 1. PII_PATTERNS 감지
    for pattern, _, label in PII_PATTERNS:
        for m in pattern.finditer(text):
            spans.append({
                "type": label,
                "char_start": m.start(),
                "char_end": m.end(),
                "matched_text": m.group(0)
            })

    # 2. 이름 마스킹 감지
    if enable_name_masking:
        for m in _SURNAME_PATTERN.finditer(text):
            s = m.group(1)
            g = m.group(2)
            before = text[:m.start()]
            after = text[m.end():]

            if _is_likely_name_with_context(s, g, before, after):
                spans.append({
                    "type": "이름",
                    "char_start": m.start(),
                    "char_end": m.end(),
                    "matched_text": m.group(0)
                })

    return spans


def mask_pii(text: str, enable_name_masking: bool = False) -> dict:
    """텍스트에서 PII를 마스킹하고 결과를 반환한다."""
    spans = detect_pii_spans(text, enable_name_masking)

    # 중첩된 span 처리: 시작 위치 순, 길이 역순으로 정렬
    spans.sort(key=lambda x: (x["char_start"], -(x["char_end"] - x["char_start"])))

    # 중첩 제거 (먼저 나온 긴 패턴 우선)
    non_overlapping = []
    last_end = -1
    for span in spans:
        if span["char_start"] >= last_end:
            non_overlapping.append(span)
            last_end = span["char_end"]

    # 역순 치환 (index 보존)
    non_overlapping.sort(key=lambda x: x["char_start"], reverse=True)

    masked_chars = list(text)
    detected_summary = {}

    for span in non_overlapping:
        label = span["type"]
        matched = span["matched_text"]
        start = span["char_start"]
        end = span["char_end"]

        # 치환값 계산
        replacer_val = None
        if label == "이름":
            s = matched[0]
            g = matched[1:]
            replacer_val = f"{s}{'O' * len(g)}"
        else:
            for p, r, l in PII_PATTERNS:
                if l == label:
                    m = p.fullmatch(matched)
                    if m:
                        replacer_val = r(m) if callable(r) else r
                        break
            if not replacer_val:
                replacer_val = "*" * len(matched)

        # 치환 적용
        masked_chars[start:end] = list(replacer_val)

        # 요약 업데이트
        detected_summary[label] = detected_summary.get(label, 0) + 1

    # PII_PATTERNS 선언 순서 + 이름을 마지막에 배치 (중복 라벨 제거, 하위 호환)
    seen = set()
    pattern_order: list[str] = []
    for _, _, label in PII_PATTERNS:
        if label not in seen:
            seen.add(label)
            pattern_order.append(label)
    pattern_order.append("이름")
    pii_detected = [
        {"type": label, "count": detected_summary[label]}
        for label in pattern_order if label in detected_summary
    ]

    return {
        "masked_text": "".join(masked_chars),
        "pii_detected": pii_detected,
        "total_masked": sum(detected_summary.values()),
    }


def mask_segments(
    segments: list[dict], enable_name_masking: bool = False
) -> list[dict]:
    """세그먼트 리스트의 텍스트를 마스킹한다."""
    total_pii = []
    for seg in segments:
        result = mask_pii(seg.get("text", ""), enable_name_masking)
        seg["text"] = result["masked_text"]
        total_pii.extend(result["pii_detected"])
    return total_pii
