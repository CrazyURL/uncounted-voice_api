"""PII masking service — thin wrapper over pii_masker module.

Preserves all existing masking logic (Korean names, phone numbers,
resident IDs, etc.) without modification.
"""

from app.pii_masker import mask_pii, mask_segments


class PIIService:
    """PII detection and masking service."""

    @staticmethod
    def mask_text(text: str, enable_name_masking: bool = False) -> dict:
        """Mask PII in a single text string."""
        return mask_pii(text, enable_name_masking)

    @staticmethod
    def mask_segments(segments: list[dict], enable_name_masking: bool = False) -> list[dict]:
        """Mask PII in a list of transcript segments."""
        return mask_segments(segments, enable_name_masking)


pii_service = PIIService()
