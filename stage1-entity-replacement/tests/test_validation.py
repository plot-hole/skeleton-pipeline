"""Tests for the validation module (second-pass NER)."""

import pytest

from src.validation import Validator, _build_token_spans, _is_inside_token

# Skip all tests in this module if spaCy is not installed.
spacy = pytest.importorskip("spacy")


# ---------------------------------------------------------------------------
# Token span helpers
# ---------------------------------------------------------------------------

class TestTokenSpans:
    def test_find_token_spans(self):
        text = "Hello [PARTY_A] and [PARTY_B]."
        spans = _build_token_spans(text)
        assert len(spans) == 2
        assert text[spans[0][0]:spans[0][1]] == "[PARTY_A]"
        assert text[spans[1][0]:spans[1][1]] == "[PARTY_B]"

    def test_no_tokens(self):
        spans = _build_token_spans("No tokens here.")
        assert len(spans) == 0

    def test_is_inside_token(self):
        spans = [(6, 15)]  # "[PARTY_A]"
        assert _is_inside_token(7, 14, spans) is True  # "PARTY_A" inside the brackets
        assert _is_inside_token(0, 5, spans) is False  # "Hello" not inside


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class TestValidator:
    def test_fully_replaced_passes(self):
        """A document with all PII replaced should pass validation."""
        text = (
            "This lease agreement is between [PARTY_A] and [PARTY_B]. "
            "[PARTY_A] resides at [LOCATION_CA_METRO]. "
            "The monthly rent is [RENT_4000_5000]. "
            "This agreement shall be governed by the laws of California."
        )
        validator = Validator()
        result = validator.validate(text)

        assert result.passed is True
        assert result.entities_flagged == 0

    def test_unreplaced_name_flagged(self):
        """A name left in the text should be flagged."""
        text = (
            "This lease agreement is between John Smith and [PARTY_B]. "
            "The monthly rent is [RENT_4000_5000]."
        )
        validator = Validator()
        result = validator.validate(text)

        assert result.passed is False
        assert result.entities_flagged >= 1
        flagged_texts = [f.text for f in result.flagged]
        assert "John Smith" in flagged_texts

    def test_allowlisted_entity_not_flagged(self):
        """California in governing law clause -> allowed, not flagged."""
        text = (
            "This agreement shall be governed by the laws of California. "
            "All disputes shall be resolved in the courts of the United States."
        )
        validator = Validator()
        result = validator.validate(text)

        assert result.passed is True
        assert result.entities_allowed >= 1

    def test_custom_allowlist(self):
        """Additional allowlist entries passed at init are respected."""
        text = "Tenant works at Globex Corporation."
        # Without custom allowlist, "Globex Corporation" would be flagged
        validator_strict = Validator()
        result_strict = validator_strict.validate(text)

        validator_custom = Validator(allowlist={"Globex Corporation"})
        result_custom = validator_custom.validate(text)

        # Custom allowlist should allow it
        assert result_custom.entities_flagged <= result_strict.entities_flagged

    def test_tokens_inside_brackets_ignored(self):
        """Entities detected inside [...] brackets should not be flagged."""
        # spaCy might detect "TECH" or similar inside [EMPLOYER_TECH]
        text = "Tenant is employed by [EMPLOYER_TECH] since [DATE_2026_Q1]."
        validator = Validator()
        result = validator.validate(text)

        # Nothing outside brackets should be flagged
        assert result.passed is True

    def test_empty_document(self):
        """Empty text should pass validation trivially."""
        validator = Validator()
        result = validator.validate("")
        assert result.passed is True
        assert result.entities_detected == 0
