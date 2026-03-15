"""Second-pass NER validation.

Runs spaCy NER on the fully replaced document text to catch any PII that
the replacement engine missed.  Entities inside ``[...]`` token brackets
and those on the allowlist are excluded from flagging.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import yaml

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "config")

# Lazy-loaded spaCy model
_nlp = None


def _get_nlp():
    """Lazy-load spaCy model on first use."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
        except ImportError as exc:
            raise ImportError(
                "spaCy is required for validation. "
                "Install with: pip install -r requirements-nlp.txt && "
                "python -m spacy download en_core_web_lg"
            ) from exc
        _nlp = spacy.load("en_core_web_lg")
    return _nlp


# ---------------------------------------------------------------------------
# Allowlist loading
# ---------------------------------------------------------------------------

_allowlist: set[str] | None = None


def _load_allowlist() -> set[str]:
    global _allowlist
    if _allowlist is None:
        path = os.path.join(_CONFIG_DIR, "ner_allowlist.yaml")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _allowlist = set(data.get("allowed_entities", []))
    return _allowlist


# ---------------------------------------------------------------------------
# Token bracket detection
# ---------------------------------------------------------------------------

# Pre-compiled regex to find all [...] token spans in the replaced text.
_TOKEN_PATTERN = re.compile(r"\[[A-Z][A-Z0-9_]*\]")


def _build_token_spans(text: str) -> list[tuple[int, int]]:
    """Return a list of (start, end) character spans for all [...] tokens."""
    return [(m.start(), m.end()) for m in _TOKEN_PATTERN.finditer(text)]


def _is_allowlisted(entity_text: str, allowlist: set[str]) -> bool:
    """Check if an entity matches the allowlist.

    Handles spaCy quirks like including articles ("the United States") by
    checking whether the entity text contains any allowlisted term.
    Also does case-insensitive matching.
    """
    if entity_text in allowlist:
        return True
    # Check if entity contains an allowlisted term (handles "the United States")
    for allowed in allowlist:
        if allowed in entity_text:
            return True
    # Case-insensitive check
    et_lower = entity_text.lower()
    for allowed in allowlist:
        if allowed.lower() in et_lower:
            return True
    return False


def _is_inside_token(start: int, end: int, token_spans: list[tuple[int, int]]) -> bool:
    """Check if a character span falls within (or overlaps) any token bracket."""
    for ts, te in token_spans:
        if start >= ts and end <= te:
            return True
        # Partial overlap — entity straddles a bracket boundary
        if start < te and end > ts:
            return True
    return False


def _is_adjacent_to_token(text: str, start: int, end: int, token_spans: list[tuple[int, int]]) -> bool:
    """Check if a PERSON entity is directly adjacent to a [...] token.

    This catches orphaned first names like "SOPHIA [PARTY_D]" or "AMANDA [PARTY_J]"
    where spaCy tagged the first name as PERSON but it's part of a replaced full name.
    """
    # Check if a token bracket immediately follows (within a few chars)
    after = text[end:end + 5].lstrip()
    if after.startswith("["):
        return True
    # Check if a token bracket immediately precedes
    before = text[max(0, start - 5):start].rstrip()
    if before.endswith("]"):
        return True
    return False


# ---------------------------------------------------------------------------
# Pattern-based filtering for common non-PII entities
# ---------------------------------------------------------------------------

# ZIP codes spaCy tags as DATE
_ZIP_PATTERN = re.compile(r"^\d{5}(?:-\d{4})?$")

# Address-like ORG entities (number + street name)
_ADDRESS_PATTERN = re.compile(r"^\d+\s+\w+\s+(?:Lane|Drive|Way|Road|Street|Avenue|Boulevard|"
                              r"Court|Place|Circle|Terrace|Ridge|Trail|Parkway|Highway|Path|Loop|"
                              r"Run|Crossing|Pike|Square|Alley|Row)",
                              re.IGNORECASE)

# Street suffix at end of entity
_STREET_SUFFIX_PATTERN = re.compile(r"\b(?:Lane|Drive|Way|Road|Street|Avenue|Boulevard|"
                                     r"Court|Place|Circle|Terrace|Ridge|Trail|Parkway|Highway|"
                                     r"Path|Loop|Run|Crossing|Pike|Square|Alley|Row)\s*$",
                                     re.IGNORECASE)

# Duration-like text durations
_TEXT_DURATION_KEYWORDS = [
    "years", "months", "weeks", "days", "hours",
    "the age of", "age of majority",
]

# All-caps single word adjacent to token bracket (likely partial name)
_ALL_CAPS_PATTERN = re.compile(r"^[A-Z]{2,}$")


def _is_benign_entity(text: str, ent_text: str, ent_label: str,
                      start_char: int, end_char: int,
                      token_spans: list[tuple[int, int]]) -> bool:
    """Check if a detected entity is a known-benign pattern that shouldn't be flagged."""
    stripped = ent_text.strip()
    lower = stripped.lower()

    # Skip entities that contain newlines (multi-line noise from spaCy)
    if "\n" in ent_text:
        return True

    # DATE entities
    if ent_label == "DATE":
        # ZIP codes
        if _ZIP_PATTERN.match(stripped):
            return True
        # Text durations and ages
        if any(kw in lower for kw in _TEXT_DURATION_KEYWORDS):
            return True
        # Standalone 4-digit numbers that look like years but are in address context
        if re.match(r"^\d{4}$", stripped):
            # Check if it's preceded by address-like text
            before = text[max(0, start_char - 30):start_char].lower()
            if any(kw in before for kw in ["address:", "prior address:", "located at"]):
                return True

    # ORG entities
    if ent_label == "ORG":
        # Address components
        if _ADDRESS_PATTERN.match(stripped):
            return True
        if _STREET_SUFFIX_PATTERN.search(stripped):
            return True
        # Header/metadata noise
        if any(kw in lower for kw in [
            "privacy engineering", "entity replacement", "synthetic test",
        ]):
            return True
        # State abbreviation + ZIP patterns ("CA 95822", "TX 78745")
        if re.match(r"^[A-Z]{2}\s+\d{5}", stripped):
            return True

    # GPE entities — county names, state abbreviations, small jurisdictions
    if ent_label == "GPE":
        # 2-letter state abbreviations (TX, CA, OH, etc.)
        if re.match(r"^[A-Z]{2}$", stripped):
            return True
        # Check if preceded by "County of" or "County"
        before = text[max(0, start_char - 15):start_char].lower()
        if "county" in before:
            return True
        # Check if preceded by a replaced token + "of" (e.g., "[LOCATION_CA_RURAL] of Hennepin")
        before_stripped = text[max(0, start_char - 30):start_char].rstrip()
        if re.search(r"\]\s+of$", before_stripped):
            return True

    # PERSON / ORG entities — orphaned name parts adjacent to tokens
    # Handles "SOPHIA [PARTY_D]", "[PARTY_C] HUTCHINSON", "AMANDA [PARTY_J]"
    if ent_label in ("PERSON", "ORG"):
        if _ALL_CAPS_PATTERN.match(stripped) and len(stripped) <= 15:
            if _is_adjacent_to_token(text, start_char, end_char, token_spans):
                return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class FlaggedEntity:
    text: str
    label: str
    start_char: int
    end_char: int


@dataclass
class ValidationResult:
    passed: bool
    entities_detected: int
    entities_allowed: int
    entities_flagged: int
    flagged: list[FlaggedEntity] = field(default_factory=list)


class Validator:
    """Validate a de-identified document using second-pass NER."""

    # Entity labels to check
    LABELS_OF_INTEREST = {"PERSON", "ORG", "GPE", "DATE"}

    def __init__(self, allowlist: set[str] | None = None) -> None:
        self._custom_allowlist = allowlist

    @property
    def allowlist(self) -> set[str]:
        base = _load_allowlist()
        if self._custom_allowlist:
            return base | self._custom_allowlist
        return base

    def validate(self, replaced_text: str) -> ValidationResult:
        """Run NER on the replaced text and flag un-replaced entities."""
        nlp = _get_nlp()
        doc = nlp(replaced_text)
        token_spans = _build_token_spans(replaced_text)
        allowlist = self.allowlist

        allowed_count = 0
        flagged: list[FlaggedEntity] = []

        for ent in doc.ents:
            if ent.label_ not in self.LABELS_OF_INTEREST:
                continue

            # Skip entities inside [...] brackets
            if _is_inside_token(ent.start_char, ent.end_char, token_spans):
                continue

            # Check allowlist — exact match or entity contains an allowlisted term
            if _is_allowlisted(ent.text, allowlist):
                allowed_count += 1
                continue

            # Check pattern-based benign entity filtering
            if _is_benign_entity(replaced_text, ent.text, ent.label_,
                                 ent.start_char, ent.end_char, token_spans):
                allowed_count += 1
                continue

            flagged.append(FlaggedEntity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
            ))

        total_detected = allowed_count + len(flagged)
        return ValidationResult(
            passed=len(flagged) == 0,
            entities_detected=total_detected,
            entities_allowed=allowed_count,
            entities_flagged=len(flagged),
            flagged=flagged,
        )
