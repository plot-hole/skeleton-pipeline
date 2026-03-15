"""Replacement engine — applies the replacement policy to every classified element."""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import yaml

from src.generalization import (
    bucket_financial,
    generalize_address,
    generalize_date,
    generalize_dob,
    generalize_employer,
)
from src.schemas import (
    ClassificationMetadata,
    ClassifiedElement,
    EntityGroup,
    TokenMapping,
)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "config")


# ---------------------------------------------------------------------------
# ByteOffsetDocument — works in UTF-8 byte space
# ---------------------------------------------------------------------------

class ByteOffsetDocument:
    """Wraps a document string and performs replacements using UTF-8 byte offsets.

    Cyera provides ``byte_offset_start`` / ``byte_offset_end`` as positions
    within the UTF-8 encoded representation of the document.  This class
    keeps the document as a ``bytearray`` so that all slicing and replacement
    operations use the same coordinate system.
    """

    def __init__(self, text: str) -> None:
        self._bytes = bytearray(text.encode("utf-8"))

    def extract(self, start: int, end: int) -> str:
        """Return the decoded substring at the given byte offsets."""
        return bytes(self._bytes[start:end]).decode("utf-8")

    def replace_at(self, start: int, end: int, replacement: str) -> None:
        """Replace the byte range ``[start, end)`` with *replacement*."""
        replacement_bytes = replacement.encode("utf-8")
        self._bytes[start:end] = replacement_bytes

    def to_str(self) -> str:
        """Return the full document as a string."""
        return bytes(self._bytes).decode("utf-8")


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

_policies: list[dict[str, Any]] | None = None


def _load_policies() -> list[dict[str, Any]]:
    global _policies
    if _policies is None:
        path = os.path.join(_CONFIG_DIR, "replacement_policies.yaml")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _policies = data["policies"]
    return _policies


def _find_policy(category: str, subcategory: str) -> dict[str, Any] | None:
    """Look up the replacement policy for a (category, subcategory) pair.

    First tries an exact (category, subcategory) match, then falls back to
    a category-only match (where no subcategory is specified in the policy).
    """
    policies = _load_policies()
    # Exact match first
    for p in policies:
        if p["category"] == category and p.get("subcategory") == subcategory:
            return p
    # Category-level fallback
    for p in policies:
        if p["category"] == category and "subcategory" not in p:
            return p
    return None


# ---------------------------------------------------------------------------
# Replacement engine
# ---------------------------------------------------------------------------

class ReplacementEngine:
    """Apply the replacement policy to every element in the document.

    Parameters
    ----------
    document_text : str
        The raw document text.
    metadata : ClassificationMetadata
        Cyera classification metadata.
    entity_groups : list[EntityGroup]
        Groups produced by the co-reference engine.
    reference_date : date | None
        Reference date for DOB age-range calculation (defaults to today).
    """

    def __init__(
        self,
        document_text: str,
        metadata: ClassificationMetadata,
        entity_groups: list[EntityGroup],
        reference_date: date | None = None,
    ) -> None:
        self.doc = ByteOffsetDocument(document_text)
        self.metadata = metadata
        self.entity_groups = entity_groups
        self.reference_date = reference_date or date.today()

        # Build element_id -> token lookup
        self._element_token: dict[str, str] = {}
        for group in entity_groups:
            for eid in group.element_ids:
                self._element_token[eid] = group.token

        # Track token mapping entries
        self._token_entries: dict[str, TokenMapping] = {}
        # Track replaced byte ranges to handle overlapping elements
        self._replaced_ranges: list[tuple[int, int]] = []

    def replace_all(self) -> tuple[str, list[TokenMapping]]:
        """Run all replacements and return (replaced_text, token_mappings)."""
        elements_replaced = 0
        elements_passed = 0

        # Sort by byte_offset_start descending (and end descending for ties)
        sorted_elements = sorted(
            self.metadata.elements,
            key=lambda e: (e.byte_offset_start, e.byte_offset_end),
            reverse=True,
        )

        for element in sorted_elements:
            if self._is_already_replaced(element.byte_offset_start, element.byte_offset_end):
                continue

            replacement = self._get_replacement(element)
            if replacement is None:
                # PASS_THROUGH (boilerplate)
                elements_passed += 1
                continue

            self.doc.replace_at(
                element.byte_offset_start,
                element.byte_offset_end,
                replacement,
            )
            self._replaced_ranges.append(
                (element.byte_offset_start, element.byte_offset_end)
            )
            self._record_token(element, replacement)
            elements_replaced += 1

        return self.doc.to_str(), list(self._token_entries.values())

    # -- Replacement dispatch -----------------------------------------------

    def _get_replacement(self, element: ClassifiedElement) -> str | None:
        """Return the replacement string for an element, or None for pass-through."""
        policy = _find_policy(element.category, element.subcategory)
        if policy is None:
            return None

        strategy = policy["strategy"]

        if strategy == "PASS_THROUGH":
            return None

        if strategy == "HALT":
            from src.schemas import PrivilegeHaltError
            raise PrivilegeHaltError(
                f"PRIVILEGE_INDICATOR encountered: element {element.element_id}"
            )

        if strategy == "STATIC_REPLACE":
            return policy["token"]

        if strategy == "TOKEN_REPLACE":
            token = self._element_token.get(element.element_id)
            if token:
                return token
            # Fallback if element wasn't grouped by coreference
            return f"[{element.category}_REDACTED]"

        if strategy == "GENERALIZE_ADDRESS":
            return generalize_address(
                element.original_text, element.context.jurisdiction
            )

        if strategy == "GENERALIZE_DATE":
            return generalize_date(element.original_text)

        if strategy == "GENERALIZE_DOB":
            return generalize_dob(element.original_text, self.reference_date)

        if strategy == "GENERALIZE_EMPLOYER":
            industry = element.context.industry
            return generalize_employer(element.original_text, industry)

        if strategy == "BUCKET_FINANCIAL":
            bucket_category = policy.get("bucket_category", element.subcategory)
            return bucket_financial(element.original_text, bucket_category)

        # Unknown strategy — treat as static redaction
        return f"[{element.category}_REDACTED]"

    # -- Helpers ------------------------------------------------------------

    def _is_already_replaced(self, start: int, end: int) -> bool:
        """Check if a byte range is fully contained within an already-replaced range."""
        for rs, re_ in self._replaced_ranges:
            if start >= rs and end <= re_:
                return True
        return False

    def _record_token(self, element: ClassifiedElement, token: str) -> None:
        """Record a token mapping entry (accumulates element_ids per token)."""
        if token not in self._token_entries:
            self._token_entries[token] = TokenMapping(
                token=token,
                category=element.category,
                element_ids=[element.element_id],
                occurrence_count=1,
                data_subject_role=element.context.data_subject_role,
                original_jurisdiction=element.context.jurisdiction or None,
            )
        else:
            entry = self._token_entries[token]
            entry.element_ids.append(element.element_id)
            entry.occurrence_count += 1
