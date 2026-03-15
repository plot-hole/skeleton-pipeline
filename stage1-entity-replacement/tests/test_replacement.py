"""Tests for the replacement engine."""

from datetime import date

import pytest

from src.replacement import ByteOffsetDocument, ReplacementEngine
from src.schemas import (
    ClassificationMetadata,
    ClassificationSummary,
    ClassifiedElement,
    ElementContext,
    EntityGroup,
    PrivilegeHaltError,
)


# ---------------------------------------------------------------------------
# ByteOffsetDocument
# ---------------------------------------------------------------------------

class TestByteOffsetDocument:
    def test_extract_ascii(self):
        doc = ByteOffsetDocument("Hello World")
        assert doc.extract(0, 5) == "Hello"
        assert doc.extract(6, 11) == "World"

    def test_replace_at_ascii(self):
        doc = ByteOffsetDocument("Hello World")
        doc.replace_at(6, 11, "[TOKEN]")
        assert doc.to_str() == "Hello [TOKEN]"

    def test_replace_backward_preserves_offsets(self):
        doc = ByteOffsetDocument("AAA BBB CCC")
        # Replace CCC first (offsets 8-11), then AAA (offsets 0-3)
        doc.replace_at(8, 11, "[C]")
        doc.replace_at(0, 3, "[A]")
        assert doc.to_str() == "[A] BBB [C]"

    def test_multibyte_chars(self):
        # Smart quotes are 3 bytes each in UTF-8
        text = "\u201cHello\u201d"  # "Hello"
        doc = ByteOffsetDocument(text)
        raw = text.encode("utf-8")
        # The opening smart quote is bytes 0-3, "Hello" is bytes 3-8,
        # closing smart quote is bytes 8-11
        assert doc.extract(3, 8) == "Hello"

    def test_replace_with_multibyte(self):
        text = "Name: Ren\u00e9"  # René — é is 2 bytes in UTF-8
        doc = ByteOffsetDocument(text)
        raw = text.encode("utf-8")
        # "René" starts at byte 6, é is 2 bytes, total = 4+1 = 5 bytes?
        # R(1) e(1) n(1) é(2) = 5 bytes
        start = raw.index(b"Ren")
        end = start + len("René".encode("utf-8"))
        doc.replace_at(start, end, "[PARTY_A]")
        assert doc.to_str() == "Name: [PARTY_A]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(
    elements: list[ClassifiedElement],
    doc_id: str = "doc-1",
    privilege_flag: bool = False,
    minor_data: bool = False,
) -> ClassificationMetadata:
    critical = sum(1 for e in elements if e.sensitivity == "CRITICAL")
    high = sum(1 for e in elements if e.sensitivity == "HIGH")
    low = sum(1 for e in elements if e.sensitivity == "LOW")
    return ClassificationMetadata(
        document_id=doc_id,
        tenant_id="tenant-1",
        classification_timestamp="2026-01-01T00:00:00Z",
        classifier_version="1.0",
        elements=elements,
        summary=ClassificationSummary(
            total_elements=len(elements),
            critical_count=critical,
            high_count=high,
            low_count=low,
            privilege_flag=privilege_flag,
            minor_data_present=minor_data,
        ),
    )


def _make_element(
    text: str,
    doc_text: str,
    element_id: str = "e1",
    category: str = "DIRECT_IDENTIFIER",
    subcategory: str = "FULL_NAME",
    sensitivity: str = "CRITICAL",
    role: str = "PARTY_PRIMARY",
    jurisdiction: str = "US_CA",
    industry: str | None = None,
) -> ClassifiedElement:
    """Create an element with byte offsets computed from doc_text."""
    raw = doc_text.encode("utf-8")
    start = raw.index(text.encode("utf-8"))
    end = start + len(text.encode("utf-8"))
    return ClassifiedElement(
        element_id=element_id,
        category=category,
        subcategory=subcategory,
        sensitivity=sensitivity,
        byte_offset_start=start,
        byte_offset_end=end,
        original_text=text,
        context=ElementContext(
            data_subject_role=role,
            jurisdiction=jurisdiction,
            industry=industry,
        ),
    )


# ---------------------------------------------------------------------------
# ReplacementEngine
# ---------------------------------------------------------------------------

class TestReplacementEngine:
    def test_single_name_replacement(self):
        doc_text = "The tenant is Sarah Mitchell."
        element = _make_element("Sarah Mitchell", doc_text, element_id="e1")
        metadata = _make_metadata([element])
        groups = [
            EntityGroup(
                token="[PARTY_A]",
                category="DIRECT_IDENTIFIER",
                element_ids=["e1"],
                data_subject_role="PARTY_PRIMARY",
                occurrence_count=1,
            )
        ]

        engine = ReplacementEngine(doc_text, metadata, groups)
        result, mappings = engine.replace_all()

        assert result == "The tenant is [PARTY_A]."
        assert any(m.token == "[PARTY_A]" for m in mappings)

    def test_two_names_backward_replacement(self):
        doc_text = "Sarah Mitchell and David Chen signed the lease."
        e1 = _make_element("Sarah Mitchell", doc_text, element_id="e1")
        e2 = _make_element("David Chen", doc_text, element_id="e2", role="PARTY_SECONDARY")
        metadata = _make_metadata([e1, e2])
        groups = [
            EntityGroup(token="[PARTY_A]", category="DIRECT_IDENTIFIER",
                        element_ids=["e1"], data_subject_role="PARTY_PRIMARY", occurrence_count=1),
            EntityGroup(token="[PARTY_B]", category="DIRECT_IDENTIFIER",
                        element_ids=["e2"], data_subject_role="PARTY_SECONDARY", occurrence_count=1),
        ]

        engine = ReplacementEngine(doc_text, metadata, groups)
        result, _ = engine.replace_all()

        assert "[PARTY_A]" in result
        assert "[PARTY_B]" in result
        assert "Sarah Mitchell" not in result
        assert "David Chen" not in result

    def test_ssn_stripped(self):
        doc_text = "SSN: 123-45-6789 provided."
        element = _make_element(
            "123-45-6789", doc_text, element_id="e1",
            subcategory="SSN", role="PARTY_PRIMARY",
        )
        metadata = _make_metadata([element])

        engine = ReplacementEngine(doc_text, metadata, entity_groups=[])
        result, mappings = engine.replace_all()

        assert result == "SSN: [SSN_REDACTED] provided."

    def test_address_generalized(self):
        doc_text = "Address: 142 Oak St, Sacramento, CA 95814."
        element = _make_element(
            "142 Oak St, Sacramento, CA 95814", doc_text, element_id="e1",
            category="QUASI_IDENTIFIER", subcategory="ADDRESS",
            sensitivity="HIGH", role="PARTY_PRIMARY",
        )
        metadata = _make_metadata([element])

        engine = ReplacementEngine(doc_text, metadata, entity_groups=[])
        result, _ = engine.replace_all()

        assert result == "Address: [LOCATION_CA_METRO]."

    def test_date_generalized(self):
        doc_text = "Effective date: March 15, 2026."
        element = _make_element(
            "March 15, 2026", doc_text, element_id="e1",
            category="QUASI_IDENTIFIER", subcategory="DATE",
            sensitivity="HIGH", role="PARTY_PRIMARY",
        )
        metadata = _make_metadata([element])

        engine = ReplacementEngine(doc_text, metadata, entity_groups=[])
        result, _ = engine.replace_all()

        assert result == "Effective date: [DATE_2026_Q1]."

    def test_dob_generalized(self):
        doc_text = "Date of birth: 03/15/1988."
        element = _make_element(
            "03/15/1988", doc_text, element_id="e1",
            category="QUASI_IDENTIFIER", subcategory="DATE_OF_BIRTH",
            sensitivity="HIGH", role="PARTY_PRIMARY",
        )
        metadata = _make_metadata([element])

        engine = ReplacementEngine(
            doc_text, metadata, entity_groups=[],
            reference_date=date(2026, 1, 1),
        )
        result, _ = engine.replace_all()

        assert result == "Date of birth: [AGE_RANGE_35_40]."

    def test_financial_bucketed(self):
        doc_text = "Monthly rent: $4,200/month."
        element = _make_element(
            "$4,200/month", doc_text, element_id="e1",
            category="FINANCIAL_DATA", subcategory="RENT",
            sensitivity="HIGH", role="PARTY_PRIMARY",
        )
        metadata = _make_metadata([element])

        engine = ReplacementEngine(doc_text, metadata, entity_groups=[])
        result, _ = engine.replace_all()

        assert result == "Monthly rent: [RENT_4000_5000]."

    def test_health_stripped(self):
        doc_text = "Tenant has a documented disability."
        element = _make_element(
            "Tenant has a documented disability", doc_text, element_id="e1",
            category="HEALTH_PROTECTED", subcategory="MEDICAL_CONDITION",
            sensitivity="CRITICAL", role="PARTY_PRIMARY",
        )
        metadata = _make_metadata([element])

        engine = ReplacementEngine(doc_text, metadata, entity_groups=[])
        result, _ = engine.replace_all()

        assert result == "[PROTECTED_STATUS_REDACTED]."

    def test_boilerplate_unchanged(self):
        doc_text = "This agreement shall be governed by the laws of California."
        element = _make_element(
            "This agreement shall be governed by the laws of California", doc_text,
            element_id="e1", category="BOILERPLATE", subcategory="LEGAL_CLAUSE",
            sensitivity="LOW", role="PARTY_PRIMARY",
        )
        metadata = _make_metadata([element])

        engine = ReplacementEngine(doc_text, metadata, entity_groups=[])
        result, _ = engine.replace_all()

        assert result == doc_text  # Unchanged

    def test_privilege_indicator_halts(self):
        doc_text = "Attorney-client privileged communication."
        element = _make_element(
            "Attorney-client privileged communication", doc_text,
            element_id="e1", category="PRIVILEGE_INDICATOR",
            subcategory="ATTORNEY_CLIENT", sensitivity="CRITICAL",
            role="PARTY_PRIMARY",
        )
        metadata = _make_metadata([element], privilege_flag=True)

        engine = ReplacementEngine(doc_text, metadata, entity_groups=[])
        with pytest.raises(PrivilegeHaltError):
            engine.replace_all()

    def test_minor_token_replace(self):
        doc_text = "The child Emma attends school."
        element = _make_element(
            "Emma", doc_text, element_id="m1",
            category="MINOR_INFORMATION", subcategory="CHILD_NAME",
            sensitivity="CRITICAL", role="MINOR_DEPENDENT",
        )
        metadata = _make_metadata([element], minor_data=True)
        groups = [
            EntityGroup(
                token="[MINOR_CHILD_1]",
                category="MINOR_INFORMATION",
                element_ids=["m1"],
                data_subject_role="MINOR_DEPENDENT",
                occurrence_count=1,
            )
        ]

        engine = ReplacementEngine(doc_text, metadata, groups)
        result, _ = engine.replace_all()

        assert result == "The child [MINOR_CHILD_1] attends school."

    def test_token_mapping_has_no_original_values(self):
        """Token mapping must NOT contain original_text (PII)."""
        doc_text = "Tenant: Sarah Mitchell."
        element = _make_element("Sarah Mitchell", doc_text, element_id="e1")
        metadata = _make_metadata([element])
        groups = [
            EntityGroup(token="[PARTY_A]", category="DIRECT_IDENTIFIER",
                        element_ids=["e1"], data_subject_role="PARTY_PRIMARY", occurrence_count=1),
        ]

        engine = ReplacementEngine(doc_text, metadata, groups)
        _, mappings = engine.replace_all()

        for m in mappings:
            # Check that no mapping field contains the original PII value
            assert "Sarah Mitchell" not in m.token
            assert "Sarah Mitchell" not in str(m.element_ids)
