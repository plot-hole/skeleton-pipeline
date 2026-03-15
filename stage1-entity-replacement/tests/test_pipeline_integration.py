"""Integration tests — run the full pipeline on fixture documents."""

from __future__ import annotations

import json
import os
from datetime import date

import pytest

from src.pipeline import Pipeline
from src.schemas import ClassificationMetadata, PipelineOutput, PrivilegeHaltError

spacy = pytest.importorskip("spacy")

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load_fixture(name: str) -> tuple[str, ClassificationMetadata]:
    """Load a document + metadata fixture pair."""
    with open(os.path.join(FIXTURES, f"{name}.txt"), encoding="utf-8") as f:
        doc_text = f.read()
    with open(os.path.join(FIXTURES, f"{name}_metadata.json"), encoding="utf-8") as f:
        raw = json.load(f)
    metadata = ClassificationMetadata(**raw)
    return doc_text, metadata


# ---------------------------------------------------------------------------
# Lease integration test
# ---------------------------------------------------------------------------

class TestLeaseIntegration:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.doc_text, self.metadata = _load_fixture("sample_lease")
        pipeline = Pipeline(reference_date=date(2026, 3, 7))
        self.output = pipeline.run(self.doc_text, self.metadata)

    def test_no_flagged_entities(self):
        """Validation should pass (or flag only allowlisted entities)."""
        report = self.output.validation_report
        # We allow the test to pass if flagged count is 0
        # If spaCy picks up an edge case, we can add it to the allowlist
        assert report.validation_passed is True, (
            f"Flagged: {report.ner_entities_flagged} entities. "
            f"Check allowlist for false positives."
        )

    def test_all_critical_elements_replaced(self):
        """No CRITICAL-sensitivity original text should survive."""
        for el in self.metadata.elements:
            if el.sensitivity == "CRITICAL" and el.category != "BOILERPLATE":
                assert el.original_text not in self.output.deidentified_text, (
                    f"CRITICAL element '{el.original_text}' still in output"
                )

    def test_all_high_elements_replaced(self):
        """No HIGH-sensitivity original text should survive."""
        for el in self.metadata.elements:
            if el.sensitivity == "HIGH" and el.category != "BOILERPLATE":
                assert el.original_text not in self.output.deidentified_text, (
                    f"HIGH element '{el.original_text}' still in output"
                )

    def test_boilerplate_unchanged(self):
        """BOILERPLATE text should appear unchanged in the output."""
        for el in self.metadata.elements:
            if el.category == "BOILERPLATE":
                assert el.original_text in self.output.deidentified_text

    def test_token_mapping_no_original_values(self):
        """Token mapping table must not contain any original PII values."""
        pii_values = {
            el.original_text
            for el in self.metadata.elements
            if el.category != "BOILERPLATE"
        }
        mapping_json = self.output.token_mapping.model_dump_json()
        for pii in pii_values:
            assert pii not in mapping_json, (
                f"PII value '{pii}' found in token mapping"
            )

    def test_relationship_graph_non_empty(self):
        """Multi-party document should have at least one relationship."""
        assert len(self.output.relationship_graph.relationships) > 0

    def test_party_tokens_present(self):
        """Should have [PARTY_A] and [PARTY_B] tokens."""
        text = self.output.deidentified_text
        assert "[PARTY_A]" in text
        assert "[PARTY_B]" in text

    def test_ssn_redacted(self):
        assert "[SSN_REDACTED]" in self.output.deidentified_text
        assert "123-45-6789" not in self.output.deidentified_text

    def test_rent_bucketed(self):
        assert "[RENT_" in self.output.deidentified_text
        assert "$4,200/month" not in self.output.deidentified_text

    def test_minor_token_present(self):
        assert "[MINOR_CHILD_1]" in self.output.deidentified_text

    def test_health_redacted(self):
        assert "[PROTECTED_STATUS_REDACTED]" in self.output.deidentified_text
        assert "documented disability" not in self.output.deidentified_text


# ---------------------------------------------------------------------------
# NDA integration test
# ---------------------------------------------------------------------------

class TestNdaIntegration:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.doc_text, self.metadata = _load_fixture("sample_nda")
        pipeline = Pipeline(reference_date=date(2026, 3, 7))
        self.output = pipeline.run(self.doc_text, self.metadata)

    def test_no_flagged_entities(self):
        assert self.output.validation_report.validation_passed is True

    def test_names_replaced(self):
        assert "Lisa Park" not in self.output.deidentified_text
        assert "Brian Torres" not in self.output.deidentified_text
        assert "[PARTY_A]" in self.output.deidentified_text
        assert "[PARTY_B]" in self.output.deidentified_text

    def test_dates_generalized(self):
        assert "January 10, 2026" not in self.output.deidentified_text
        assert "[DATE_" in self.output.deidentified_text

    def test_boilerplate_unchanged(self):
        for el in self.metadata.elements:
            if el.category == "BOILERPLATE":
                assert el.original_text in self.output.deidentified_text

    def test_no_minor_or_health_data(self):
        """NDA should have no minor or health tokens."""
        text = self.output.deidentified_text
        assert "[MINOR_CHILD_" not in text
        assert "[PROTECTED_STATUS_REDACTED]" not in text


# ---------------------------------------------------------------------------
# Custody integration test
# ---------------------------------------------------------------------------

class TestCustodyIntegration:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.doc_text, self.metadata = _load_fixture("sample_custody")
        pipeline = Pipeline(reference_date=date(2026, 3, 7))
        self.output = pipeline.run(self.doc_text, self.metadata)

    def test_no_flagged_entities(self):
        assert self.output.validation_report.validation_passed is True

    def test_all_names_replaced(self):
        text = self.output.deidentified_text
        assert "Maria Gonzalez" not in text
        assert "Robert Nguyen" not in text

    def test_two_minor_tokens(self):
        text = self.output.deidentified_text
        assert "[MINOR_CHILD_1]" in text
        assert "[MINOR_CHILD_2]" in text

    def test_minor_names_removed(self):
        text = self.output.deidentified_text
        # The original names "Sofia" and "Lucas" should be gone
        # (replaced by MINOR_CHILD tokens)
        assert "Sofia" not in text
        assert "Lucas" not in text

    def test_two_addresses_generalized(self):
        text = self.output.deidentified_text
        assert "500 Pine Ave" not in text
        assert "78 Maple Dr" not in text
        assert "[LOCATION_CA_" in text

    def test_employers_generalized(self):
        text = self.output.deidentified_text
        assert "Global Health Inc" not in text
        assert "Pacific Engineering" not in text

    def test_income_bucketed(self):
        text = self.output.deidentified_text
        assert "$82,000" not in text
        assert "$97,500" not in text

    def test_relationship_graph_has_parent_child(self):
        rels = self.output.relationship_graph.relationships
        pc = [r for r in rels if r.type == "PARENT_CHILD"]
        assert len(pc) >= 1

    def test_relationship_graph_has_employment(self):
        rels = self.output.relationship_graph.relationships
        emp = [r for r in rels if r.type == "EMPLOYMENT"]
        assert len(emp) >= 1


# ---------------------------------------------------------------------------
# Privilege halt test
# ---------------------------------------------------------------------------

class TestPrivilegeHalt:
    def test_privilege_flag_halts_pipeline(self):
        """Document with privilege_flag=true should halt."""
        doc_text = "Some text."
        metadata = ClassificationMetadata(
            document_id="priv-001",
            tenant_id="tenant-001",
            classification_timestamp="2026-01-01T00:00:00Z",
            classifier_version="1.0",
            elements=[],
            summary={
                "total_elements": 0,
                "critical_count": 0,
                "high_count": 0,
                "low_count": 0,
                "privilege_flag": True,
                "minor_data_present": False,
            },
        )
        pipeline = Pipeline()
        with pytest.raises(PrivilegeHaltError):
            pipeline.run(doc_text, metadata)


# ---------------------------------------------------------------------------
# All-boilerplate document
# ---------------------------------------------------------------------------

class TestAllBoilerplate:
    def test_empty_token_mapping(self):
        """A doc with only boilerplate should pass through with empty mappings."""
        doc_text = "This agreement shall be governed by the laws of California."
        metadata = ClassificationMetadata(
            document_id="bp-001",
            tenant_id="tenant-001",
            classification_timestamp="2026-01-01T00:00:00Z",
            classifier_version="1.0",
            elements=[
                {
                    "element_id": "b1",
                    "category": "BOILERPLATE",
                    "subcategory": "LEGAL_CLAUSE",
                    "sensitivity": "LOW",
                    "byte_offset_start": 0,
                    "byte_offset_end": len(doc_text.encode("utf-8")),
                    "original_text": doc_text,
                    "context": {
                        "data_subject_role": "PARTY_PRIMARY",
                        "jurisdiction": "US_CA",
                    },
                }
            ],
            summary={
                "total_elements": 1,
                "critical_count": 0,
                "high_count": 0,
                "low_count": 1,
                "privilege_flag": False,
                "minor_data_present": False,
            },
        )
        pipeline = Pipeline()
        output = pipeline.run(doc_text, metadata)

        assert output.deidentified_text == doc_text
        assert len(output.token_mapping.tokens) == 0
        assert output.validation_report.elements_passed_through == 1
