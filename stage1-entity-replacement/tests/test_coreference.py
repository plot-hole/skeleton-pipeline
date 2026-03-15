"""Tests for co-reference resolution engine."""

import pytest

from src.coreference import CoReferenceEngine, _parse_name, _party_letter, _strip_honorifics
from src.schemas import ClassifiedElement, ElementContext


def _make_name_element(
    original_text: str,
    element_id: str = "e1",
    role: str = "PARTY_PRIMARY",
    offset_start: int = 0,
    offset_end: int | None = None,
    category: str = "DIRECT_IDENTIFIER",
    subcategory: str = "FULL_NAME",
) -> ClassifiedElement:
    """Helper to create a name ClassifiedElement."""
    return ClassifiedElement(
        element_id=element_id,
        category=category,
        subcategory=subcategory,
        sensitivity="CRITICAL",
        byte_offset_start=offset_start,
        byte_offset_end=offset_end or (offset_start + len(original_text)),
        original_text=original_text,
        context=ElementContext(
            data_subject_role=role,
            jurisdiction="US_CA",
            applicable_regulations=["CCPA"],
        ),
    )


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestStripHonorifics:
    def test_mr(self):
        assert _strip_honorifics("Mr. Smith") == "Smith"

    def test_ms(self):
        assert _strip_honorifics("Ms. Mitchell") == "Mitchell"

    def test_dr(self):
        assert _strip_honorifics("Dr. Jane Doe") == "Jane Doe"

    def test_no_honorific(self):
        assert _strip_honorifics("Sarah Mitchell") == "Sarah Mitchell"

    def test_suffix_jr(self):
        assert _strip_honorifics("James Smith Jr.") == "James Smith"


class TestParseName:
    def test_full_name(self):
        p = _parse_name("Sarah Mitchell")
        assert p.first == "Sarah"
        assert p.last == "Mitchell"

    def test_single_name(self):
        p = _parse_name("Sarah")
        assert p.first == "Sarah"
        assert p.last == ""

    def test_with_honorific(self):
        p = _parse_name("Ms. Mitchell")
        assert p.first == "Mitchell"
        assert p.last == ""

    def test_three_part_name(self):
        p = _parse_name("Mary Jane Watson")
        assert p.first == "Mary"
        assert p.last == "Watson"


class TestPartyLetter:
    def test_first(self):
        assert _party_letter(0) == "A"

    def test_second(self):
        assert _party_letter(1) == "B"

    def test_26th(self):
        assert _party_letter(25) == "Z"

    def test_27th(self):
        assert _party_letter(26) == "AA"


# ---------------------------------------------------------------------------
# CoReferenceEngine — name grouping
# ---------------------------------------------------------------------------

class TestNameResolution:
    def test_three_mentions_same_person(self):
        """'Sarah Mitchell', 'Ms. Mitchell', 'Sarah' -> one group [PARTY_A]."""
        elements = [
            _make_name_element("Sarah Mitchell", element_id="e1", offset_start=0),
            _make_name_element("Ms. Mitchell", element_id="e2", offset_start=100),
            _make_name_element("Sarah", element_id="e3", offset_start=200),
        ]
        engine = CoReferenceEngine(elements)
        groups = engine.resolve()

        assert len(groups) == 1
        assert groups[0].token == "[PARTY_A]"
        assert set(groups[0].element_ids) == {"e1", "e2", "e3"}
        assert groups[0].occurrence_count == 3

    def test_two_different_parties(self):
        """Two different people with different names -> two groups."""
        elements = [
            _make_name_element("Sarah Mitchell", element_id="e1", role="PARTY_PRIMARY", offset_start=0),
            _make_name_element("David Chen", element_id="e2", role="PARTY_SECONDARY", offset_start=100),
        ]
        engine = CoReferenceEngine(elements)
        groups = engine.resolve()

        assert len(groups) == 2
        tokens = {g.token for g in groups}
        assert tokens == {"[PARTY_A]", "[PARTY_B]"}

    def test_same_surname_different_roles_stay_separate(self):
        """Same surname, different roles -> separate groups."""
        elements = [
            _make_name_element("Sarah Mitchell", element_id="e1", role="PARTY_PRIMARY", offset_start=0),
            _make_name_element("James Mitchell", element_id="e2", role="PARTY_SECONDARY", offset_start=100),
        ]
        engine = CoReferenceEngine(elements)
        groups = engine.resolve()

        assert len(groups) == 2

    def test_same_surname_same_role_merge(self):
        """Same surname, same role -> merged."""
        elements = [
            _make_name_element("Sarah Mitchell", element_id="e1", role="PARTY_PRIMARY", offset_start=0),
            _make_name_element("Ms. Mitchell", element_id="e2", role="PARTY_PRIMARY", offset_start=100),
        ]
        engine = CoReferenceEngine(elements)
        groups = engine.resolve()

        assert len(groups) == 1
        assert set(groups[0].element_ids) == {"e1", "e2"}

    def test_ordering_by_first_appearance(self):
        """Groups are ordered by the earliest byte_offset_start of their members."""
        elements = [
            _make_name_element("David Chen", element_id="e1", role="PARTY_SECONDARY", offset_start=50),
            _make_name_element("Sarah Mitchell", element_id="e2", role="PARTY_PRIMARY", offset_start=0),
        ]
        engine = CoReferenceEngine(elements)
        groups = engine.resolve()

        # Sarah appears first (offset 0) -> gets PARTY_A
        assert groups[0].token == "[PARTY_A]"
        assert "e2" in groups[0].element_ids

    def test_no_name_elements_empty(self):
        """No FULL_NAME elements -> empty groups."""
        elements = [
            _make_name_element("123-45-6789", element_id="e1", subcategory="SSN"),
        ]
        engine = CoReferenceEngine(elements)
        groups = engine.resolve()
        assert len(groups) == 0


# ---------------------------------------------------------------------------
# CoReferenceEngine — minor grouping
# ---------------------------------------------------------------------------

class TestMinorResolution:
    def test_single_minor(self):
        elements = [
            _make_name_element(
                "Emma", element_id="m1", role="MINOR_DEPENDENT",
                category="MINOR_INFORMATION", subcategory="CHILD_NAME",
                offset_start=0,
            ),
        ]
        engine = CoReferenceEngine(elements)
        groups = engine.resolve()

        minor_groups = [g for g in groups if g.category == "MINOR_INFORMATION"]
        assert len(minor_groups) == 1
        assert minor_groups[0].token == "[MINOR_CHILD_1]"

    def test_two_different_minors(self):
        elements = [
            _make_name_element(
                "Emma", element_id="m1", role="MINOR_DEPENDENT",
                category="MINOR_INFORMATION", subcategory="CHILD_NAME",
                offset_start=0,
            ),
            _make_name_element(
                "Liam", element_id="m2", role="MINOR_DEPENDENT",
                category="MINOR_INFORMATION", subcategory="CHILD_NAME",
                offset_start=100,
            ),
        ]
        engine = CoReferenceEngine(elements)
        groups = engine.resolve()

        minor_groups = [g for g in groups if g.category == "MINOR_INFORMATION"]
        assert len(minor_groups) == 2
        tokens = {g.token for g in minor_groups}
        assert tokens == {"[MINOR_CHILD_1]", "[MINOR_CHILD_2]"}

    def test_same_minor_mentioned_twice(self):
        elements = [
            _make_name_element(
                "Emma", element_id="m1", role="MINOR_DEPENDENT",
                category="MINOR_INFORMATION", subcategory="CHILD_NAME",
                offset_start=0,
            ),
            _make_name_element(
                "Emma", element_id="m2", role="MINOR_DEPENDENT",
                category="MINOR_INFORMATION", subcategory="CHILD_NAME",
                offset_start=200,
            ),
        ]
        engine = CoReferenceEngine(elements)
        groups = engine.resolve()

        minor_groups = [g for g in groups if g.category == "MINOR_INFORMATION"]
        assert len(minor_groups) == 1
        assert minor_groups[0].occurrence_count == 2


# ---------------------------------------------------------------------------
# CoReferenceEngine — employer grouping
# ---------------------------------------------------------------------------

class TestEmployerResolution:
    def test_employer_with_industry(self):
        elements = [
            ClassifiedElement(
                element_id="emp1",
                category="QUASI_IDENTIFIER",
                subcategory="EMPLOYER_NAME",
                sensitivity="HIGH",
                byte_offset_start=0,
                byte_offset_end=9,
                original_text="Acme Corp",
                context=ElementContext(
                    data_subject_role="EMPLOYER",
                    jurisdiction="US_CA",
                    industry="TECH",
                ),
            ),
        ]
        engine = CoReferenceEngine(elements)
        groups = engine.resolve()

        emp_groups = [g for g in groups if g.category == "QUASI_IDENTIFIER"]
        assert len(emp_groups) == 1
        assert emp_groups[0].token == "[EMPLOYER_TECH]"

    def test_employer_without_industry(self):
        elements = [
            ClassifiedElement(
                element_id="emp1",
                category="QUASI_IDENTIFIER",
                subcategory="EMPLOYER_NAME",
                sensitivity="HIGH",
                byte_offset_start=0,
                byte_offset_end=9,
                original_text="Some Firm",
                context=ElementContext(
                    data_subject_role="EMPLOYER",
                    jurisdiction="US_CA",
                ),
            ),
        ]
        engine = CoReferenceEngine(elements)
        groups = engine.resolve()

        emp_groups = [g for g in groups if g.category == "QUASI_IDENTIFIER"]
        assert len(emp_groups) == 1
        assert emp_groups[0].token == "[EMPLOYER_1]"

    def test_same_employer_mentioned_twice(self):
        elements = [
            ClassifiedElement(
                element_id="emp1",
                category="QUASI_IDENTIFIER",
                subcategory="EMPLOYER_NAME",
                sensitivity="HIGH",
                byte_offset_start=0,
                byte_offset_end=9,
                original_text="Acme Corp",
                context=ElementContext(
                    data_subject_role="EMPLOYER",
                    jurisdiction="US_CA",
                    industry="TECH",
                ),
            ),
            ClassifiedElement(
                element_id="emp2",
                category="QUASI_IDENTIFIER",
                subcategory="EMPLOYER_NAME",
                sensitivity="HIGH",
                byte_offset_start=200,
                byte_offset_end=209,
                original_text="Acme Corp",
                context=ElementContext(
                    data_subject_role="EMPLOYER",
                    jurisdiction="US_CA",
                    industry="TECH",
                ),
            ),
        ]
        engine = CoReferenceEngine(elements)
        groups = engine.resolve()

        emp_groups = [g for g in groups if g.category == "QUASI_IDENTIFIER"]
        assert len(emp_groups) == 1
        assert emp_groups[0].occurrence_count == 2
