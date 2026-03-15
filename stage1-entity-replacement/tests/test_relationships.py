"""Tests for the relationship graph builder."""

import pytest

from src.relationships import RelationshipGraphBuilder
from src.schemas import (
    ClassifiedElement,
    CohabitationRelationship,
    ElementContext,
    EmploymentRelationship,
    EntityGroup,
    ParentChildRelationship,
)


def _el(
    original_text: str,
    element_id: str,
    offset_start: int,
    category: str = "DIRECT_IDENTIFIER",
    subcategory: str = "FULL_NAME",
    role: str = "PARTY_PRIMARY",
    industry: str | None = None,
) -> ClassifiedElement:
    return ClassifiedElement(
        element_id=element_id,
        category=category,
        subcategory=subcategory,
        sensitivity="HIGH",
        byte_offset_start=offset_start,
        byte_offset_end=offset_start + len(original_text.encode("utf-8")),
        original_text=original_text,
        context=ElementContext(
            data_subject_role=role,
            jurisdiction="US_CA",
            industry=industry,
        ),
    )


class TestCohabitation:
    def test_two_parties_share_address(self):
        """Two PARTY tokens near the same ADDRESS -> COHABITATION."""
        elements = [
            _el("Sarah Mitchell", "e1", offset_start=0, role="PARTY_PRIMARY"),
            _el("David Chen", "e2", offset_start=20, role="PARTY_SECONDARY"),
            _el("142 Oak St, Sacramento, CA 95814", "e3", offset_start=40,
                 category="QUASI_IDENTIFIER", subcategory="ADDRESS"),
        ]
        groups = [
            EntityGroup(token="[PARTY_A]", category="DIRECT_IDENTIFIER",
                        element_ids=["e1"], data_subject_role="PARTY_PRIMARY", occurrence_count=1),
            EntityGroup(token="[PARTY_B]", category="DIRECT_IDENTIFIER",
                        element_ids=["e2"], data_subject_role="PARTY_SECONDARY", occurrence_count=1),
        ]

        builder = RelationshipGraphBuilder(elements, groups)
        graph = builder.build()

        cohab = [r for r in graph.relationships if r.type == "COHABITATION"]
        assert len(cohab) == 1
        assert set(cohab[0].entities) == {"[PARTY_A]", "[PARTY_B]"}

    def test_no_cohabitation_when_far_apart(self):
        """Parties too far from address -> no COHABITATION."""
        elements = [
            _el("Sarah Mitchell", "e1", offset_start=0, role="PARTY_PRIMARY"),
            _el("David Chen", "e2", offset_start=1000, role="PARTY_SECONDARY"),
            _el("142 Oak St, Sacramento, CA 95814", "e3", offset_start=2000,
                 category="QUASI_IDENTIFIER", subcategory="ADDRESS"),
        ]
        groups = [
            EntityGroup(token="[PARTY_A]", category="DIRECT_IDENTIFIER",
                        element_ids=["e1"], data_subject_role="PARTY_PRIMARY", occurrence_count=1),
            EntityGroup(token="[PARTY_B]", category="DIRECT_IDENTIFIER",
                        element_ids=["e2"], data_subject_role="PARTY_SECONDARY", occurrence_count=1),
        ]

        builder = RelationshipGraphBuilder(elements, groups)
        graph = builder.build()

        cohab = [r for r in graph.relationships if r.type == "COHABITATION"]
        assert len(cohab) == 0

    def test_single_party_near_address_no_cohabitation(self):
        """Only one party near address -> no COHABITATION (requires 2+)."""
        elements = [
            _el("Sarah Mitchell", "e1", offset_start=0, role="PARTY_PRIMARY"),
            _el("142 Oak St", "e3", offset_start=20,
                 category="QUASI_IDENTIFIER", subcategory="ADDRESS"),
        ]
        groups = [
            EntityGroup(token="[PARTY_A]", category="DIRECT_IDENTIFIER",
                        element_ids=["e1"], data_subject_role="PARTY_PRIMARY", occurrence_count=1),
        ]

        builder = RelationshipGraphBuilder(elements, groups)
        graph = builder.build()

        cohab = [r for r in graph.relationships if r.type == "COHABITATION"]
        assert len(cohab) == 0


class TestParentChild:
    def test_party_and_minor_proximate(self):
        """PARTY near MINOR -> PARENT_CHILD."""
        elements = [
            _el("Sarah Mitchell", "e1", offset_start=0, role="PARTY_PRIMARY"),
            _el("Emma", "m1", offset_start=50,
                 category="MINOR_INFORMATION", subcategory="CHILD_NAME",
                 role="MINOR_DEPENDENT"),
        ]
        groups = [
            EntityGroup(token="[PARTY_A]", category="DIRECT_IDENTIFIER",
                        element_ids=["e1"], data_subject_role="PARTY_PRIMARY", occurrence_count=1),
            EntityGroup(token="[MINOR_CHILD_1]", category="MINOR_INFORMATION",
                        element_ids=["m1"], data_subject_role="MINOR_DEPENDENT", occurrence_count=1),
        ]

        builder = RelationshipGraphBuilder(elements, groups)
        graph = builder.build()

        pc = [r for r in graph.relationships if r.type == "PARENT_CHILD"]
        assert len(pc) == 1
        assert pc[0].entity == "[PARTY_A]"
        assert "[MINOR_CHILD_1]" in pc[0].dependents

    def test_two_minors_same_parent(self):
        elements = [
            _el("Sarah Mitchell", "e1", offset_start=0, role="PARTY_PRIMARY"),
            _el("Emma", "m1", offset_start=50,
                 category="MINOR_INFORMATION", subcategory="CHILD_NAME",
                 role="MINOR_DEPENDENT"),
            _el("Liam", "m2", offset_start=70,
                 category="MINOR_INFORMATION", subcategory="CHILD_NAME",
                 role="MINOR_DEPENDENT"),
        ]
        groups = [
            EntityGroup(token="[PARTY_A]", category="DIRECT_IDENTIFIER",
                        element_ids=["e1"], data_subject_role="PARTY_PRIMARY", occurrence_count=1),
            EntityGroup(token="[MINOR_CHILD_1]", category="MINOR_INFORMATION",
                        element_ids=["m1"], data_subject_role="MINOR_DEPENDENT", occurrence_count=1),
            EntityGroup(token="[MINOR_CHILD_2]", category="MINOR_INFORMATION",
                        element_ids=["m2"], data_subject_role="MINOR_DEPENDENT", occurrence_count=1),
        ]

        builder = RelationshipGraphBuilder(elements, groups)
        graph = builder.build()

        pc = [r for r in graph.relationships if r.type == "PARENT_CHILD"]
        assert len(pc) == 1
        assert set(pc[0].dependents) == {"[MINOR_CHILD_1]", "[MINOR_CHILD_2]"}

    def test_no_minors_no_parent_child(self):
        elements = [
            _el("Sarah Mitchell", "e1", offset_start=0, role="PARTY_PRIMARY"),
        ]
        groups = [
            EntityGroup(token="[PARTY_A]", category="DIRECT_IDENTIFIER",
                        element_ids=["e1"], data_subject_role="PARTY_PRIMARY", occurrence_count=1),
        ]

        builder = RelationshipGraphBuilder(elements, groups)
        graph = builder.build()

        pc = [r for r in graph.relationships if r.type == "PARENT_CHILD"]
        assert len(pc) == 0


class TestEmployment:
    def test_party_near_employer(self):
        elements = [
            _el("Sarah Mitchell", "e1", offset_start=0, role="PARTY_PRIMARY"),
            _el("Acme Corp", "emp1", offset_start=50,
                 category="QUASI_IDENTIFIER", subcategory="EMPLOYER_NAME",
                 role="EMPLOYER", industry="TECH"),
        ]
        groups = [
            EntityGroup(token="[PARTY_A]", category="DIRECT_IDENTIFIER",
                        element_ids=["e1"], data_subject_role="PARTY_PRIMARY", occurrence_count=1),
            EntityGroup(token="[EMPLOYER_TECH]", category="QUASI_IDENTIFIER",
                        element_ids=["emp1"], data_subject_role="EMPLOYER", occurrence_count=1),
        ]

        builder = RelationshipGraphBuilder(elements, groups)
        graph = builder.build()

        emp = [r for r in graph.relationships if r.type == "EMPLOYMENT"]
        assert len(emp) == 1
        assert emp[0].entity == "[PARTY_A]"
        assert emp[0].employer == "[EMPLOYER_TECH]"
        assert emp[0].industry == "TECH"

    def test_no_employers_no_employment(self):
        elements = [
            _el("Sarah Mitchell", "e1", offset_start=0, role="PARTY_PRIMARY"),
        ]
        groups = [
            EntityGroup(token="[PARTY_A]", category="DIRECT_IDENTIFIER",
                        element_ids=["e1"], data_subject_role="PARTY_PRIMARY", occurrence_count=1),
        ]

        builder = RelationshipGraphBuilder(elements, groups)
        graph = builder.build()

        emp = [r for r in graph.relationships if r.type == "EMPLOYMENT"]
        assert len(emp) == 0


class TestEmptyGraph:
    def test_no_elements(self):
        builder = RelationshipGraphBuilder([], [])
        graph = builder.build()
        assert len(graph.relationships) == 0
