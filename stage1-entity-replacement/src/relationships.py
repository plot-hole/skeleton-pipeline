"""Relationship graph builder.

Detects structural relationships between entity tokens (COHABITATION,
PARENT_CHILD, EMPLOYMENT) based on proximity in the original document.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

from src.schemas import (
    ClassifiedElement,
    CohabitationRelationship,
    EmploymentRelationship,
    EntityGroup,
    ParentChildRelationship,
    RelationshipGraph,
)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "config")

_gen_rules: dict | None = None


def _load_rules() -> dict:
    global _gen_rules
    if _gen_rules is None:
        path = os.path.join(_CONFIG_DIR, "generalization_rules.yaml")
        with open(path, encoding="utf-8") as f:
            _gen_rules = yaml.safe_load(f)
    return _gen_rules


def _proximity_threshold() -> int:
    rules = _load_rules()
    return rules.get("relationship_proximity_threshold", 500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _midpoint(el: ClassifiedElement) -> float:
    return (el.byte_offset_start + el.byte_offset_end) / 2.0


def _are_proximate(a: ClassifiedElement, b: ClassifiedElement, threshold: int) -> bool:
    """Check if two elements are within *threshold* bytes of each other."""
    # Distance between the closest edges
    if a.byte_offset_end <= b.byte_offset_start:
        dist = b.byte_offset_start - a.byte_offset_end
    elif b.byte_offset_end <= a.byte_offset_start:
        dist = a.byte_offset_start - b.byte_offset_end
    else:
        dist = 0  # overlapping
    return dist <= threshold


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RelationshipGraphBuilder:
    """Build a relationship graph from classified elements and entity groups.

    All detection operates on the *original* metadata (pre-replacement)
    using byte-offset proximity heuristics.
    """

    def __init__(
        self,
        elements: list[ClassifiedElement],
        entity_groups: list[EntityGroup],
    ) -> None:
        self.elements = elements
        self.entity_groups = entity_groups
        self._threshold = _proximity_threshold()

        # Build lookup: element_id -> token
        self._eid_to_token: dict[str, str] = {}
        self._eid_to_group: dict[str, EntityGroup] = {}
        for group in entity_groups:
            for eid in group.element_ids:
                self._eid_to_token[eid] = group.token
                self._eid_to_group[eid] = group

    def build(self) -> RelationshipGraph:
        rels: list[Any] = []
        rels.extend(self._detect_cohabitation())
        rels.extend(self._detect_parent_child())
        rels.extend(self._detect_employment())
        return RelationshipGraph(relationships=rels)

    # -- COHABITATION -------------------------------------------------------

    def _detect_cohabitation(self) -> list[CohabitationRelationship]:
        """Two PARTY tokens proximate to the same ADDRESS -> COHABITATION."""
        address_elements = [
            e for e in self.elements
            if e.category == "QUASI_IDENTIFIER" and e.subcategory == "ADDRESS"
        ]
        party_elements = [
            e for e in self.elements
            if e.category == "DIRECT_IDENTIFIER" and e.subcategory == "FULL_NAME"
        ]

        results: list[CohabitationRelationship] = []
        seen: set[tuple[str, ...]] = set()

        for addr in address_elements:
            proximate_tokens: set[str] = set()
            for pe in party_elements:
                if _are_proximate(addr, pe, self._threshold):
                    token = self._eid_to_token.get(pe.element_id)
                    if token:
                        proximate_tokens.add(token)

            if len(proximate_tokens) >= 2:
                key = tuple(sorted(proximate_tokens))
                if key not in seen:
                    seen.add(key)
                    # Determine address token
                    addr_token = self._eid_to_token.get(addr.element_id, "ADDRESS_1")
                    results.append(CohabitationRelationship(
                        entities=sorted(proximate_tokens),
                        location=addr_token,
                    ))

        return results

    # -- PARENT_CHILD -------------------------------------------------------

    def _detect_parent_child(self) -> list[ParentChildRelationship]:
        """PARTY token proximate to MINOR token -> PARENT_CHILD."""
        party_elements = [
            e for e in self.elements
            if e.category == "DIRECT_IDENTIFIER" and e.subcategory == "FULL_NAME"
        ]
        minor_elements = [
            e for e in self.elements
            if e.category == "MINOR_INFORMATION" and e.subcategory == "CHILD_NAME"
        ]

        if not minor_elements:
            return []

        # For each minor, find the closest party
        parent_dependents: dict[str, set[str]] = {}

        for minor_el in minor_elements:
            minor_token = self._eid_to_token.get(minor_el.element_id)
            if not minor_token:
                continue

            best_party_token: str | None = None
            best_dist = float("inf")

            for pe in party_elements:
                if _are_proximate(minor_el, pe, self._threshold):
                    party_token = self._eid_to_token.get(pe.element_id)
                    if party_token:
                        dist = abs(_midpoint(minor_el) - _midpoint(pe))
                        if dist < best_dist:
                            best_dist = dist
                            best_party_token = party_token

            if best_party_token:
                parent_dependents.setdefault(best_party_token, set()).add(minor_token)

        results: list[ParentChildRelationship] = []
        for parent_token, deps in sorted(parent_dependents.items()):
            results.append(ParentChildRelationship(
                entity=parent_token,
                dependents=sorted(deps),
            ))
        return results

    # -- EMPLOYMENT ---------------------------------------------------------

    def _detect_employment(self) -> list[EmploymentRelationship]:
        """PARTY token proximate to EMPLOYER token -> EMPLOYMENT."""
        party_elements = [
            e for e in self.elements
            if e.category == "DIRECT_IDENTIFIER" and e.subcategory == "FULL_NAME"
        ]
        employer_elements = [
            e for e in self.elements
            if e.subcategory == "EMPLOYER_NAME"
        ]

        if not employer_elements:
            return []

        seen: set[tuple[str, str]] = set()
        results: list[EmploymentRelationship] = []

        for emp_el in employer_elements:
            emp_token = self._eid_to_token.get(emp_el.element_id)
            if not emp_token:
                continue
            industry = emp_el.context.industry or "UNKNOWN"

            for pe in party_elements:
                if _are_proximate(emp_el, pe, self._threshold):
                    party_token = self._eid_to_token.get(pe.element_id)
                    if party_token:
                        key = (party_token, emp_token)
                        if key not in seen:
                            seen.add(key)
                            results.append(EmploymentRelationship(
                                entity=party_token,
                                employer=emp_token,
                                industry=industry.upper(),
                            ))

        return results
