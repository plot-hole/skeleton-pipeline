"""Co-reference resolution engine.

Groups all mentions of the same real-world entity so they receive a single,
consistent replacement token throughout the document.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

from src.schemas import ClassifiedElement, EntityGroup


# ---------------------------------------------------------------------------
# Honorific / title stripping
# ---------------------------------------------------------------------------

_HONORIFICS = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor",
    "jr", "sr", "ii", "iii", "iv", "esq",
}


def _strip_honorifics(name: str) -> str:
    """Remove common honorifics and suffixes from a name string."""
    tokens = name.split()
    cleaned = [
        t for t in tokens
        if t.lower().rstrip(".") not in _HONORIFICS
    ]
    return " ".join(cleaned) if cleaned else name


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set)
# ---------------------------------------------------------------------------

class _UnionFind:
    """Simple union-find with path compression and union by rank."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def make_set(self, x: int) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def groups(self, keys: list[int]) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for k in keys:
            root = self.find(k)
            result.setdefault(root, []).append(k)
        return result


# ---------------------------------------------------------------------------
# Name parsing
# ---------------------------------------------------------------------------

@dataclass
class _ParsedName:
    original: str
    normalized: str  # stripped of honorifics, title-cased
    first: str
    last: str


def _parse_name(raw: str) -> _ParsedName:
    stripped = _strip_honorifics(raw.strip())
    normalized = stripped.title()
    parts = normalized.split()
    if len(parts) == 0:
        return _ParsedName(original=raw, normalized=normalized, first="", last="")
    if len(parts) == 1:
        return _ParsedName(original=raw, normalized=normalized, first=parts[0], last="")
    return _ParsedName(
        original=raw,
        normalized=normalized,
        first=parts[0],
        last=parts[-1],
    )


# ---------------------------------------------------------------------------
# Token generators
# ---------------------------------------------------------------------------

def _party_letter(index: int) -> str:
    """Convert 0-based index to letter(s): 0->A, 1->B, ..., 25->Z, 26->AA."""
    if index < 26:
        return string.ascii_uppercase[index]
    # For >26 parties (extremely unlikely in practice)
    return _party_letter(index // 26 - 1) + string.ascii_uppercase[index % 26]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CoReferenceEngine:
    """Resolve co-references across classified elements.

    Given a list of ``ClassifiedElement`` objects from Cyera, groups all
    mentions of the same real-world entity and assigns a stable token to
    each group.
    """

    def __init__(self, elements: list[ClassifiedElement]) -> None:
        self.elements = elements

    def resolve(self) -> list[EntityGroup]:
        """Run co-reference resolution and return entity groups."""
        groups: list[EntityGroup] = []
        groups.extend(self._resolve_names())
        groups.extend(self._resolve_minors())
        groups.extend(self._resolve_employers())
        return groups

    # -- Name resolution (DIRECT_IDENTIFIER / FULL_NAME) -------------------

    def _resolve_names(self) -> list[EntityGroup]:
        name_elements = [
            e for e in self.elements
            if e.category == "DIRECT_IDENTIFIER" and e.subcategory == "FULL_NAME"
        ]
        if not name_elements:
            return []

        # Parse each name
        parsed = [_parse_name(e.original_text) for e in name_elements]

        # Build union-find groups, scoped by data_subject_role
        uf = _UnionFind()
        indices = list(range(len(name_elements)))
        for i in indices:
            uf.make_set(i)

        for i in range(len(name_elements)):
            for j in range(i + 1, len(name_elements)):
                # NOTE: role scoping removed — the improved _names_match now
                # correctly distinguishes family members who share a surname,
                # so cross-role comparison is safe and necessary to avoid
                # assigning different tokens to the same person when the
                # alternating PRIMARY/SECONDARY role splits their mentions.
                if self._names_match(parsed[i], parsed[j]):
                    uf.union(i, j)

        # Build groups sorted by first appearance
        raw_groups = uf.groups(indices)
        sorted_roots = sorted(raw_groups.keys(), key=lambda r: min(
            name_elements[i].byte_offset_start for i in raw_groups[r]
        ))

        result: list[EntityGroup] = []
        party_idx = 0
        for root in sorted_roots:
            member_indices = raw_groups[root]
            token = f"[PARTY_{_party_letter(party_idx)}]"
            party_idx += 1
            result.append(EntityGroup(
                token=token,
                category="DIRECT_IDENTIFIER",
                element_ids=[name_elements[i].element_id for i in member_indices],
                data_subject_role=name_elements[member_indices[0]].context.data_subject_role,
                occurrence_count=len(member_indices),
            ))
        return result

    @staticmethod
    def _names_match(a: _ParsedName, b: _ParsedName) -> bool:
        """Check if two parsed names refer to the same entity.

        Surname-only matching is intentionally NOT used because legal
        documents routinely contain multiple family members who share a
        surname but are distinct people (e.g. "Gerald Hutchinson" vs
        "Thomas Hutchinson").
        """
        # 1. Exact match (after normalization)
        if a.normalized == b.normalized:
            return True

        # 2. Both names have first AND last components
        if a.last and b.last and a.first and b.first:
            # Same first + last → same person
            if (a.last.lower() == b.last.lower()
                    and a.first.lower() == b.first.lower()):
                return True
            # Word-set subset: one name's words all appear in the other.
            # Handles "Gerald Hutchinson" ↔ "Gerald Raymond Hutchinson"
            # but rejects "Thomas Hutchinson" ↔ "Gerald Raymond Hutchinson".
            a_words = set(a.normalized.lower().split())
            b_words = set(b.normalized.lower().split())
            if a_words.issubset(b_words) or b_words.issubset(a_words):
                return True
            # Same surname but different first names → different people
            if a.last.lower() == b.last.lower():
                return False

        # 3. Single-token name: match only against first name
        #    (matching against surname would transitively link distinct
        #    family members who share a surname)
        if not a.last and a.first:
            if a.first.lower() == b.first.lower():
                return True
        if not b.last and b.first:
            if b.first.lower() == a.first.lower():
                return True

        return False

    # -- Minor resolution (MINOR_INFORMATION / CHILD_NAME) -----------------

    def _resolve_minors(self) -> list[EntityGroup]:
        minor_elements = [
            e for e in self.elements
            if e.category == "MINOR_INFORMATION" and e.subcategory == "CHILD_NAME"
        ]
        if not minor_elements:
            return []

        parsed = [_parse_name(e.original_text) for e in minor_elements]

        uf = _UnionFind()
        indices = list(range(len(minor_elements)))
        for i in indices:
            uf.make_set(i)

        for i in range(len(minor_elements)):
            for j in range(i + 1, len(minor_elements)):
                if self._names_match(parsed[i], parsed[j]):
                    uf.union(i, j)

        raw_groups = uf.groups(indices)
        sorted_roots = sorted(raw_groups.keys(), key=lambda r: min(
            minor_elements[i].byte_offset_start for i in raw_groups[r]
        ))

        result: list[EntityGroup] = []
        for idx, root in enumerate(sorted_roots):
            member_indices = raw_groups[root]
            token = f"[MINOR_CHILD_{idx + 1}]"
            result.append(EntityGroup(
                token=token,
                category="MINOR_INFORMATION",
                element_ids=[minor_elements[i].element_id for i in member_indices],
                data_subject_role="MINOR_DEPENDENT",
                occurrence_count=len(member_indices),
            ))
        return result

    # -- Employer resolution (QUASI_IDENTIFIER / EMPLOYER_NAME) ------------

    def _resolve_employers(self) -> list[EntityGroup]:
        employer_elements = [
            e for e in self.elements
            if e.subcategory == "EMPLOYER_NAME"
        ]
        if not employer_elements:
            return []

        uf = _UnionFind()
        indices = list(range(len(employer_elements)))
        for i in indices:
            uf.make_set(i)

        for i in range(len(employer_elements)):
            for j in range(i + 1, len(employer_elements)):
                if employer_elements[i].original_text.lower() == employer_elements[j].original_text.lower():
                    uf.union(i, j)

        raw_groups = uf.groups(indices)
        sorted_roots = sorted(raw_groups.keys(), key=lambda r: min(
            employer_elements[i].byte_offset_start for i in raw_groups[r]
        ))

        result: list[EntityGroup] = []
        for idx, root in enumerate(sorted_roots):
            member_indices = raw_groups[root]
            rep = employer_elements[member_indices[0]]
            industry = rep.context.industry
            if industry:
                token = f"[EMPLOYER_{industry.upper()}]"
            else:
                token = f"[EMPLOYER_{idx + 1}]"
            result.append(EntityGroup(
                token=token,
                category="QUASI_IDENTIFIER",
                element_ids=[employer_elements[i].element_id for i in member_indices],
                data_subject_role="EMPLOYER",
                occurrence_count=len(member_indices),
            ))
        return result
