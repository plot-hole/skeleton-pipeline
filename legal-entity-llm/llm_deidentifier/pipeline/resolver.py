from __future__ import annotations

import logging
from typing import Any

import anthropic

from llm_deidentifier.config import MODEL
from llm_deidentifier.models.entities import ExtractedEntity, ResolvedEntity
from llm_deidentifier.prompts.extraction_prompt import RESOLUTION_PROMPT

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return text.strip().lower()


def _llm_resolve(
    client: anthropic.Anthropic,
    entity_a: str,
    type_a: str,
    entity_b: str,
    type_b: str,
) -> bool:
    """Ask the LLM whether two entities are the same. Returns True if YES."""
    prompt = RESOLUTION_PROMPT.format(
        entity_a=entity_a,
        type_a=type_a,
        entity_b=entity_b,
        type_b=type_b,
    )
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=8,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text.strip().upper()
        return answer.startswith("YES")
    except Exception as exc:
        logger.warning("LLM resolution call failed: %s", exc)
        return False


class EntityResolver:
    """Deduplicate and merge extracted entities across all chunks."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client
        # canonical_key -> ResolvedEntity
        self._canonical_map: dict[str, ResolvedEntity] = {}
        # normalized surface form -> canonical_key
        self._alias_index: dict[str, str] = {}

    @property
    def canonical_map(self) -> dict[str, ResolvedEntity]:
        return dict(self._canonical_map)

    def known_entities_for_prompt(self) -> list[dict[str, Any]]:
        """Return a summary of known entities suitable for injecting into the next prompt."""
        return [
            {
                "text": ent.canonical_text,
                "token_prefix": ent.token_prefix,
                "entity_type": ent.entity_type,
            }
            for ent in self._canonical_map.values()
        ]

    def _find_canonical(self, text: str, entity_type: str) -> str | None:
        """Try to find an existing canonical key for the given text."""
        norm = _normalize(text)

        # 1. Exact match
        if norm in self._alias_index:
            return self._alias_index[norm]

        # 2. Substring containment (only for same type)
        for alias_norm, canon_key in self._alias_index.items():
            canon_ent = self._canonical_map[canon_key]
            if canon_ent.entity_type != entity_type:
                continue
            if norm in alias_norm or alias_norm in norm:
                return canon_key

        return None

    def resolve(self, entities: list[ExtractedEntity]) -> dict[str, str]:
        """Process a batch of extracted entities and return a mapping of
        normalized-text -> canonical_key for all of them.

        Returns: {normalized_surface_form: canonical_key}
        """
        result_map: dict[str, str] = {}

        for ent in entities:
            norm = _normalize(ent.text)

            # Skip pronouns that have no alias target
            if norm in ("he", "she", "they", "his", "her", "him", "them") and not ent.alias_of:
                continue

            # If the LLM already told us this is an alias
            if ent.alias_of:
                alias_target_norm = _normalize(ent.alias_of)
                # Find or create the canonical entity for the alias target
                canon_key = self._find_canonical(ent.alias_of, ent.entity_type)
                if canon_key is None:
                    # The alias target itself hasn't been seen yet — register it
                    canon_key = alias_target_norm
                    self._canonical_map[canon_key] = ResolvedEntity(
                        canonical_text=ent.alias_of,
                        token_prefix=ent.token_prefix,
                        entity_type=ent.entity_type,
                        token=None,
                        confidence=ent.confidence,
                        reasoning=ent.reasoning,
                    )
                    self._alias_index[alias_target_norm] = canon_key

                self._alias_index[norm] = canon_key
                result_map[norm] = canon_key
                continue

            # Try to match to an existing canonical
            canon_key = self._find_canonical(ent.text, ent.entity_type)
            if canon_key is not None:
                self._alias_index[norm] = canon_key
                result_map[norm] = canon_key
                continue

            # No match found — register as new canonical entity
            canon_key = norm
            self._canonical_map[canon_key] = ResolvedEntity(
                canonical_text=ent.text,
                token_prefix=ent.token_prefix,
                entity_type=ent.entity_type,
                token=None,
                confidence=ent.confidence,
                reasoning=ent.reasoning,
            )
            self._alias_index[norm] = canon_key
            result_map[norm] = canon_key

        return result_map

    def resolve_ambiguous(
        self,
        entities: list[ExtractedEntity],
    ) -> None:
        """Second pass: use LLM to resolve entities that might be the same
        but weren't caught by exact/substring matching."""
        unresolved: list[tuple[str, ResolvedEntity]] = []
        for key, ent in self._canonical_map.items():
            if ent.entity_type == "PERSON":
                unresolved.append((key, ent))

        # Pairwise comparison (only for small sets to avoid cost explosion)
        if len(unresolved) > 20:
            logger.info("Too many unresolved entities (%d) — skipping LLM resolution", len(unresolved))
            return

        merged: set[str] = set()
        for i, (key_a, ent_a) in enumerate(unresolved):
            if key_a in merged:
                continue
            for key_b, ent_b in unresolved[i + 1 :]:
                if key_b in merged:
                    continue
                if ent_a.entity_type != ent_b.entity_type:
                    continue

                is_same = _llm_resolve(
                    self._client,
                    ent_a.canonical_text,
                    ent_a.entity_type,
                    ent_b.canonical_text,
                    ent_b.entity_type,
                )
                if is_same:
                    # Merge B into A
                    self._alias_index[key_b] = key_a
                    # Update all aliases that pointed to B
                    for alias, target in list(self._alias_index.items()):
                        if target == key_b:
                            self._alias_index[alias] = key_a
                    merged.add(key_b)
                    del self._canonical_map[key_b]
                    logger.info("Merged '%s' into '%s'", ent_b.canonical_text, ent_a.canonical_text)

    def get_token_for_text(self, text: str) -> str | None:
        """Look up the assigned token for a given surface form."""
        norm = _normalize(text)
        canon_key = self._alias_index.get(norm)
        if canon_key is None:
            return None
        canon_ent = self._canonical_map.get(canon_key)
        if canon_ent is None:
            return None
        return canon_ent.token

    def set_token(self, canonical_key: str, token: str) -> None:
        """Assign a replacement token to a canonical entity."""
        if canonical_key in self._canonical_map:
            self._canonical_map[canonical_key].token = token
