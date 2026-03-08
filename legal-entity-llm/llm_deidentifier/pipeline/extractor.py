from __future__ import annotations

import json
import logging
import time
from typing import Any

import anthropic

from llm_deidentifier.config import MAX_TOKENS_RESPONSE, MODEL
from llm_deidentifier.models.entities import (
    Chunk,
    ExtractedEntity,
    ExtractionResponse,
)
from llm_deidentifier.prompts.extraction_prompt import (
    RETRY_SUFFIX,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


def _build_known_entities_json(known_entities: list[dict[str, Any]]) -> str:
    """Format previously-seen entities for injection into the prompt."""
    if not known_entities:
        return "None yet — this is the first chunk."
    return json.dumps(known_entities, indent=2)


def _call_llm(
    client: anthropic.Anthropic,
    user_prompt: str,
    *,
    retry: bool = False,
) -> ExtractionResponse | None:
    """Make a single LLM call and parse the JSON response."""
    if retry:
        user_prompt += RETRY_SUFFIX

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS_RESPONSE,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            break
        except anthropic.RateLimitError:
            wait = (2 ** attempt) + (time.monotonic() % 1)  # exp backoff + jitter
            logger.warning("Rate limited, retrying in %.1fs", wait)
            time.sleep(wait)
    else:
        logger.error("LLM call failed after retries (rate limit)")
        return None

    raw_text = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_text = "\n".join(lines)

    try:
        parsed = ExtractionResponse.model_validate_json(raw_text)
        return parsed
    except Exception:
        # Try loading as plain JSON first
        try:
            data = json.loads(raw_text)
            parsed = ExtractionResponse.model_validate(data)
            return parsed
        except Exception as exc:
            logger.warning("JSON parse failure: %s", exc)
            return None


def _is_word_boundary(doc: str, pos: int) -> bool:
    """Check if position is at a word boundary (start of string, or preceded by non-alnum)."""
    if pos <= 0 or pos >= len(doc):
        return True
    return not doc[pos - 1].isalnum()


def _is_word_end_boundary(doc: str, pos: int) -> bool:
    """Check if position is at a word end boundary (end of string, or followed by non-alnum)."""
    if pos >= len(doc):
        return True
    return not doc[pos].isalnum()


def _find_entity_spans(
    entity_text: str,
    chunk: Chunk,
    original_doc: str,
) -> list[tuple[int, int]]:
    """Find all occurrences of entity_text in the chunk, returning absolute offsets.

    Enforces word boundaries so short strings like 'his' don't match inside 'this'.
    """
    spans: list[tuple[int, int]] = []
    search_text = chunk.text
    start = 0
    while True:
        idx = search_text.find(entity_text, start)
        if idx == -1:
            break
        abs_start = chunk.char_offset + idx
        abs_end = abs_start + len(entity_text)
        # Verify against original doc and enforce word boundaries
        if (
            original_doc[abs_start:abs_end] == entity_text
            and _is_word_boundary(original_doc, abs_start)
            and _is_word_end_boundary(original_doc, abs_end)
        ):
            spans.append((abs_start, abs_end))
        start = idx + 1
    return spans


def extract_entities_from_chunk(
    client: anthropic.Anthropic,
    chunk: Chunk,
    original_doc: str,
    known_entities: list[dict[str, Any]],
) -> list[ExtractedEntity]:
    """Run LLM extraction on a single chunk and return positioned entities."""
    if not chunk.text.strip():
        return []

    known_json = _build_known_entities_json(known_entities)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        known_entities_json=known_json,
        chunk_text=chunk.text,
    )

    result = _call_llm(client, user_prompt)

    # Retry once on parse failure
    if result is None:
        logger.info("Retrying chunk %d with stricter prompt", chunk.chunk_index)
        result = _call_llm(client, user_prompt, retry=True)

    if result is None:
        logger.error("Skipping chunk %d — LLM extraction failed", chunk.chunk_index)
        return []

    entities: list[ExtractedEntity] = []
    for raw in result.entities:
        spans = _find_entity_spans(raw.text, chunk, original_doc)
        if not spans:
            logger.warning(
                "Entity '%s' not found in chunk %d via exact match, trying find()",
                raw.text,
                chunk.chunk_index,
            )
            # Fallback: search in original doc within chunk bounds
            idx = original_doc.find(
                raw.text,
                chunk.char_offset,
                chunk.char_offset + len(chunk.text),
            )
            if idx != -1:
                spans = [(idx, idx + len(raw.text))]
            else:
                logger.warning("Entity '%s' not found at all — skipping", raw.text)
                continue

        for char_start, char_end in spans:
            entities.append(ExtractedEntity(
                text=raw.text,
                token_prefix=raw.token_prefix,
                entity_type=raw.entity_type,
                alias_of=raw.alias_of,
                reasoning=raw.reasoning,
                confidence=raw.confidence,
                char_start=char_start,
                char_end=char_end,
            ))

    return entities
