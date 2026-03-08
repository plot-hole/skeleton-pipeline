from __future__ import annotations

import logging

from llm_deidentifier.models.entities import ReplacementSpan

logger = logging.getLogger(__name__)


def replace_entities(
    text: str,
    spans: list[ReplacementSpan],
) -> str:
    """Replace entity spans in the document text with their tokens.

    Processes spans right-to-left so earlier offsets are not invalidated
    by replacement length changes.
    """
    if not spans:
        return text

    # Deduplicate overlapping spans: keep the longest span at each position
    spans = _deduplicate_spans(spans)

    # Sort by char_start descending for right-to-left replacement
    sorted_spans = sorted(spans, key=lambda s: s.char_start, reverse=True)

    result = text
    for span in sorted_spans:
        # Validate span matches expected text
        actual = result[span.char_start : span.char_end]
        if actual != span.original_text:
            logger.warning(
                "Span mismatch at %d:%d — expected '%s', got '%s'. Skipping.",
                span.char_start,
                span.char_end,
                span.original_text,
                actual,
            )
            continue

        result = result[: span.char_start] + span.replacement_token + result[span.char_end :]

    return result


def _deduplicate_spans(spans: list[ReplacementSpan]) -> list[ReplacementSpan]:
    """Remove overlapping spans, preferring longer spans."""
    # Sort by start position, then by length descending
    sorted_spans = sorted(spans, key=lambda s: (s.char_start, -(s.char_end - s.char_start)))

    result: list[ReplacementSpan] = []
    last_end = -1

    for span in sorted_spans:
        if span.char_start >= last_end:
            result.append(span)
            last_end = span.char_end
        else:
            # Overlapping — skip the shorter/later one
            logger.debug(
                "Dropping overlapping span '%s' at %d:%d",
                span.original_text,
                span.char_start,
                span.char_end,
            )

    return result
