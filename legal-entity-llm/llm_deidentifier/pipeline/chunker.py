from __future__ import annotations

import logging

import nltk
from nltk.tokenize import sent_tokenize

from llm_deidentifier.config import CHUNK_OVERLAP_SENTENCES, CHUNK_SIZE_WORDS
from llm_deidentifier.models.entities import Chunk

logger = logging.getLogger(__name__)

# Ensure punkt tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def chunk_document(text: str) -> list[Chunk]:
    """Split a document into overlapping sentence-aligned chunks.

    Each chunk is ~CHUNK_SIZE_WORDS words. The last CHUNK_OVERLAP_SENTENCES
    sentences of a chunk are carried into the next to provide context.
    Character offsets map each chunk back to its position in the original text.
    """
    if not text.strip():
        return []

    sentences = sent_tokenize(text)
    if not sentences:
        return []

    # Pre-compute the absolute char offset of each sentence in the original text
    sentence_offsets: list[int] = []
    search_start = 0
    for sent in sentences:
        idx = text.find(sent, search_start)
        if idx == -1:
            # Fallback: use current search position
            idx = search_start
        sentence_offsets.append(idx)
        search_start = idx + len(sent)

    chunks: list[Chunk] = []
    i = 0
    chunk_index = 0

    while i < len(sentences):
        # Accumulate sentences until we hit the word limit
        chunk_sentences: list[int] = []  # indices into `sentences`
        word_count = 0

        j = i
        while j < len(sentences):
            words_in_sent = len(sentences[j].split())
            if word_count + words_in_sent > CHUNK_SIZE_WORDS and chunk_sentences:
                break
            chunk_sentences.append(j)
            word_count += words_in_sent
            j += 1

        if not chunk_sentences:
            break

        # Build chunk text from the original document to preserve exact characters
        first_idx = chunk_sentences[0]
        last_idx = chunk_sentences[-1]
        char_start = sentence_offsets[first_idx]
        char_end = sentence_offsets[last_idx] + len(sentences[last_idx])
        chunk_text = text[char_start:char_end]

        chunks.append(Chunk(
            text=chunk_text,
            char_offset=char_start,
            chunk_index=chunk_index,
        ))
        chunk_index += 1

        # Advance: start next chunk after removing overlap sentences
        overlap_start = max(0, len(chunk_sentences) - CHUNK_OVERLAP_SENTENCES)
        i = chunk_sentences[overlap_start]

        # Guard against infinite loops when overlap >= chunk size
        if i == chunk_sentences[0] and len(chunk_sentences) > 0:
            i = chunk_sentences[-1] + 1

    return chunks
