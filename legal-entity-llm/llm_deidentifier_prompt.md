# Claude Code Prompt: LLM-Based Legal Document De-Identifier

## Project Brief

Build a legal document de-identification pipeline that uses an LLM (Claude) as the entity detection and classification engine, replacing the spaCy NER + regex approach. The LLM handles entity extraction, role inference, and semantic token naming in a single pass — including relationships that rule-based systems miss.

---

## Why LLM Over NER

Traditional NER (spaCy) identifies *what* something is. It cannot reason about *who* someone is in context.

A rule-based system sees:
```
"Dr. Martinez testified that her patient, John Doe, was present at the scene."
```
And produces: `PERSON, PERSON`

The LLM sees the same text and produces:
```
EXPERT_WITNESS_1, PLAINTIFF_1
```

The LLM approach handles:
- **Role inference from narrative context** — plaintiff vs. defendant vs. witness from how they're described, not just a keyword match
- **Implicit relationships** — "his attorney", "the decedent's family", "the employer"
- **Ambiguous addresses** — distinguishes a workplace from a residence from a crime scene
- **Unusual legal entities** — trust names, estate names, DBA names, aliases
- **Cross-sentence reasoning** — a name mentioned once as "plaintiff" informs how the same name is classified 10 paragraphs later

---

## Architecture Overview

```
Plain text document
        ↓
┌─────────────────────────────┐
│   1. Chunker                │  Split long docs into overlapping chunks
│      (~2000 tokens each,    │  with context carry-over between chunks
│       200 token overlap)    │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│   2. LLM Extraction Pass    │  Claude extracts entities as structured JSON
│      (claude-sonnet-4-5)    │  with type, role, token_prefix, and context
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│   3. Entity Resolver        │  Deduplicate across chunks
│                             │  Unify aliases ("Mr. Smith" = "John Smith")
│                             │  Assign final canonical token per entity
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│   4. SQLite Store           │  Cross-document persistence
│                             │  Same entity → same token across all docs
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│   5. Replacer               │  Span-based text replacement
│                             │  Right-to-left to preserve offsets
└────────────┬────────────────┘
             ↓
Tokenized plain text output
```

---

## Project Structure

```
llm_deidentifier/
├── main.py
├── config.py                    # model name, chunk size, overlap, token schema
├── pipeline/
│   ├── chunker.py               # sentence-aware document splitter
│   ├── extractor.py             # LLM API call + structured JSON parsing
│   ├── resolver.py              # cross-chunk entity deduplication + alias merging
│   └── replacer.py              # span replacement engine
├── store/
│   └── entity_store.py          # SQLite cross-document mapping (reuse from v1)
├── prompts/
│   └── extraction_prompt.py     # system + user prompt templates
├── models/
│   └── entities.py              # Pydantic models for LLM response schema
├── requirements.txt
└── README.md
```

---

## Extraction Prompt Design

This is the most important part of the system. The prompt must instruct the LLM to:

1. Return **only valid JSON**, no prose
2. Extract every PII entity in the chunk
3. Assign a **semantic token prefix** (not just a type label)
4. Provide the **exact verbatim text** as it appears in the document (for span matching)
5. Identify **aliases** — different surface forms of the same underlying entity
6. Flag **uncertainty** when role cannot be determined confidently

### System Prompt

```
You are a legal document de-identification engine. Your job is to extract all personally
identifiable information (PII) from legal document text and return structured JSON only.

For each entity found, determine:
- Its exact text as it appears in the document
- Its semantic role in the legal context
- The best token prefix for replacement

Return ONLY a JSON object. No explanation, no markdown, no preamble.
```

### User Prompt Template

```
Extract all PII entities from the following legal document excerpt.

KNOWN ENTITIES FROM PRIOR CHUNKS (use these to resolve aliases):
{known_entities_json}

DOCUMENT EXCERPT:
{chunk_text}

Return this exact JSON structure:
{
  "entities": [
    {
      "text": "exact text from document",
      "token_prefix": "PLAINTIFF | DEFENDANT | JUDGE | ATTORNEY | EXPERT_WITNESS | WITNESS | VICTIM | MINOR | LAW_FIRM | CORPORATION | COURT | GOVERNMENT_AGENCY | RURAL_{STATE}_ADDRESS | URBAN_{STATE}_ADDRESS | SUBURBAN_{STATE}_ADDRESS | INCIDENT_DATE | FILING_DATE | HEARING_DATE | BIRTH_DATE | SETTLEMENT_AMOUNT | DAMAGES_AMOUNT | CASE_NUMBER | SSN | PHONE | EMAIL | EXHIBIT | STATUTE | PERSON | ORGANIZATION | ADDRESS | DATE | DOLLAR_AMOUNT",
      "entity_type": "PERSON | ORG | ADDRESS | DATE | MONEY | ID",
      "alias_of": "canonical entity text this is an alias for, or null",
      "reasoning": "one sentence explaining role inference",
      "confidence": "HIGH | MEDIUM | LOW"
    }
  ]
}

Rules:
- text must be the EXACT string from the document, character-for-character
- For addresses, classify as RURAL/URBAN/SUBURBAN based on context clues (rural route, farm, township vs city, downtown, metropolitan)
- For persons, infer role from surrounding context (how they are described, referenced, their actions)
- Mark pronouns (he, she, they, his, her) as aliases pointing to their canonical name
- Mark honorifics/partial names ("Mr. Smith", "the plaintiff") as aliases
- If role is genuinely ambiguous, use the generic type (PERSON, ORGANIZATION, ADDRESS)
- Return LOW confidence when role is inferred with limited context
```

---

## Key Implementation Details

### 1. Chunker (`pipeline/chunker.py`)

- Split on sentence boundaries (use `nltk.sent_tokenize` or spaCy sentences), never mid-sentence
- Each chunk: ~2000 tokens (~1500 words)
- Overlap: carry the last 2–3 sentences of each chunk into the next
- Track absolute character offsets for each chunk so span positions map back to the original document

```python
@dataclass
class Chunk:
    text: str
    char_offset: int    # absolute start position in original document
    chunk_index: int
```

### 2. Extractor (`pipeline/extractor.py`)

- Call `claude-sonnet-4-5` via the Anthropic SDK
- Pass `known_entities` from prior chunks to help resolve aliases
- Parse JSON response — validate with Pydantic before using
- On JSON parse failure: retry once with a stricter prompt, then log and skip chunk
- Find entity spans using exact string search within the chunk (accounting for char_offset)

```python
@dataclass
class ExtractedEntity:
    text: str
    token_prefix: str
    entity_type: str
    alias_of: str | None
    reasoning: str
    confidence: str       # HIGH | MEDIUM | LOW
    char_start: int       # absolute position in original doc
    char_end: int
```

### 3. Resolver (`pipeline/resolver.py`)

Cross-chunk deduplication and alias merging is the hardest part:

**Deduplication rules (apply in order):**
1. Exact match (case-insensitive, stripped) → same entity
2. Alias resolution — if entity A has `alias_of` pointing to entity B, merge into B
3. Substring containment — "John Smith" and "Smith" likely the same if same type
4. LLM-assisted resolution — for ambiguous cases, make a second cheap LLM call:
   *"Are these the same person in a legal document? 'John Smith (plaintiff)' and 'Mr. Smith'. Answer YES or NO only."*

**Resolution produces a canonical entity map:**
```python
{
  "john smith":     {"canonical": "John Smith", "token_prefix": "PLAINTIFF", "token": None},
  "mr. smith":      {"canonical": "John Smith", "token_prefix": "PLAINTIFF", "token": None},
  "the plaintiff":  {"canonical": "John Smith", "token_prefix": "PLAINTIFF", "token": None},
  "he":             {"canonical": "John Smith", "token_prefix": "PLAINTIFF", "token": None},
}
```

After resolution, assign tokens from the SQLite store (get_or_create_token).

### 4. Replacer (`pipeline/replacer.py`)

Same right-to-left replacement strategy as v1. Key difference: the span list comes from
LLM-extracted character positions rather than spaCy, so validate spans:
- Confirm `doc[start:end] == entity.text` before replacing
- If mismatch (LLM hallucinated position), fall back to `str.find()` for the entity text

---

## Pydantic Models (`models/entities.py`)

```python
from pydantic import BaseModel
from typing import Literal

class ExtractedEntityRaw(BaseModel):
    text: str
    token_prefix: str
    entity_type: Literal["PERSON", "ORG", "ADDRESS", "DATE", "MONEY", "ID"]
    alias_of: str | None = None
    reasoning: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]

class ExtractionResponse(BaseModel):
    entities: list[ExtractedEntityRaw]
```

---

## Config (`config.py`)

```python
MODEL = "claude-sonnet-4-5"
MAX_TOKENS_RESPONSE = 4096
CHUNK_SIZE_WORDS = 1500
CHUNK_OVERLAP_SENTENCES = 3
MIN_CONFIDENCE_TO_REPLACE = "LOW"   # or "MEDIUM" to be more conservative
DB_PATH = "entity_store.db"

# Token prefixes the LLM is allowed to use — include these in the prompt
VALID_TOKEN_PREFIXES = [
    "PLAINTIFF", "DEFENDANT", "JUDGE", "ATTORNEY",
    "EXPERT_WITNESS", "WITNESS", "VICTIM", "MINOR",
    "LAW_FIRM", "CORPORATION", "COURT", "GOVERNMENT_AGENCY",
    "RURAL_{STATE}_ADDRESS", "URBAN_{STATE}_ADDRESS", "SUBURBAN_{STATE}_ADDRESS",
    "INCIDENT_DATE", "FILING_DATE", "HEARING_DATE", "BIRTH_DATE",
    "SETTLEMENT_AMOUNT", "DAMAGES_AMOUNT", "MEDICAL_COST",
    "CASE_NUMBER", "SSN", "PHONE", "EMAIL", "EXHIBIT", "STATUTE",
    # Generic fallbacks
    "PERSON", "ORGANIZATION", "ADDRESS", "DATE", "DOLLAR_AMOUNT",
]
```

---

## Requirements

```
anthropic>=0.25.0
pydantic>=2.0.0
nltk>=3.8.0        # sentence tokenization for chunker
```

No spaCy, no geopy, no pgeocode — the LLM handles classification entirely.

---

## Comparison to v1 (spaCy-based)

| Concern | v1 (spaCy + Rules) | v2 (LLM) |
|---|---|---|
| Role inference | Keyword matching in context window | Full narrative reasoning |
| Address enrichment | ZIP → RUCC lookup table | Context clues ("rural route", "farm", "downtown") |
| Co-reference | Separate coreferee model | Native, single pass |
| Alias merging | Exact/normalized match only | Semantic deduplication |
| Legal ID patterns | Hardcoded regex | LLM recognizes novel formats |
| Speed | Fast (local, no API) | Slower (API calls per chunk) |
| Cost | Free | ~$0.01–0.05 per document |
| Accuracy on edge cases | Limited | High |
| Offline operation | Yes | No |
| Determinism | Yes | Near-deterministic with low temp |

---

## Error Handling Requirements

- **JSON parse failure**: retry once with `"Return only raw JSON, no other text."` appended; if still fails, log chunk as unprocessed and continue
- **Span mismatch**: when LLM-reported char positions don't match, use `text.find()` fallback; log mismatches for review
- **Low confidence entities**: still replace by default, but tag in manifest as `"confidence": "LOW"` for human review
- **Rate limits**: implement exponential backoff with jitter on 429 responses
- **Empty chunk**: skip gracefully

---

## Output

Same interface as v1:

```python
di = DeIdentifier()
result = di.process_file("complaint.txt")
result.save("complaint_clean.txt")
result.save_manifest("manifest.json")   # includes LLM reasoning per entity
```

The manifest should include the `reasoning` field from the LLM so reviewers can audit
*why* John Smith was classified as PLAINTIFF_1 rather than WITNESS_1.

---

## Out of Scope

- Multi-language support
- PDF/DOCX parsing (plain text input only)
- Fine-tuning or custom model training
- Real-time streaming (process full document, then return)
- GUI or web interface
