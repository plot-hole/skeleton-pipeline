# Stage 1: Entity Resolution & Replacement Pipeline

## Context

You are building the first custom stage of a privacy engineering pipeline for a legal technology platform. The pipeline converts sensitive legal documents into de-identified structural skeletons that can be used in shared AI features (recommendations, analytics, model improvement) without exposing customer PII.

This stage receives classification metadata from Cyera's DSPM platform. Cyera has already scanned the document and identified every sensitive element — names, addresses, dates, financial figures, health information, children's information, and legal boilerplate. Your job is to take that classification output and transform the document text.

**You are not building classification. Cyera does that. You are building what happens after classification.**

## What This Stage Does

1. **Consumes** Cyera classification metadata (a JSON record listing every sensitive element, its category, sensitivity level, and location in the document)
2. **Resolves** co-references (groups all mentions of the same real-world entity — "Sarah Mitchell", "Ms. Mitchell", "Sarah" → same person)
3. **Replaces** each sensitive element according to its category's replacement policy
4. **Preserves** structural relationships between entities using a relationship graph
5. **Validates** that no raw direct identifiers survive (second-pass NER scan)
6. **Outputs** the de-identified document text + token mapping table + relationship graph

## Input Schema

The pipeline receives two inputs:

### 1. Raw Document Text
Plain text extracted from the uploaded document (PDF/DOCX already converted upstream).

### 2. Cyera Classification Metadata
```json
{
  "document_id": "uuid",
  "tenant_id": "uuid",
  "classification_timestamp": "ISO-8601",
  "classifier_version": "string",
  "elements": [
    {
      "element_id": "uuid",
      "category": "DIRECT_IDENTIFIER | QUASI_IDENTIFIER | FINANCIAL_DATA | HEALTH_PROTECTED | MINOR_INFORMATION | PRIVILEGE_INDICATOR | BOILERPLATE",
      "subcategory": "FULL_NAME | SSN | ADDRESS | DATE_OF_BIRTH | INCOME | RENT | MEDICAL_CONDITION | CHILD_NAME | CHILD_AGE | ...",
      "sensitivity": "CRITICAL | HIGH | LOW",
      "byte_offset_start": 1247,
      "byte_offset_end": 1262,
      "original_text": "Sarah Mitchell",
      "context": {
        "data_subject_role": "PARTY_PRIMARY | PARTY_SECONDARY | MINOR_DEPENDENT | EMPLOYER | LEGAL_REPRESENTATIVE",
        "jurisdiction": "US_CA",
        "applicable_regulations": ["CCPA", "SOC2"]
      }
    }
  ],
  "summary": {
    "total_elements": 47,
    "critical_count": 12,
    "high_count": 18,
    "low_count": 17,
    "privilege_flag": false,
    "minor_data_present": true
  }
}
```

## Replacement Policy Table

Apply these rules based on the `category` field from Cyera's metadata:

| Category | Sensitivity | Strategy | Example Input | Example Output |
|---|---|---|---|---|
| `DIRECT_IDENTIFIER` (name) | CRITICAL | Token replace | Sarah Mitchell | `[PARTY_A]` |
| `DIRECT_IDENTIFIER` (SSN) | CRITICAL | Strip entirely | 123-45-6789 | `[SSN_REDACTED]` |
| `DIRECT_IDENTIFIER` (DL/passport) | CRITICAL | Strip entirely | D1234567 | `[ID_REDACTED]` |
| `QUASI_IDENTIFIER` (address) | HIGH | Generalize to region | 142 Oak St, Sacramento, CA 95814 | `[LOCATION_CA_METRO]` |
| `QUASI_IDENTIFIER` (date of birth) | HIGH | Generalize to age range | 03/15/1988 | `[AGE_RANGE_35_40]` |
| `QUASI_IDENTIFIER` (date) | HIGH | Generalize to period | March 15, 2026 | `[DATE_2026_Q1]` |
| `QUASI_IDENTIFIER` (employer) | HIGH | Generalize to industry | Acme Corp | `[EMPLOYER_TECH]` |
| `FINANCIAL_DATA` (income/rent) | HIGH | Bucket to range | $4,200/month | `[RENT_4000_5000]` |
| `FINANCIAL_DATA` (account number) | CRITICAL | Strip entirely | 9876543210 | `[ACCOUNT_REDACTED]` |
| `HEALTH_PROTECTED` | CRITICAL | Strip entirely | Tenant has a documented disability | `[PROTECTED_STATUS_REDACTED]` |
| `MINOR_INFORMATION` | CRITICAL | Structural token only | Emma, age 7, attends Lincoln Elementary | `[MINOR_CHILD_1]` |
| `PRIVILEGE_INDICATOR` | CRITICAL | **Halt pipeline** — document should not have reached this stage | — | — |
| `BOILERPLATE` | LOW | Pass through unchanged | This agreement shall be governed by... | *(unchanged)* |

## Architecture

### Component 1: Co-Reference Resolution Engine

**Purpose:** Group all mentions of the same real-world entity so they receive the same token.

**Why this matters:** Cyera classifies each mention independently. "Sarah Mitchell" in paragraph 1 and "Ms. Mitchell" in paragraph 14 are separate elements in the metadata. They must resolve to the same `[PARTY_A]` token or the de-identified document leaks information through inconsistency.

**Approach:**
- Build an entity graph from the classified elements
- Group by: exact match, surname match, first-name-only match (within same data_subject_role), pronoun resolution (optional for v1)
- Assign stable tokens: `[PARTY_A]`, `[PARTY_B]`, etc. ordered by first appearance
- For minors: `[MINOR_CHILD_1]`, `[MINOR_CHILD_2]`, ordered by first appearance
- For employers: `[EMPLOYER_1]`, `[EMPLOYER_2]` with industry tag from context

**Libraries to consider:**
- `spacy` with `en_core_web_lg` for NER and co-reference (used in validation, not classification)
- Custom grouping logic for legal document patterns (parties are usually introduced in the recitals)
- Do NOT use Cyera for this step — Cyera classifies, this step groups

### Component 2: Replacement Engine

**Purpose:** Apply the replacement policy to every classified element in the document.

**Processing order:**
1. Sort elements by `byte_offset_start` descending (replace from end of document backward to preserve offsets)
2. For each element, apply the replacement strategy from the policy table
3. Use the token assigned by the co-reference engine for DIRECT_IDENTIFIER replacements
4. For QUASI_IDENTIFIER generalization, use the generalization functions (see below)

**Generalization functions:**

```python
def generalize_address(address: str, jurisdiction: str) -> str:
    """
    Reduce to metro region.
    '142 Oak St, Sacramento, CA 95814' -> '[LOCATION_CA_METRO]'
    '89 Main St, Springfield, IL 62701' -> '[LOCATION_IL_METRO]'
    
    Use state code + METRO/RURAL based on population threshold.
    Cities over 100k population = METRO. Under = RURAL.
    """

def generalize_date(date_str: str) -> str:
    """
    Reduce to quarter.
    'March 15, 2026' -> '[DATE_2026_Q1]'
    'November 3, 2025' -> '[DATE_2025_Q4]'
    """

def generalize_dob(dob_str: str, reference_date: str) -> str:
    """
    Convert to age range (5-year buckets).
    '03/15/1988' (ref: 2026) -> '[AGE_RANGE_35_40]'
    """

def bucket_financial(value: float, category: str) -> str:
    """
    Bucket to range.
    Rent/income: $1000 buckets. $4200 -> '[RENT_4000_5000]'
    Purchase price: $50k buckets. $375000 -> '[PRICE_350K_400K]'
    Settlement: $25k buckets.
    """

def generalize_employer(employer: str, industry: str) -> str:
    """
    Replace with industry token.
    'Acme Corp' (tech) -> '[EMPLOYER_TECH]'
    Industry comes from Cyera context metadata.
    If industry unknown: '[EMPLOYER_1]'
    """
```

### Component 3: Relationship Graph Builder

**Purpose:** Capture structural relationships between tokens so the skeleton preserves legal meaning without identity.

**Output format:**
```json
{
  "relationships": [
    {
      "type": "COHABITATION",
      "entities": ["PARTY_A", "PARTY_B"],
      "location": "ADDRESS_1"
    },
    {
      "type": "PARENT_CHILD",
      "entity": "PARTY_A",
      "dependents": ["MINOR_CHILD_1", "MINOR_CHILD_2"]
    },
    {
      "type": "EMPLOYMENT",
      "entity": "PARTY_A",
      "employer": "EMPLOYER_1",
      "industry": "TECH"
    }
  ]
}
```

**How to detect relationships:**
- COHABITATION: Two PARTY tokens share the same ADDRESS element (same `byte_offset` range or adjacent)
- PARENT_CHILD: PARTY token and MINOR token appear in the same clause or within N sentences
- EMPLOYMENT: PARTY token and EMPLOYER token in same sentence or clause

### Component 4: Validation (Second-Pass NER)

**Purpose:** Catch anything the replacement engine missed.

**Implementation:**
- Run `spacy` NER (`en_core_web_lg`) on the fully replaced document text
- Flag any detected PERSON, ORG, GPE, or DATE entities that are NOT tokens (i.e., not wrapped in `[]`)
- Exceptions: known boilerplate entities (state names in governing law clauses, "United States", etc.) — maintain an allowlist
- If any non-allowed entity is detected: **fail the document** and log the missed element for upstream Cyera retraining

## Output Schema

### 1. De-identified Document Text
The full document text with all replacements applied. Ready for Stage 2 (structural extraction).

### 2. Token Mapping Table
```json
{
  "document_id": "uuid",
  "tokens": [
    {
      "token": "[PARTY_A]",
      "category": "DIRECT_IDENTIFIER",
      "element_ids": ["uuid-1", "uuid-7", "uuid-22"],
      "occurrence_count": 14,
      "data_subject_role": "PARTY_PRIMARY"
    },
    {
      "token": "[LOCATION_CA_METRO]",
      "category": "QUASI_IDENTIFIER",
      "element_ids": ["uuid-3"],
      "generalization_level": "STATE_METRO",
      "original_jurisdiction": "US_CA"
    }
  ]
}
```

**Note:** The token mapping table does NOT contain original values. It maps tokens to Cyera element IDs. The original values exist only in the tenant-isolated source document. This separation is intentional — the mapping table can be stored alongside the skeleton without re-introducing PII.

### 3. Relationship Graph
As defined above.

### 4. Validation Report
```json
{
  "document_id": "uuid",
  "validation_passed": true,
  "ner_entities_detected": 3,
  "ner_entities_allowed": 3,
  "ner_entities_flagged": 0,
  "elements_replaced": 47,
  "elements_passed_through": 17,
  "co_references_resolved": 8,
  "unique_tokens_assigned": 12
}
```

## Project Structure

```
stage1-entity-replacement/
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # Main orchestrator — runs all components in sequence
│   ├── coreference.py           # Co-reference resolution engine
│   ├── replacement.py           # Replacement engine with policy table
│   ├── generalization.py        # Generalization functions (address, date, financial, etc.)
│   ├── relationships.py         # Relationship graph builder
│   ├── validation.py            # Second-pass NER validation
│   └── schemas.py               # Pydantic models for input/output schemas
├── tests/
│   ├── test_coreference.py
│   ├── test_replacement.py
│   ├── test_generalization.py
│   ├── test_relationships.py
│   ├── test_validation.py
│   ├── test_pipeline_integration.py
│   └── fixtures/
│       ├── sample_lease.txt                # Sample lease agreement with fake PII
│       ├── sample_lease_metadata.json      # Simulated Cyera classification output
│       ├── sample_nda.txt
│       ├── sample_nda_metadata.json
│       ├── sample_custody.txt
│       └── sample_custody_metadata.json
├── config/
│   ├── replacement_policies.yaml           # Policy table (editable without code changes)
│   ├── generalization_rules.yaml           # Bucketing thresholds, metro population cutoffs
│   └── ner_allowlist.yaml                  # Known safe entities for validation pass
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python 3.11+**
- **spacy** (`en_core_web_lg`) — co-reference resolution assistance and validation NER
- **pydantic** — input/output schema validation
- **PyYAML** — configuration loading
- **pytest** — testing

Do NOT use:
- Any external classification API (Cyera handles this upstream)
- Any database (this stage is a stateless transformation — input in, output out)
- Any ML model for classification (that's Cyera's job)

## Test Strategy

### Unit Tests
- **Co-reference:** Given a list of elements with names "Sarah Mitchell", "Ms. Mitchell", "Sarah" all tagged as PARTY_PRIMARY, verify they resolve to the same token
- **Replacement:** Given a document string and metadata, verify all elements are replaced correctly and byte offsets are handled (replacing from end backward)
- **Generalization:** Test each function independently: address → region, date → quarter, DOB → age range, financial → bucket
- **Relationships:** Given tokens with shared addresses, verify COHABITATION relationship is detected
- **Validation:** Given a fully replaced document, verify NER finds no un-replaced entities. Given a document with a deliberately missed name, verify validation catches it

### Integration Test
- Load `sample_lease.txt` and `sample_lease_metadata.json`
- Run full pipeline
- Assert: zero flagged entities in validation
- Assert: all CRITICAL and HIGH elements replaced
- Assert: BOILERPLATE text unchanged
- Assert: token mapping table contains no original values
- Assert: relationship graph is non-empty for documents with multiple parties

### Edge Cases to Test
- Document where the same name appears 50+ times (performance + consistency)
- Document with minors' information (verify complete stripping, no partial leakage)
- Document with health/disability information embedded in a clause (verify full clause redaction, not just the keyword)
- Document with no PII (all boilerplate) — should pass through with empty token mapping
- Document where Cyera flagged a privilege indicator — pipeline should halt and return error

## What Success Looks Like

A document enters this stage with names, addresses, social security numbers, income figures, children's details, and medical information. It exits with none of those things. What remains is the legal language, the clause structure, and typed tokens that preserve relationships without identity.

The de-identified output should be readable as a legal document. You should be able to read the transformed lease and understand its structure, its terms, its modifications from template — without knowing who signed it, where they live, what they earn, or who their children are.

If you can read the output and reconstruct any real person's identity, the stage has failed.
