# Stage 1: Entity Resolution & Replacement

Consumes Cyera classification metadata and transforms legal document text into de-identified output. See the [design spec](../stage1-entity-replacement.md) for full details.

## Components

| Module | Purpose |
|--------|---------|
| `coreference.py` | Groups all mentions of the same entity into a single token |
| `replacement.py` | Applies replacement policies (tokenize, generalize, redact, bucket) |
| `generalization.py` | Address → region, date → quarter, DOB → age range, financial → bucket |
| `relationships.py` | Detects cohabitation, parent-child, and employment relationships |
| `validation.py` | Second-pass spaCy NER scan to catch missed entities |
| `schemas.py` | Pydantic models for input/output |
| `pipeline.py` | Orchestrates all components in sequence |

## Replacement Policy

| Category | Strategy | Example |
|----------|----------|---------|
| Direct identifier (name) | Token replace | Sarah Mitchell → `[PARTY_A]` |
| Direct identifier (SSN/ID) | Strip entirely | 123-45-6789 → `[SSN_REDACTED]` |
| Quasi-identifier (address) | Generalize to region | Sacramento, CA → `[LOCATION_CA_METRO]` |
| Quasi-identifier (DOB) | Generalize to age range | 03/15/1988 → `[AGE_RANGE_35_40]` |
| Financial data | Bucket to range | $4,200/month → `[RENT_4000_5000]` |
| Health protected | Strip entirely | `[PROTECTED_STATUS_REDACTED]` |
| Minor information | Structural token | `[MINOR_CHILD_1]` |
| Privilege indicator | Halt pipeline | Document should not have reached this stage |
| Boilerplate | Pass through | Unchanged |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-nlp.txt
python -m spacy download en_core_web_lg

# Run pipeline
python run_pipeline.py                              # sample lease fixture
python run_pipeline.py doc.txt metadata.json        # custom files
python run_pipeline.py doc.txt                      # auto-generate metadata
python run_pipeline.py --batch input/               # batch directory

# Run tests
pytest tests/
```

## Output

The pipeline produces:
1. **De-identified document text** — all PII replaced with typed tokens
2. **Token mapping table** — maps tokens to Cyera element IDs (no original values)
3. **Relationship graph** — structural relationships between tokens
4. **Validation report** — NER scan results confirming no leaked entities
