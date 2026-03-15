# Skeleton Pipeline

A privacy engineering pipeline that converts sensitive legal documents into de-identified structural skeletons. The skeletons preserve legal meaning and clause structure while stripping all personally identifiable information, enabling safe use in shared AI features (recommendations, analytics, model improvement).

## How It Works

The pipeline receives documents that have already been scanned and classified by [Cyera's DSPM platform](https://www.cyera.io/). It takes that classification output and transforms the document text by:

1. **Resolving co-references** — grouping all mentions of the same entity ("Sarah Mitchell", "Ms. Mitchell", "Sarah") into a single token (`[PARTY_A]`)
2. **Replacing sensitive elements** — applying category-specific strategies (tokenization, generalization, redaction, bucketing)
3. **Building a relationship graph** — capturing structural relationships (cohabitation, parent-child, employment) between tokens
4. **Validating** — running a second-pass NER scan to catch anything the replacement engine missed

## Stages

| Stage | Description | Status |
|-------|-------------|--------|
| **Stage 1** | Entity Resolution & Replacement | Built |
| Stage 2 | Structural Extraction | Planned |

## Project Structure

```
skeleton-pipeline/
├── stage1-entity-replacement/   # Entity resolution & replacement pipeline
│   ├── src/                     # Pipeline source code
│   ├── tests/                   # Unit and integration tests
│   ├── config/                  # Replacement policies, generalization rules, NER allowlist
│   ├── input/                   # Sample legal documents with simulated Cyera metadata
│   ├── run_pipeline.py          # CLI runner
│   └── generate_metadata.py     # Generates simulated Cyera metadata using spaCy
├── stage1-entity-replacement.md # Stage 1 design specification
└── .gitignore
```

## Quick Start

```bash
cd stage1-entity-replacement

# Install core dependencies
pip install -r requirements.txt

# Install NLP dependencies (needed for validation and metadata generation)
pip install -r requirements-nlp.txt
python -m spacy download en_core_web_lg

# Run on sample fixture
python run_pipeline.py

# Run on all fixtures
python run_pipeline.py --all

# Run on a custom document (auto-generates metadata with spaCy)
python run_pipeline.py path/to/document.txt

# Batch process a directory
python run_pipeline.py --batch input/

# Run tests
pytest tests/
```

## Tech Stack

- Python 3.11+
- spaCy (`en_core_web_lg`) — co-reference assistance and validation NER
- Pydantic — schema validation
- PyYAML — configuration
- pytest — testing
