"""Run the Stage 1 pipeline on a document and print the de-identified output.

Usage:
    python run_pipeline.py                                          # runs on sample_lease fixture
    python run_pipeline.py <document.txt> <metadata.json>           # runs on custom files
    python run_pipeline.py <document.txt>                           # auto-generates metadata with spaCy
    python run_pipeline.py --all                                    # runs on all three fixtures
    python run_pipeline.py --batch input/                           # runs on all .txt files in a directory
"""

from __future__ import annotations

import json
import sys
import os
from datetime import date
from pathlib import Path

# Fix Windows console encoding for Unicode output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import Pipeline
from src.schemas import ClassificationMetadata, PrivilegeHaltError


FIXTURES = Path(__file__).parent / "tests" / "fixtures"
OUTPUT_DIR = Path(__file__).parent / "output"

SEPARATOR = "=" * 80
THIN_SEP = "-" * 80


def run_one(doc_path: str, meta_path: str, save_output: bool = False) -> bool:
    """Run the pipeline on a single document and print results.

    Returns True if validation passed, False otherwise.
    """
    with open(doc_path, encoding="utf-8") as f:
        doc_text = f.read()
    with open(meta_path, encoding="utf-8") as f:
        metadata = ClassificationMetadata(**json.load(f))

    doc_name = os.path.basename(doc_path)
    print(SEPARATOR)
    print(f"  DOCUMENT: {doc_name}")
    print(f"  ID:       {metadata.document_id}")
    print(f"  Elements: {metadata.summary.total_elements}")
    print(SEPARATOR)

    pipeline = Pipeline(reference_date=date(2026, 3, 7))

    try:
        output = pipeline.run(doc_text, metadata)
    except PrivilegeHaltError as e:
        print(f"\n  *** PIPELINE HALTED: {e} ***\n")
        return False

    # -- De-identified text --
    print("\n  DE-IDENTIFIED DOCUMENT")
    print(THIN_SEP)
    # Handle Unicode chars that Windows console can't display
    try:
        print(output.deidentified_text)
    except UnicodeEncodeError:
        print(output.deidentified_text.encode("utf-8", errors="replace").decode("utf-8"))
    print(THIN_SEP)

    # -- Token mapping --
    print("\n  TOKEN MAPPING TABLE")
    print(THIN_SEP)
    print(f"  {'Token':<30} {'Category':<22} {'Occurrences':<12} {'Role'}")
    print(f"  {'-----':<30} {'--------':<22} {'-----------':<12} {'----'}")
    for t in output.token_mapping.tokens:
        print(f"  {t.token:<30} {t.category:<22} {t.occurrence_count:<12} {t.data_subject_role or ''}")
    print()

    # -- Relationship graph --
    print("  RELATIONSHIP GRAPH")
    print(THIN_SEP)
    if not output.relationship_graph.relationships:
        print("  (no relationships detected)")
    for rel in output.relationship_graph.relationships:
        if rel.type == "COHABITATION":
            print(f"  COHABITATION: {', '.join(rel.entities)} at {rel.location}")
        elif rel.type == "PARENT_CHILD":
            print(f"  PARENT_CHILD: {rel.entity} -> {', '.join(rel.dependents)}")
        elif rel.type == "EMPLOYMENT":
            print(f"  EMPLOYMENT:   {rel.entity} works at {rel.employer} ({rel.industry})")
    print()

    # -- Validation report --
    report = output.validation_report
    status = "PASSED" if report.validation_passed else "FAILED"
    print("  VALIDATION REPORT")
    print(THIN_SEP)
    print(f"  Status:               {status}")
    print(f"  NER entities detected: {report.ner_entities_detected}")
    print(f"  NER entities allowed:  {report.ner_entities_allowed}")
    print(f"  NER entities flagged:  {report.ner_entities_flagged}")
    print(f"  Elements replaced:     {report.elements_replaced}")
    print(f"  Elements passed thru:  {report.elements_passed_through}")
    print(f"  Co-refs resolved:      {report.co_references_resolved}")
    print(f"  Unique tokens:         {report.unique_tokens_assigned}")

    # Show flagged entities if validation failed
    if not report.validation_passed:
        print(f"\n  FLAGGED ENTITIES:")
        # Access the raw validation result from pipeline
        # We'll retrieve from the JSON output
        json_output = json.loads(output.model_dump_json())
        flagged = json_output.get("validation_report", {}).get("flagged_entities", [])
        if flagged:
            for f_ent in flagged:
                print(f"    - \"{f_ent.get('text', 'N/A')}\" ({f_ent.get('label', 'N/A')})")
    print()

    # -- Save output files if requested --
    if save_output:
        OUTPUT_DIR.mkdir(exist_ok=True)
        stem = Path(doc_path).stem

        # Save de-identified text
        deident_path = OUTPUT_DIR / f"{stem}_deidentified.txt"
        deident_path.write_text(output.deidentified_text, encoding="utf-8")

        # Save full JSON
        json_path = OUTPUT_DIR / f"{stem}_output.json"
        json_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")

        print(f"  OUTPUT SAVED:")
        print(f"    De-identified: {deident_path}")
        print(f"    Full JSON:     {json_path}")
        print()

    return report.validation_passed


def run_batch(directory: str) -> None:
    """Run the pipeline on all .txt files in a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    txt_files = sorted(dir_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {directory}")
        sys.exit(1)

    # Clear any previous metadata files
    print(f"Found {len(txt_files)} documents in {directory}\n")

    from generate_metadata import generate_metadata

    results: list[tuple[str, bool, int]] = []

    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n{'#' * 80}")
        print(f"  [{i}/{len(txt_files)}] Processing: {txt_file.name}")
        print(f"{'#' * 80}\n")

        # Generate metadata
        text = txt_file.read_text(encoding="utf-8")
        metadata = generate_metadata(text)
        meta_path = txt_file.with_name(txt_file.stem + "_metadata.json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        n_elements = len(metadata["elements"])
        print(f"  Generated {n_elements} elements -> {meta_path.name}\n")

        # Run pipeline
        passed = run_one(str(txt_file), str(meta_path), save_output=True)
        results.append((txt_file.name, passed, n_elements))

    # Print summary table
    print("\n" + "=" * 80)
    print("  BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"  {'Document':<45} {'Elements':<10} {'Validation'}")
    print(f"  {'--------':<45} {'--------':<10} {'----------'}")
    passed_count = 0
    for name, passed, n_elem in results:
        status = "PASSED" if passed else "FAILED"
        icon = "+" if passed else "X"
        print(f"  [{icon}] {name:<42} {n_elem:<10} {status}")
        if passed:
            passed_count += 1
    print(f"\n  Total: {len(results)} documents | Passed: {passed_count} | Failed: {len(results) - passed_count}")
    print(f"  Output saved to: {OUTPUT_DIR.resolve()}")
    print()


def main() -> None:
    args = sys.argv[1:]

    if not args:
        # Default: run on sample lease
        run_one(
            str(FIXTURES / "sample_lease.txt"),
            str(FIXTURES / "sample_lease_metadata.json"),
        )
    elif args == ["--all"]:
        for name in ["sample_lease", "sample_nda", "sample_custody"]:
            run_one(
                str(FIXTURES / f"{name}.txt"),
                str(FIXTURES / f"{name}_metadata.json"),
            )
    elif len(args) == 2 and args[0] == "--batch":
        run_batch(args[1])
    elif len(args) == 2:
        run_one(args[0], args[1])
    elif len(args) == 1 and args[0].endswith(".txt"):
        # Auto-generate metadata from plain text
        doc_path = Path(args[0])
        if not doc_path.exists():
            print(f"Error: {doc_path} not found")
            sys.exit(1)
        meta_path = doc_path.with_name(doc_path.stem + "_metadata.json")
        print(f"No metadata file provided. Generating with spaCy...\n")
        from generate_metadata import generate_metadata
        text = doc_path.read_text(encoding="utf-8")
        metadata = generate_metadata(text)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Wrote {meta_path.name} ({len(metadata['elements'])} elements)\n")
        run_one(str(doc_path), str(meta_path), save_output=True)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
