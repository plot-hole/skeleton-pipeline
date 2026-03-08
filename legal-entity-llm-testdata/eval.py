"""
eval.py
=======
Batch evaluation script for the legal entity de-identification pipeline.

Compares pipeline predictions against ground truth annotations and computes
precision, recall, and F1 scores — overall and per entity type.

Usage:
    python eval.py --gold synthetic_legal_data/ --pred predictions/

    # Evaluate a single document:
    python eval.py --gold synthetic_legal_data/ --pred predictions/ --doc civil_complaint_rural

    # Evaluate only cross-document consistency:
    python eval.py --gold synthetic_legal_data/ --pred predictions/ --corpus martinez_v_greenfield

    # Evaluate only adversarial documents:
    python eval.py --gold synthetic_legal_data/ --pred predictions/ --adversarial

    # Output as JSON:
    python eval.py --gold synthetic_legal_data/ --pred predictions/ --json

Prediction files should be named <scenario_id>_pred.json and contain:
{
    "entities": [
        {
            "text": "exact text found",
            "entity_type": "PERSON | ORG | ADDRESS | DATE | MONEY | ID",
            "token_prefix": "PLAINTIFF | DEFENDANT | ..."
        }
    ]
}
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_index(gold_dir: Path) -> dict:
    """Load the index.json from the gold directory."""
    index_path = gold_dir / "index.json"
    if not index_path.exists():
        print(f"ERROR: {index_path} not found.")
        sys.exit(1)
    return json.loads(index_path.read_text(encoding="utf-8"))


def load_ground_truth(gold_dir: Path, scenario_id: str) -> dict:
    """Load ground truth for a scenario."""
    gt_path = gold_dir / f"{scenario_id}_ground_truth.json"
    if not gt_path.exists():
        return None
    return json.loads(gt_path.read_text(encoding="utf-8"))


def load_predictions(pred_dir: Path, scenario_id: str) -> dict | None:
    """Load predictions for a scenario."""
    pred_path = pred_dir / f"{scenario_id}_pred.json"
    if not pred_path.exists():
        return None
    return json.loads(pred_path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    """Normalize entity text for comparison (lowercase, strip, collapse whitespace)."""
    return " ".join(text.lower().split())


def make_entity_key(entity: dict) -> tuple:
    """Create a hashable key for an entity mention (text + entity_type)."""
    return (normalize_text(entity["text"]), entity.get("entity_type", "UNKNOWN"))


def make_entity_key_text_only(entity: dict) -> str:
    """Create a hashable key using only normalized text (for relaxed matching)."""
    return normalize_text(entity["text"])


def compute_metrics(gold_entities: list[dict], pred_entities: list[dict],
                    strict: bool = True) -> dict:
    """
    Compute precision, recall, F1 for a single document.

    Args:
        gold_entities: Ground truth entity mentions.
        pred_entities: Predicted entity mentions.
        strict: If True, match on (text, entity_type). If False, match on text only.

    Returns:
        Dict with tp, fp, fn, precision, recall, f1.
    """
    if strict:
        key_fn = make_entity_key
    else:
        key_fn = lambda e: (make_entity_key_text_only(e),)

    # Build multisets (count duplicates)
    gold_counts = defaultdict(int)
    for e in gold_entities:
        gold_counts[key_fn(e)] += 1

    pred_counts = defaultdict(int)
    for e in pred_entities:
        pred_counts[key_fn(e)] += 1

    # True positives = min of gold and pred counts for each key
    tp = 0
    for key in gold_counts:
        tp += min(gold_counts[key], pred_counts.get(key, 0))

    fp = sum(pred_counts.values()) - tp
    fn = sum(gold_counts.values()) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_per_type_metrics(gold_entities: list[dict], pred_entities: list[dict]) -> dict:
    """Compute metrics broken down by entity_type."""
    all_types = set()
    for e in gold_entities:
        all_types.add(e.get("entity_type", "UNKNOWN"))
    for e in pred_entities:
        all_types.add(e.get("entity_type", "UNKNOWN"))

    per_type = {}
    for etype in sorted(all_types):
        gold_of_type = [e for e in gold_entities if e.get("entity_type", "UNKNOWN") == etype]
        pred_of_type = [e for e in pred_entities if e.get("entity_type", "UNKNOWN") == etype]
        per_type[etype] = compute_metrics(gold_of_type, pred_of_type)

    return per_type


def check_cross_doc_consistency(gold_dir: Path, pred_dir: Path,
                                corpus_id: str, doc_ids: list[str]) -> dict:
    """
    Check whether predictions reuse the same tokens for shared entities
    across documents in a corpus.

    Returns a report of consistency issues.
    """
    # Collect canonical -> token_prefix mappings from predictions per doc
    canonical_tokens = defaultdict(dict)  # canonical -> {doc_id: token_prefix}

    for doc_id in doc_ids:
        pred = load_predictions(pred_dir, doc_id)
        if not pred:
            continue
        for entity in pred.get("entities", []):
            canonical = entity.get("canonical", entity["text"])
            token = entity.get("token_prefix", "")
            if canonical and token:
                canonical_tokens[normalize_text(canonical)][doc_id] = token

    inconsistencies = []
    for canonical, doc_tokens in canonical_tokens.items():
        unique_tokens = set(doc_tokens.values())
        if len(unique_tokens) > 1:
            inconsistencies.append({
                "canonical": canonical,
                "tokens_by_doc": dict(doc_tokens),
                "issue": f"Inconsistent token assignment: {unique_tokens}",
            })

    return {
        "corpus_id": corpus_id,
        "documents": doc_ids,
        "shared_entities_checked": len(canonical_tokens),
        "inconsistencies": inconsistencies,
        "consistent": len(inconsistencies) == 0,
    }


def check_adversarial(gold_dir: Path, pred_dir: Path, scenario_id: str) -> dict:
    """
    For adversarial documents, check if the system correctly avoided
    tagging negative examples.

    Returns a report of false positive issues.
    """
    gt = load_ground_truth(gold_dir, scenario_id)
    pred = load_predictions(pred_dir, scenario_id)

    if not gt or not pred:
        return {"scenario_id": scenario_id, "status": "missing_data"}

    negative_examples = gt.get("negative_examples", [])
    pred_texts = {normalize_text(e["text"]) for e in pred.get("entities", [])}

    false_positives = []
    for neg in negative_examples:
        neg_text = normalize_text(neg["text"])
        if neg_text in pred_texts:
            false_positives.append({
                "text": neg["text"],
                "reason": neg.get("reason", "should not be tagged"),
            })

    return {
        "scenario_id": scenario_id,
        "negative_examples_checked": len(negative_examples),
        "false_positives": false_positives,
        "false_positive_count": len(false_positives),
        "passed": len(false_positives) == 0,
    }


def format_table(headers: list[str], rows: list[list], col_widths: list[int] | None = None) -> str:
    """Format a simple text table."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                      for i, h in enumerate(headers)]

    lines = []
    header_line = "  ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("  ".join("-" * w for w in col_widths))
    for row in rows:
        line = "  ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate legal entity de-identification predictions against ground truth."
    )
    parser.add_argument("--gold", required=True, help="Directory containing ground truth files")
    parser.add_argument("--pred", required=True, help="Directory containing prediction files")
    parser.add_argument("--doc", help="Evaluate a single document by scenario_id")
    parser.add_argument("--corpus", help="Evaluate only documents in a specific corpus")
    parser.add_argument("--adversarial", action="store_true", help="Evaluate only adversarial documents")
    parser.add_argument("--relaxed", action="store_true",
                        help="Use relaxed matching (text only, ignore entity_type)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    gold_dir = Path(args.gold)
    pred_dir = Path(args.pred)

    if not gold_dir.exists():
        print(f"ERROR: Gold directory {gold_dir} does not exist.")
        sys.exit(1)
    if not pred_dir.exists():
        print(f"ERROR: Prediction directory {pred_dir} does not exist.")
        sys.exit(1)

    index = load_index(gold_dir)
    documents = index.get("documents", [])

    # Filter documents based on CLI flags
    if args.doc:
        documents = [d for d in documents if d["id"] == args.doc]
        if not documents:
            print(f"ERROR: Document '{args.doc}' not found in index.")
            sys.exit(1)
    elif args.corpus:
        documents = [d for d in documents if d.get("corpus_id") == args.corpus]
        if not documents:
            print(f"ERROR: No documents found for corpus '{args.corpus}'.")
            sys.exit(1)
    elif args.adversarial:
        documents = [d for d in documents if d.get("adversarial")]
        if not documents:
            print(f"ERROR: No adversarial documents found.")
            sys.exit(1)

    # ── Per-document evaluation ────────────────────────────────────────────
    all_results = []
    aggregate_tp = 0
    aggregate_fp = 0
    aggregate_fn = 0
    per_type_aggregate = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    missing_preds = []

    for doc in documents:
        scenario_id = doc["id"]
        gt = load_ground_truth(gold_dir, scenario_id)
        pred = load_predictions(pred_dir, scenario_id)

        if gt is None:
            continue
        if pred is None:
            missing_preds.append(scenario_id)
            continue

        gold_entities = gt.get("entities", [])
        pred_entities = pred.get("entities", [])

        metrics = compute_metrics(gold_entities, pred_entities, strict=not args.relaxed)
        per_type = compute_per_type_metrics(gold_entities, pred_entities)

        result = {
            "scenario_id": scenario_id,
            "doc_type": doc["doc_type"],
            "gold_count": len(gold_entities),
            "pred_count": len(pred_entities),
            **metrics,
            "per_type": per_type,
        }

        if doc.get("corpus_id"):
            result["corpus_id"] = doc["corpus_id"]
        if doc.get("adversarial"):
            result["adversarial"] = True

        all_results.append(result)

        aggregate_tp += metrics["tp"]
        aggregate_fp += metrics["fp"]
        aggregate_fn += metrics["fn"]

        for etype, emetrics in per_type.items():
            per_type_aggregate[etype]["tp"] += emetrics["tp"]
            per_type_aggregate[etype]["fp"] += emetrics["fp"]
            per_type_aggregate[etype]["fn"] += emetrics["fn"]

    # ── Aggregate metrics ─────────────────────────────────────────────────
    agg_precision = aggregate_tp / (aggregate_tp + aggregate_fp) if (aggregate_tp + aggregate_fp) > 0 else 0.0
    agg_recall = aggregate_tp / (aggregate_tp + aggregate_fn) if (aggregate_tp + aggregate_fn) > 0 else 0.0
    agg_f1 = (2 * agg_precision * agg_recall / (agg_precision + agg_recall)) if (agg_precision + agg_recall) > 0 else 0.0

    aggregate = {
        "documents_evaluated": len(all_results),
        "documents_missing_predictions": missing_preds,
        "tp": aggregate_tp,
        "fp": aggregate_fp,
        "fn": aggregate_fn,
        "precision": round(agg_precision, 4),
        "recall": round(agg_recall, 4),
        "f1": round(agg_f1, 4),
    }

    # Per-type aggregate
    per_type_final = {}
    for etype in sorted(per_type_aggregate):
        t = per_type_aggregate[etype]
        p = t["tp"] / (t["tp"] + t["fp"]) if (t["tp"] + t["fp"]) > 0 else 0.0
        r = t["tp"] / (t["tp"] + t["fn"]) if (t["tp"] + t["fn"]) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_type_final[etype] = {
            "tp": t["tp"], "fp": t["fp"], "fn": t["fn"],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
        }

    # ── Cross-document consistency checks ─────────────────────────────────
    crossdoc_results = []
    corpora = index.get("corpora") or {}
    if not args.doc:  # Skip cross-doc for single-doc eval
        for corpus_id, doc_ids in corpora.items():
            if args.corpus and corpus_id != args.corpus:
                continue
            crossdoc_results.append(
                check_cross_doc_consistency(gold_dir, pred_dir, corpus_id, doc_ids)
            )

    # ── Adversarial checks ────────────────────────────────────────────────
    adversarial_results = []
    adv_docs = [d for d in documents if d.get("adversarial")]
    for doc in adv_docs:
        adversarial_results.append(check_adversarial(gold_dir, pred_dir, doc["id"]))

    # ── Output ────────────────────────────────────────────────────────────
    full_report = {
        "aggregate": aggregate,
        "per_entity_type": per_type_final,
        "per_document": all_results,
        "cross_document_consistency": crossdoc_results if crossdoc_results else None,
        "adversarial_checks": adversarial_results if adversarial_results else None,
    }

    if args.json:
        print(json.dumps(full_report, indent=2))
        return

    # Human-readable output
    print("\n" + "=" * 72)
    print("  LEGAL ENTITY DE-IDENTIFICATION EVALUATION REPORT")
    print("=" * 72)

    if missing_preds:
        print(f"\n  WARNING: Missing predictions for {len(missing_preds)} document(s):")
        for mp in missing_preds:
            print(f"    - {mp}")

    # Overall metrics
    print(f"\n  OVERALL METRICS ({aggregate['documents_evaluated']} documents)")
    print(f"  {'─' * 40}")
    print(f"  Precision:  {aggregate['precision']:.4f}")
    print(f"  Recall:     {aggregate['recall']:.4f}")
    print(f"  F1 Score:   {aggregate['f1']:.4f}")
    print(f"  TP: {aggregate['tp']}  FP: {aggregate['fp']}  FN: {aggregate['fn']}")

    # Per entity type
    if per_type_final:
        print(f"\n  PER ENTITY TYPE")
        print(f"  {'─' * 40}")
        type_headers = ["Type", "TP", "FP", "FN", "Prec", "Recall", "F1"]
        type_rows = []
        for etype, m in per_type_final.items():
            type_rows.append([etype, m["tp"], m["fp"], m["fn"],
                              f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}"])
        print("  " + format_table(type_headers, type_rows).replace("\n", "\n  "))

    # Per document
    if all_results:
        print(f"\n  PER DOCUMENT")
        print(f"  {'─' * 40}")
        doc_headers = ["Document", "Type", "Gold", "Pred", "TP", "FP", "FN", "F1"]
        doc_rows = []
        for r in all_results:
            sid = r["scenario_id"]
            if len(sid) > 30:
                sid = sid[:27] + "..."
            tag = ""
            if r.get("adversarial"):
                tag = " *"
            if r.get("corpus_id"):
                tag = " ^"
            doc_rows.append([
                sid + tag,
                r["doc_type"][:15],
                r["gold_count"],
                r["pred_count"],
                r["tp"], r["fp"], r["fn"],
                f"{r['f1']:.4f}",
            ])
        print("  " + format_table(doc_headers, doc_rows).replace("\n", "\n  "))
        print("  (* = adversarial, ^ = cross-doc corpus)")

    # Cross-document consistency
    if crossdoc_results:
        print(f"\n  CROSS-DOCUMENT CONSISTENCY")
        print(f"  {'─' * 40}")
        for cr in crossdoc_results:
            status = "PASS" if cr["consistent"] else "FAIL"
            print(f"  Corpus: {cr['corpus_id']} — {status}")
            print(f"    Documents: {', '.join(cr['documents'])}")
            print(f"    Shared entities checked: {cr['shared_entities_checked']}")
            if cr["inconsistencies"]:
                print(f"    Inconsistencies ({len(cr['inconsistencies'])}):")
                for inc in cr["inconsistencies"]:
                    print(f"      - \"{inc['canonical']}\": {inc['issue']}")

    # Adversarial checks
    if adversarial_results:
        print(f"\n  ADVERSARIAL CHECKS")
        print(f"  {'─' * 40}")
        for ar in adversarial_results:
            if ar.get("status") == "missing_data":
                print(f"  {ar['scenario_id']}: SKIPPED (missing data)")
                continue
            status = "PASS" if ar["passed"] else "FAIL"
            print(f"  {ar['scenario_id']}: {status}")
            print(f"    Negative examples checked: {ar['negative_examples_checked']}")
            if ar["false_positives"]:
                print(f"    False positives ({ar['false_positive_count']}):")
                for fp in ar["false_positives"]:
                    print(f"      - \"{fp['text']}\": {fp['reason']}")

    print("\n" + "=" * 72 + "\n")


if __name__ == "__main__":
    main()
