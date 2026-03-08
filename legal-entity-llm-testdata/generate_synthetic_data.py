"""
generate_synthetic_data.py
==========================
Generates synthetic legal documents for testing the de-identification pipeline.
Each document comes with a ground truth manifest of all PII entities.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."   # or: set ANTHROPIC_API_KEY=sk-ant-... (Windows)
    python generate_synthetic_data.py

Output:
    synthetic_legal_data/
        civil_complaint_rural.txt
        civil_complaint_rural_ground_truth.json
        deposition_transcript.txt
        deposition_transcript_ground_truth.json
        settlement_agreement.txt
        settlement_agreement_ground_truth.json
        criminal_complaint.txt
        criminal_complaint_ground_truth.json
        expert_witness_report.txt
        expert_witness_report_ground_truth.json
        index.json
"""

import anthropic
import json
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("synthetic_legal_data")
MODEL = "claude-sonnet-4-20250514"

# ── Document scenarios ────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id": "civil_complaint_rural",
        "doc_type": "CIVIL COMPLAINT",
        "description": """
            Personal injury civil complaint. Plaintiff is a rural Ohio farmer (give him a full name,
            SSN, address on a rural route, phone, email). He was injured when a freight truck crossed
            into his lane. Defendants: the trucking company (corporation) and the truck driver (individual,
            full name, SSN, address). Two plaintiff attorneys at a named law firm. One defense attorney
            at a different named firm. One expert witness (orthopedic surgeon, Dr., named, with hospital
            affiliation). Include: case number, incident date (March 2023), filing date, medical costs,
            damages amount, insurance company name. The plaintiff is referenced throughout as both his
            full name, last name only, "the plaintiff", and "he/his/him".
        """,
    },
    {
        "id": "deposition_transcript",
        "doc_type": "DEPOSITION TRANSCRIPT",
        "description": """
            Deposition transcript of a fact witness in a contract dispute. The witness (full name,
            urban Chicago address, phone, email) is being questioned by two attorneys — one for
            the plaintiff corporation and one for the defendant corporation. A judge's name appears
            in the caption. The witness refers to the plaintiff's CEO (named) and the defendant's
            COO (named) extensively, sometimes by name, sometimes as "he", "she", "the CEO", "her boss".
            Include: case number, contract execution date, breach date, two exhibit references
            (Exhibit A, Exhibit C-2), a dollar figure for the disputed contract value. Deposition
            date in the header. Heavy use of co-references and pronouns throughout.
        """,
    },
    {
        "id": "settlement_agreement",
        "doc_type": "SETTLEMENT AGREEMENT",
        "description": """
            Settlement agreement between two corporations in a product liability case. Plaintiff
            is an individual (full name, SSN, suburban New Jersey address, phone, email) who was
            injured by the defendant corporation's product. Defendant corporation has a named
            General Counsel (full name, email, direct phone). Plaintiff is represented by a named
            attorney at a named law firm. A mediator is named. Include: case number, large
            settlement dollar amount ($1.2M+), payment schedule dates, confidentiality clause,
            EIN for defendant corporation, specific release date. Plaintiff referred to by name,
            "the Claimant", "he/his", and "Mr. [last name]" throughout.
        """,
    },
    {
        "id": "criminal_complaint",
        "doc_type": "CRIMINAL COMPLAINT",
        "description": """
            Criminal complaint filed by a district attorney against a defendant accused of
            wire fraud. Defendant (full name, SSN, urban Detroit address, phone) ran a fraudulent
            investment scheme. Named victim (full name, suburban Michigan address, phone, email)
            lost significant money. A minor witness (age 15) is referenced carefully as "the minor
            witness" with initials only (e.g. "J.T.M."). An investigating detective (full name,
            badge number as ID) and the DA (full name, office address) are named. Include: case
            number, two federal statute citations (18 U.S.C.), incident date range, arrest date,
            filing date, dollar amounts defrauded, one email address used in the scheme.
        """,
    },
    {
        "id": "expert_witness_report",
        "doc_type": "EXPERT WITNESS REPORT",
        "description": """
            Medical expert witness report in a surgical malpractice case. The expert is a named
            cardiologist (Dr., full name, email, named hospital affiliation, suburban Texas address
            for the hospital). The patient/plaintiff (full name, DOB, SSN, medical record number,
            home phone) underwent a procedure performed by the defendant surgeon (full name, MD).
            Expert reviews medical records from two named dates, references the treating facility
            (named hospital, urban Texas address). Include: case number, plaintiff's birth date,
            surgery date as incident date, report date, medical cost amounts, one drug name,
            one exhibit reference. Plaintiff referred to as full name, "the patient", "she/her",
            and "Ms. [last name]" throughout.
        """,
    },
]

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are generating synthetic legal documents for software testing.
All names, addresses, SSNs, case numbers, phone numbers, and other identifiers are
completely fictional and randomly invented. Do not use real people or real cases.

You write with realistic legal language and formatting appropriate to each document type."""


def build_prompt(scenario: dict) -> str:
    return f"""Generate a realistic synthetic {scenario["doc_type"]} for testing a PII de-identification system.

SCENARIO REQUIREMENTS:
{scenario["description"].strip()}

FORMATTING REQUIREMENTS:
- 700–1000 words of realistic legal document text
- Use real-sounding but entirely fictional names, addresses, case numbers
- Vary how each person is referenced: full name on first mention, then mix of last name only,
  role title ("the plaintiff", "the witness"), and pronouns (he/she/they/his/her)
- Make addresses fully specific: street number, street name, city, state, ZIP
- Use realistic formats: SSN as XXX-XX-XXXX, phone as (XXX) XXX-XXXX, case numbers as YYYY-CV-NNNNN

After the document text, output EXACTLY this line by itself:
---GROUND_TRUTH---

Then output a JSON array tagging EVERY PII entity that appears in the document.
Include all surface forms (each alias, pronoun, and role reference as a separate entry pointing to the same token_prefix).

Use this exact structure:
[
  {{
    "text": "exact text as it appears in the document",
    "token_prefix": "e.g. PLAINTIFF, DEFENDANT, RURAL_OHIO_ADDRESS, URBAN_MICHIGAN_ADDRESS, INCIDENT_DATE, SSN, PHONE, EMAIL, CASE_NUMBER, LAW_FIRM, EXPERT_WITNESS, SETTLEMENT_AMOUNT, STATUTE, EXHIBIT, etc.",
    "entity_type": "PERSON | ORG | ADDRESS | DATE | MONEY | ID",
    "canonical": "the primary/full form this entity is an alias of (or same as text if this IS the canonical form)",
    "notes": "one short phrase explaining the classification"
  }}
]

Tag every single instance — if a name appears 6 times, include all 6. If "he" refers to the plaintiff 8 times, include all 8."""


# ── Generation ────────────────────────────────────────────────────────────────

def generate_document(client: anthropic.Anthropic, scenario: dict) -> tuple[str, list[dict]]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=16384,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(scenario)}],
    )

    full = response.content[0].text

    if "---GROUND_TRUTH---" not in full:
        raise ValueError("Ground truth separator missing from response")

    doc_text, gt_raw = full.split("---GROUND_TRUTH---", 1)
    doc_text = doc_text.strip()
    gt_raw = gt_raw.strip()

    # Strip markdown code fences if present (handles ```json, ``` json, etc.)
    lines = gt_raw.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    gt_raw = "\n".join(lines)

    ground_truth = json.loads(gt_raw.strip())
    return doc_text, ground_truth


def save(scenario: dict, doc_text: str, ground_truth: list[dict], output_dir: Path):
    doc_path = output_dir / f"{scenario['id']}.txt"
    gt_path = output_dir / f"{scenario['id']}_ground_truth.json"

    doc_path.write_text(doc_text, encoding="utf-8")

    manifest = {
        "scenario_id": scenario["id"],
        "doc_type": scenario["doc_type"],
        "word_count": len(doc_text.split()),
        "entity_count": len(ground_truth),
        "unique_canonical_count": len({e.get("canonical", e["text"]) for e in ground_truth}),
        "entities": ground_truth,
    }
    gt_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    unique = manifest["unique_canonical_count"]
    total = manifest["entity_count"]
    words = manifest["word_count"]
    print(f"  ✓  {scenario['id']}.txt  —  {words} words, {unique} unique entities ({total} total mentions)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("  Windows:  set ANTHROPIC_API_KEY=sk-ant-...")
        print("  Mac/Linux: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    client = anthropic.Anthropic(api_key=api_key)

    print(f"\nGenerating {len(SCENARIOS)} synthetic legal documents → {output_dir}/\n")

    success = 0
    successful_scenarios = []
    max_retries = 3
    for scenario in SCENARIOS:
        print(f"[{scenario['id']}]  Generating {scenario['doc_type']}...")
        for attempt in range(1, max_retries + 1):
            try:
                doc_text, ground_truth = generate_document(client, scenario)
                save(scenario, doc_text, ground_truth, output_dir)
                success += 1
                successful_scenarios.append(scenario)
                break
            except json.JSONDecodeError as e:
                print(f"  ✗  Attempt {attempt}/{max_retries} — JSON parse error: {e}")
            except Exception as e:
                print(f"  ✗  Attempt {attempt}/{max_retries} — Failed: {e}")
        else:
            print(f"  ✗  Skipping {scenario['id']} after {max_retries} failed attempts.")

    # Write index (only include successfully generated documents)
    index = {
        "generated_with": MODEL,
        "documents": [
            {
                "id": s["id"],
                "doc_type": s["doc_type"],
                "text_file": f"{s['id']}.txt",
                "ground_truth_file": f"{s['id']}_ground_truth.json",
            }
            for s in successful_scenarios
        ],
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2))

    print(f"\n{success}/{len(SCENARIOS)} documents generated successfully.")
    print(f"Output: {output_dir.resolve()}/\n")


if __name__ == "__main__":
    main()
