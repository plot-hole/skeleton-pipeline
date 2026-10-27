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
        <scenario_id>.txt
        <scenario_id>_ground_truth.json
        ...
        index.json

Scenarios cover: document type variety, entity coverage gaps, adversarial
negative cases, and cross-document consistency (shared corpus_id).
"""

import anthropic
import json
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("synthetic_legal_data")
MODEL = "claude-sonnet-4-20250514"

# ── Shared entities for cross-document consistency ────────────────────────────

CROSSDOC_SHARED_ENTITIES = {
    "corpus_id": "martinez_v_greenfield",
    "plaintiff_name": "Maria Elena Martinez",
    "plaintiff_ssn": "412-78-3956",
    "plaintiff_address": "2847 Sycamore Lane, Edison, New Jersey 08817",
    "plaintiff_phone": "(732) 548-9021",
    "plaintiff_email": "m.martinez847@gmail.com",
    "defendant_name": "Greenfield Properties LLC",
    "defendant_address": "1500 Commerce Drive, Suite 400, Newark, New Jersey 07102",
    "defendant_agent": "Harold T. Greenfield",
    "case_number": "2:24-cv-03891-JLW",
    "plaintiff_attorney": "Robert Chen",
    "plaintiff_law_firm": "Chen & Patel LLP",
    "plaintiff_firm_address": "88 Broad Street, 12th Floor, New York, New York 10004",
    "property_manager": "Angela Torres",
    "incident_date": "September 14, 2023",
    "property_address": "742 Elm Street, Unit 3A, Edison, New Jersey 08817",
}

# ── Document scenarios ────────────────────────────────────────────────────────

SCENARIOS = [
    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 1: Original document types (5)
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 2: Document type variety (4)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "motion_to_dismiss",
        "doc_type": "MOTION TO DISMISS",
        "description": """
            Defense motion to dismiss under Rule 12(b)(6) in a breach of contract case.
            Defendant corporation (named, with registered agent full name and corporate address
            in Delaware) moves to dismiss the complaint filed by plaintiff (named individual,
            address in Portland, Oregon). Defense attorney (full name, bar number, law firm name,
            firm address in Seattle, WA, phone, email). Plaintiff attorney (full name, bar number,
            firm in Portland, OR). Include: case number (state court format), filing date.

            IMPORTANT: The document must contain substantial legal boilerplate that should NOT be
            tagged as PII: "Pursuant to Rule 12(b)(6) of the Oregon Rules of Civil Procedure...",
            standard of review language, "COMES NOW the Defendant...", "WHEREFORE". Use "the Court"
            at least 10 times as a generic institutional reference (NOT a PII entity). Include
            references to "the State of Oregon" as a jurisdiction (NOT a PII entity).
        """,
    },
    {
        "id": "motion_summary_judgment",
        "doc_type": "MOTION FOR SUMMARY JUDGMENT",
        "description": """
            Plaintiff's motion for summary judgment in an employment discrimination case under
            Title VII. Plaintiff (named, female, address in Atlanta, GA, phone, email) sues
            defendant employer corporation (named, corporate address in Charlotte, NC). Plaintiff
            attorney (full name, firm name, address). Include: federal case number format
            (e.g. 1:23-cv-05678-RWS), filing date, multiple exhibit references (Exhibit 1
            through Exhibit 5).

            Include BARE DATES without explicit context labels — deposition dates, filing
            deadlines — stated as "March 15, 2024", "the deadline of April 30, 2024" without
            labels like "incident date" or "filing date".

            Include NON-PII DOLLAR SIGNS: "$300,000 statutory cap under 42 U.S.C. § 1981a",
            "$75,000 amount-in-controversy requirement", "$50 filing fee". These are generic
            legal references, NOT PII amounts. Separately include the actual PII damages sought.
        """,
    },
    {
        "id": "court_order_ruling",
        "doc_type": "COURT ORDER",
        "description": """
            Court order granting in part and denying in part a motion to compel discovery.
            Judge (full name, e.g. "the Honorable Patricia M. Blackwell") presides. Plaintiff
            (named individual) and defendant (named corporation). Both attorneys named. Case
            number in caption. Hearing date. Include: specific discovery deadlines.

            Use "the Court" extensively (15+ times) as an institutional self-reference by the
            judge — this is standard judicial writing and should NOT be tagged as PII. Use
            "the State" and "the People" as generic references. Reference "Rule 37" and
            "Rule 26(b)(1)" as procedural rules. Only the judge's actual name and the parties'
            names/case number are PII.
        """,
    },
    {
        "id": "demand_letter",
        "doc_type": "DEMAND LETTER",
        "description": """
            Pre-litigation demand letter from an attorney on behalf of a client injured in a
            slip-and-fall at a commercial property. Attorney (full name, law firm, firm address
            in Denver, CO, direct phone, fax number, email). Client/plaintiff (full name, home
            address in Boulder, CO, phone, DOB). Recipient: property management company (named
            corporation, corporate address in Denver, CO) attention: named risk manager (full
            name, direct email, direct phone).

            Include WRITTEN-OUT DOLLAR AMOUNTS: "Two Hundred Fifty Thousand Dollars ($250,000.00)"
            for the demand, "Forty-Seven Thousand Three Hundred Twenty-Two Dollars and 18/100
            ($47,322.18)" for medical bills. Include: incident date, demand deadline date,
            insurance claim number.
        """,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 3: Entity coverage gaps (10)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "multi_plaintiff_class_action",
        "doc_type": "CIVIL COMPLAINT",
        "description": """
            Consumer fraud class action with FOUR named plaintiffs and THREE defendants.

            Plaintiffs:
            - PLAINTIFF_1: Named individual, address in Sacramento, CA, phone, email
            - PLAINTIFF_2: Named individual, address in Los Angeles, CA, phone
            - PLAINTIFF_3: Named individual, address in San Diego, CA, phone, email
            - PLAINTIFF_4: Named individual, address in Fresno, CA, phone

            Defendants:
            - DEFENDANT_1: Named corporation (the product manufacturer)
            - DEFENDANT_2: Named corporation (the retailer)
            - DEFENDANT_3: Named individual (CEO of DEFENDANT_1, full name, address)

            Each plaintiff must be referenced by full name at least once, then by last name only
            multiple times. The key test: PLAINTIFF_1 vs PLAINTIFF_2 assignment must stay
            stable — never swap which person gets which token. Include: case number, filing date,
            class certification date, aggregate damages amount. Two plaintiff attorneys at one
            firm, one defense attorney.
        """,
    },
    {
        "id": "dual_role_attorney_expert",
        "doc_type": "CIVIL COMPLAINT",
        "description": """
            Breach of fiduciary duty case where entity roles are AMBIGUOUS:

            - Person A (full name, address in Boston, MA, phone, email) is BOTH the plaintiff's
              attorney of record AND serves as an expert witness on legal malpractice standards.
              They are referenced as "Attorney [Last Name]", "the expert", "[Last Name]", and
              by full name. The ground truth must tag all references with a DUAL role notation.

            - Person B (full name, address in Providence, RI) is BOTH a named defendant AND
              a fact witness (corporate officer who testifies). Referenced as "Defendant [Last Name]",
              "the witness", and by name.

            - Person C is the plaintiff AND the corporate representative of a plaintiff LLC.
              Referenced as "the Plaintiff", "the Managing Member", "Mr./Ms. [Last Name]".

            Include: case number, filing date, two law firm names, the plaintiff LLC name.
        """,
    },
    {
        "id": "multi_state_addresses",
        "doc_type": "CIVIL COMPLAINT",
        "description": """
            Diversity jurisdiction case with addresses in MULTIPLE STATES (no Ohio addresses):

            - Plaintiff: Full name, address in Billings, Montana (MT), phone, email
            - Defendant corporation: Incorporated in Delaware (DE), principal place of business
              at a full address in San Francisco, California (CA)
            - Witness 1: Full name, address in Miami, Florida (FL), phone
            - Witness 2: Full name, address in Austin, Texas (TX), email
            - Plaintiff attorney: Firm address in New York, New York (NY)
            - Defense attorney: Firm address in Los Angeles, California (CA)

            Use ZIP+4 format (XXXXX-XXXX) for at least two addresses. Include: federal case
            number, filing date, incident date, damages amount. This tests whether state
            extraction works for all US states, not just Ohio.
        """,
    },
    {
        "id": "written_dollar_amounts",
        "doc_type": "SETTLEMENT AGREEMENT",
        "description": """
            Settlement agreement where ALL dollar amounts appear in BOTH written-out and numeric form:

            - Settlement: "Seven Hundred Fifty Thousand and No/100 Dollars ($750,000.00)"
            - First installment: "Two Hundred Thousand Dollars ($200,000.00)"
            - Second installment: "One Hundred Fifty Thousand Dollars ($150,000.00)"
            - Third installment: "Four Hundred Thousand Dollars ($400,000.00)"
            - Attorney fees: "two hundred fifty thousand dollars ($250,000.00)"
            - Medical lien: "Sixty-Three Thousand Four Hundred Twelve Dollars and 87/100 ($63,412.87)"

            Plaintiff (named, address in suburban Virginia), defendant corporation (named, address
            in Maryland). Both attorneys named. Include: case number, payment schedule dates (four
            specific dates), execution date, notary named. All written-out dollar amounts are PII
            MONEY entities and must be tagged.
        """,
    },
    {
        "id": "federal_bankruptcy_case_numbers",
        "doc_type": "CIVIL COMPLAINT",
        "description": """
            Complaint with EDGE-CASE CASE NUMBERS in multiple formats:

            - Primary case number: federal format "1:23-cv-04821-ABC" (district:year-type-number-judge)
            - Related bankruptcy case: "23-BK-10458" (bankruptcy format)
            - State court predecessor: "2022-CV-00891" (state format)
            - Administrative proceeding reference: "AP-2023-0147"
            - SEC enforcement reference: "SEC File No. HO-12345"

            Each case number must appear at least twice in the document. Plaintiff (named
            individual, address in Manhattan, NY, SSN, phone). Defendant (named corporation,
            address in Wilmington, DE). Clerk of Court named. Include: filing date, hearing date,
            attorney names for both sides.
        """,
    },
    {
        "id": "contact_info_dense",
        "doc_type": "DEMAND LETTER",
        "description": """
            Demand letter in a data breach case, DENSE with contact information:

            EMAIL ADDRESSES (6 different):
            - Plaintiff personal email
            - Plaintiff work email
            - Plaintiff attorney email
            - Defendant corporate email (general counsel)
            - Defendant officer personal email
            - Defendant IT security contact email

            PHONE NUMBERS (6 different):
            - Plaintiff cell phone
            - Plaintiff work phone
            - Attorney office phone
            - Attorney fax number
            - Defendant main number
            - Defendant officer direct line

            Plaintiff (full name, home address in Portland, OR, DOB, SSN last four digits).
            Defendant corporation (named, address in Seattle, WA). Defendant CEO (named).
            Defendant CISO (named). Include: case reference number, incident date, demand
            amount, response deadline date.
        """,
    },
    {
        "id": "orgs_that_look_like_people",
        "doc_type": "CIVIL COMPLAINT",
        "description": """
            Product liability case where ORGANIZATION NAMES resemble person names:

            Defendants (all are CORPORATIONS, not people):
            - "Johnson & Johnson" (looks like two person names)
            - "Morgan Stanley" (looks like a person name)
            - "Baker Hughes" (looks like a person name)

            Plaintiff attorney's firm: "Williams & Connolly LLP" (looks like person names)
            Plaintiff's employer: "Marshall & Associates Inc." (ambiguous)

            The plaintiff (named individual, address in Philadelphia, PA, phone, email) was
            injured by a product. Include an individual defendant too — a named VP of Johnson
            & Johnson (full name, title, corporate email). Include: case number, filing date,
            incident date, medical costs. The ground truth must correctly classify all the above
            as ORG, not PERSON, despite the person-like names.
        """,
    },
    {
        "id": "people_that_look_like_places",
        "doc_type": "COURT ORDER",
        "description": """
            Court order where PERSON NAMES resemble place names:

            - Presiding judge: "the Honorable Judge Samuel Washington" (Washington = place)
            - Plaintiff attorney: "Virginia Austin, Esq." (Virginia = state, Austin = city)
            - Defendant: "Mr. Dallas Houston" (Dallas = city, Houston = city)
            - Expert witness: "Dr. Georgia Carolina Nash" (Georgia = state, Carolina = state)
            - Plaintiff: "Ms. Savannah Charlotte Brooks" (Savannah = city, Charlotte = city)

            All of these are PERSON entities. The ground truth must tag them as PERSON, not
            ADDRESS/LOCATION. Include: case number, hearing date, order date, one address for
            the plaintiff (in Richmond, Virginia — note: the STATE Virginia is part of the address,
            but the ATTORNEY Virginia Austin is a person). Short ruling on a discovery motion.
        """,
    },
    {
        "id": "ambiguous_dates",
        "doc_type": "DEMAND LETTER",
        "description": """
            Demand letter with dates in MANY FORMATS, some PII and some not:

            PII DATES (should be tagged):
            - Incident date: "January 5, 2024"
            - Client DOB: "March 12, 1985"
            - Medical treatment dates: "January 8, 2024" and "02/15/2024"
            - Demand deadline: "the 5th day of April, 2024"

            NON-PII DATES (should NOT be tagged):
            - Statute effective date: "effective January 1, 2020" (part of law reference)
            - Statute of limitations reference: "the two-year period under O.R.S. 12.110"
            - Generic calendar reference: "within thirty (30) days"

            Date formats to include: "January 5, 2024", "1/5/2024", "01-05-2024",
            "the 5th day of January, 2024". Plaintiff (named, address in Eugene, Oregon, phone).
            Defendant (named corporation, address in Portland, Oregon). Attorney (named, firm,
            firm address). Include: demand amount, medical costs.
        """,
    },
    {
        "id": "minor_witness_initials",
        "doc_type": "CRIMINAL COMPLAINT",
        "description": """
            Child endangerment case with sensitive PII handling:

            - TWO minor victims referenced by initials only: "A.B.C." (age 7) and "D.E.F." (age 4)
            - A CONFIDENTIAL INFORMANT referenced as "CI-1" (no name given)
            - Guardian ad litem (full name, bar number, address in Minneapolis, MN)
            - Defendant (full name, DOB, address in St. Paul, MN, phone)
            - Investigating officer (full name, badge number "Badge No. 4721", precinct address)
            - Prosecutor (full name, office address in Minneapolis, MN)

            Initials "A.B.C." and "D.E.F." are PII (ID type). "CI-1" is PII (ID type).
            Include: case number (state format), arrest date, incident date range, two statute
            citations (Minnesota state statutes). The minor victims are referred to as "the
            children", "the older child", "the younger child", and by initials.
        """,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 4: Adversarial / negative cases (4)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "adversarial_boilerplate",
        "doc_type": "MOTION TO DISMISS",
        "description": """
            Motion to dismiss that is EXTREMELY DENSE with legal boilerplate. The document
            should be ~900 words but contain only 3 actual PII entities buried in the text:
            - One plaintiff name (full name, mentioned twice)
            - One defendant name (corporation name, mentioned twice)
            - One case number (mentioned once in caption)

            The rest is pure legal language: "Pursuant to Rule 12(b)(6) of the Federal Rules
            of Civil Procedure...", "Ashcroft v. Iqbal, 556 U.S. 662 (2009)", "Bell Atlantic
            Corp. v. Twombly, 550 U.S. 544 (2007)", standard of review paragraphs, procedural
            history. Include at least 5 case citations (these are NOT PII). Include statutory
            references like "28 U.S.C. § 1332" (NOT PII). Use "the Court" 15+ times (NOT PII).
            Use "the Honorable Court" (NOT PII). Reference "the United States" and "the State
            of New York" as jurisdictions (NOT PII).

            The ground truth for this document should have VERY FEW entries — only the actual
            PII. This tests that the system does NOT over-tag boilerplate.
        """,
        "adversarial": True,
    },
    {
        "id": "adversarial_generic_references",
        "doc_type": "COURT ORDER",
        "description": """
            Court order that uses GENERIC INSTITUTIONAL REFERENCES extensively. These should
            NOT be tagged as PII entities:

            NOT PII (do not tag):
            - "the Court" (30+ uses as institutional self-reference)
            - "the State" (15+ uses as prosecutorial entity)
            - "the People" (10+ uses as prosecutorial party)
            - "the Commonwealth" (5+ uses)
            - "the Government" (5+ uses)
            - "the United States" (as sovereign, not PII)
            - "this Court", "the trial court", "the appellate court"

            Actual PII (tag these):
            - Judge's name in the caption (one name, mentioned twice)
            - Defendant's name (one individual, mentioned 4 times)
            - Case number
            - One address (defendant's home address)
            - Arrest date

            This is a sentencing order in a criminal case. Keep actual PII minimal but
            make the generic references very dense. The ground truth should have ~10 entries
            maximum despite the document being 800+ words.
        """,
        "adversarial": True,
    },
    {
        "id": "adversarial_dollar_and_numbers",
        "doc_type": "MOTION FOR SUMMARY JUDGMENT",
        "description": """
            Motion containing MANY dollar signs and number patterns that are NOT PII:

            NOT PII (do not tag):
            - "$75,000 amount-in-controversy requirement" (jurisdictional threshold)
            - "$300,000 statutory cap" (damages cap reference)
            - "$50 filing fee" (generic cost)
            - "No. 1 priority" (not a case number)
            - "Exhibit No. 1" through "Exhibit No. 5" (exhibit references, not case numbers)
            - "Docket No. 1", "Docket No. 15" (docket references)
            - "35%" (percentage)
            - "Rule 56(a)" (procedural rule)
            - "42 U.S.C. § 1983" (statute)

            Actual PII (tag these):
            - Plaintiff name, defendant corporation name
            - Case number (one, federal format)
            - Actual damages sought by plaintiff: "$2,450,000"
            - Plaintiff's lost wages: "$187,500"
            - Plaintiff attorney, defense attorney
            - Filing date

            This tests discrimination between PII dollar amounts and generic legal dollar
            references. The ground truth must ONLY include the actual PII items.
        """,
        "adversarial": True,
    },
    {
        "id": "adversarial_pattern_overlap",
        "doc_type": "CIVIL COMPLAINT",
        "description": """
            Complaint designed to create PATTERN OVERLAPS that confuse entity extractors:

            TRICKY PATTERNS (test correct handling):
            - Plaintiff lives at "1234 Court Street, Suite No. 5, Springfield, IL 62701"
              ("Court" in address, "No." in suite number — do NOT confuse with case number)
            - A company named "Court Services International, Inc." is a defendant
              ("Court" in org name — should be ORG, not confused with "the Court")
            - References to "Exhibit No. 1" through "Exhibit No. 7" — these contain "No."
              but are NOT case numbers
            - Case is styled "Case No. 2024-CV-00100" — this IS the case number
            - Plaintiff's street is "Justice Lane" — NOT a reference to justice/court system
            - Defendant officer named "Bill Case" — "Case" is a surname, not a legal term
            - Dollar amount in address context: "the $5 million property at 500 Lake Shore Drive"

            Include: plaintiff (named, with the Court Street address, SSN, phone, email),
            defendant corporation "Court Services International, Inc." (with address), individual
            defendant "Bill Case" (with address), plaintiff attorney, defense attorney, case
            number, filing date, damages amount.
        """,
        "adversarial": True,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # GROUP 5: Cross-document consistency (3, shared corpus_id)
    # ══════════════════════════════════════════════════════════════════════════
    {
        "id": "crossdoc_complaint",
        "doc_type": "CIVIL COMPLAINT",
        "corpus_id": "martinez_v_greenfield",
        "description": """
            Civil complaint: Martinez v. Greenfield Properties LLC.
            Habitability / negligent maintenance case. Plaintiff tenant sues landlord.

            USE THESE EXACT ENTITIES (shared across 3 documents in this corpus):
            - Plaintiff: Maria Elena Martinez, SSN 412-78-3956, residing at
              2847 Sycamore Lane, Edison, New Jersey 08817, phone (732) 548-9021,
              email m.martinez847@gmail.com
            - Defendant: Greenfield Properties LLC, 1500 Commerce Drive, Suite 400,
              Newark, New Jersey 07102, registered agent Harold T. Greenfield
            - Case Number: 2:24-cv-03891-JLW
            - Plaintiff Attorney: Robert Chen of Chen & Patel LLP, 88 Broad Street,
              12th Floor, New York, New York 10004
            - Property Manager: Angela Torres (employee of defendant, witnessed conditions)
            - Incident Date: September 14, 2023
            - Rental Property: 742 Elm Street, Unit 3A, Edison, New Jersey 08817

            Include: filing date, medical costs from mold exposure, damages sought.
            Martinez referenced as full name, "Martinez", "the Plaintiff", "she/her".
            Torres referenced as full name, "Torres", "the property manager".
        """,
    },
    {
        "id": "crossdoc_motion",
        "doc_type": "MOTION TO DISMISS",
        "corpus_id": "martinez_v_greenfield",
        "description": """
            Defendant's motion to dismiss in Martinez v. Greenfield Properties LLC.

            USE THESE EXACT ENTITIES (must match the complaint document):
            - Plaintiff: Maria Elena Martinez (referenced by name, "Martinez", "Plaintiff")
            - Defendant: Greenfield Properties LLC (referenced by name, "Greenfield", "Defendant")
            - Case Number: 2:24-cv-03891-JLW
            - Plaintiff Attorney: Robert Chen (referenced by name)
            - Property Manager: Angela Torres (referenced in factual summary)
            - Incident Date: September 14, 2023

            NEW ENTITIES introduced in this document:
            - Defense Attorney: Patricia Knowles of Knowles & Rivera LLP, 200 Park Avenue,
              Suite 3100, New York, New York 10166, phone (212) 555-0147, email
              pknowles@knowlesrivera.com
            - Expert cited: Dr. James Whitfield, environmental consultant

            Include: filing date (different from complaint), references to Rules 12(b)(1)
            and 12(b)(6). Use "the Court" generically. Legal boilerplate.
        """,
    },
    {
        "id": "crossdoc_order",
        "doc_type": "COURT ORDER",
        "corpus_id": "martinez_v_greenfield",
        "description": """
            Court order ruling on the motion to dismiss in Martinez v. Greenfield Properties LLC.

            USE THESE EXACT ENTITIES (must match both prior documents):
            - Plaintiff: Maria Elena Martinez
            - Defendant: Greenfield Properties LLC
            - Case Number: 2:24-cv-03891-JLW
            - Plaintiff Attorney: Robert Chen
            - Defense Attorney: Patricia Knowles
            - Property Manager: Angela Torres (referenced in findings)
            - Expert: Dr. James Whitfield (referenced in analysis)
            - Incident Date: September 14, 2023

            NEW ENTITIES introduced in this document:
            - Judge: the Honorable Linda R. Watkins
            - Hearing Date: March 8, 2025

            The order grants the motion in part (dismissing one count) and denies it in part.
            Use "the Court" extensively as self-reference (NOT PII). Minimal new PII — mostly
            re-references of existing entities from the complaint and motion. Include order date.
        """,
    },
]

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are generating synthetic legal documents for software testing.
All names, addresses, SSNs, case numbers, phone numbers, and other identifiers are
completely fictional and randomly invented. Do not use real people or real cases.

You write with realistic legal language and formatting appropriate to each document type."""


def build_prompt(scenario: dict) -> str:
    adversarial_note = ""
    if scenario.get("adversarial"):
        adversarial_note = """
ADVERSARIAL TEST DOCUMENT:
This document is designed to test that the de-identification system does NOT over-tag.
The ground truth must ONLY include actual PII entities. Legal boilerplate, generic
institutional references ("the Court", "the State"), case citations, statute numbers,
generic dollar amounts, and procedural references are NOT PII and must NOT appear in
the ground truth JSON.

In the ground truth, also include a separate "negative_examples" array listing items
that appear in the document but should NOT be tagged. Use this structure:
  "negative_examples": [
    {"text": "the Court", "reason": "generic institutional reference"},
    {"text": "28 U.S.C. § 1332", "reason": "statute citation, not PII"}
  ]
Output the negative_examples array AFTER the main entities array, separated by a line:
---NEGATIVE_EXAMPLES---
Then the JSON array of negative examples.
"""

    corpus_note = ""
    if scenario.get("corpus_id"):
        corpus_note = f"""
CROSS-DOCUMENT CONSISTENCY:
This document is part of corpus "{scenario['corpus_id']}". The entities listed in
the scenario MUST be used EXACTLY as specified — same spelling, same formats, same
numbers. Do not invent alternatives. These entities will appear across multiple
documents and must be identical for cross-document testing.
"""

    return f"""Generate a realistic synthetic {scenario["doc_type"]} for testing a PII de-identification system.

SCENARIO REQUIREMENTS:
{scenario["description"].strip()}
{adversarial_note}{corpus_note}
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
    "token_prefix": "e.g. PLAINTIFF, DEFENDANT, PLAINTIFF_1, PLAINTIFF_2, DEFENDANT_COMPANY, ADDRESS, INCIDENT_DATE, SSN, PHONE, EMAIL, CASE_NUMBER, LAW_FIRM, EXPERT_WITNESS, SETTLEMENT_AMOUNT, STATUTE, EXHIBIT, etc.",
    "entity_type": "PERSON | ORG | ADDRESS | DATE | MONEY | ID",
    "canonical": "the primary/full form this entity is an alias of (or same as text if this IS the canonical form)",
    "notes": "one short phrase explaining the classification"
  }}
]

Tag every single instance — if a name appears 6 times, include all 6. If "he" refers to the plaintiff 8 times, include all 8."""


# ── Generation ────────────────────────────────────────────────────────────────

def generate_document(client: anthropic.Anthropic, scenario: dict) -> tuple[str, list[dict], list[dict] | None]:
    """Generate a document and its ground truth. Returns (doc_text, entities, negative_examples)."""
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

    # Check for negative examples section (adversarial docs)
    negative_examples = None
    if "---NEGATIVE_EXAMPLES---" in gt_raw:
        gt_part, neg_part = gt_raw.split("---NEGATIVE_EXAMPLES---", 1)
        gt_raw = gt_part.strip()
        neg_raw = neg_part.strip()
        neg_raw = _strip_code_fences(neg_raw)
        try:
            negative_examples = json.loads(neg_raw)
        except json.JSONDecodeError:
            negative_examples = None  # Best effort
    else:
        gt_raw = gt_raw.strip()

    gt_raw = _strip_code_fences(gt_raw)
    ground_truth = json.loads(gt_raw)
    return doc_text, ground_truth, negative_examples


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences if present (handles ```json, ``` json, etc.)."""
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def save(scenario: dict, doc_text: str, ground_truth: list[dict],
         negative_examples: list[dict] | None, output_dir: Path):
    doc_path = output_dir / f"{scenario['id']}.txt"
    gt_path = output_dir / f"{scenario['id']}_ground_truth.json"

    doc_path.write_text(doc_text, encoding="utf-8")

    manifest = {
        "scenario_id": scenario["id"],
        "doc_type": scenario["doc_type"],
        "word_count": len(doc_text.split()),
        "entity_count": len(ground_truth),
        "unique_canonical_count": len({e.get("canonical", e["text"]) for e in ground_truth}),
    }

    if scenario.get("corpus_id"):
        manifest["corpus_id"] = scenario["corpus_id"]

    if scenario.get("adversarial"):
        manifest["adversarial"] = True

    manifest["entities"] = ground_truth

    if negative_examples:
        manifest["negative_examples"] = negative_examples

    gt_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    unique = manifest["unique_canonical_count"]
    total = manifest["entity_count"]
    words = manifest["word_count"]
    tag = ""
    if scenario.get("corpus_id"):
        tag += f" [corpus:{scenario['corpus_id']}]"
    if scenario.get("adversarial"):
        tag += " [adversarial]"
    print(f"  ✓  {scenario['id']}.txt  —  {words} words, {unique} unique entities ({total} total mentions){tag}")


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
        tag = ""
        if scenario.get("corpus_id"):
            tag = f" [corpus:{scenario['corpus_id']}]"
        if scenario.get("adversarial"):
            tag += " [adversarial]"
        print(f"[{scenario['id']}]  Generating {scenario['doc_type']}...{tag}")
        for attempt in range(1, max_retries + 1):
            try:
                doc_text, ground_truth, negative_examples = generate_document(client, scenario)
                save(scenario, doc_text, ground_truth, negative_examples, output_dir)
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
    # Group by corpus_id if present
    corpora = {}
    for s in successful_scenarios:
        cid = s.get("corpus_id")
        if cid:
            corpora.setdefault(cid, []).append(s["id"])

    index = {
        "generated_with": MODEL,
        "total_documents": len(successful_scenarios),
        "corpora": corpora if corpora else None,
        "documents": [
            {
                "id": s["id"],
                "doc_type": s["doc_type"],
                "text_file": f"{s['id']}.txt",
                "ground_truth_file": f"{s['id']}_ground_truth.json",
                **({"corpus_id": s["corpus_id"]} if s.get("corpus_id") else {}),
                **({"adversarial": True} if s.get("adversarial") else {}),
            }
            for s in successful_scenarios
        ],
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2))

    # Summary
    adversarial_count = sum(1 for s in successful_scenarios if s.get("adversarial"))
    crossdoc_count = sum(1 for s in successful_scenarios if s.get("corpus_id"))
    print(f"\n{success}/{len(SCENARIOS)} documents generated successfully.")
    if adversarial_count:
        print(f"  {adversarial_count} adversarial (negative test) documents")
    if crossdoc_count:
        print(f"  {crossdoc_count} cross-document consistency documents ({len(corpora)} corpora)")
    print(f"Output: {output_dir.resolve()}/\n")


if __name__ == "__main__":
    main()
