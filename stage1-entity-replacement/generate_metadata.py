"""Generate simulated Cyera classification metadata from a plain text document.

Uses spaCy NER to scan the document and produce a metadata JSON file
compatible with the Stage 1 pipeline.

Usage:
    python generate_metadata.py document.txt                    # writes document_metadata.json
    python generate_metadata.py document.txt -o output.json     # writes to custom path
    python generate_metadata.py document.txt --run              # generates metadata AND runs pipeline
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# spaCy label -> Cyera classification mapping
# ---------------------------------------------------------------------------

# Maps spaCy NER labels to (category, subcategory, sensitivity, role)
_LABEL_MAP: dict[str, dict] = {
    "PERSON": {
        "category": "DIRECT_IDENTIFIER",
        "subcategory": "FULL_NAME",
        "sensitivity": "CRITICAL",
        "role": "PARTY_PRIMARY",  # will alternate for multiple persons
    },
    "ORG": {
        "category": "QUASI_IDENTIFIER",
        "subcategory": "EMPLOYER_NAME",
        "sensitivity": "HIGH",
        "role": "EMPLOYER",
    },
    "GPE": {
        "category": "QUASI_IDENTIFIER",
        "subcategory": "ADDRESS",
        "sensitivity": "HIGH",
        "role": "PARTY_PRIMARY",
    },
    "DATE": {
        "category": "QUASI_IDENTIFIER",
        "subcategory": "DATE",
        "sensitivity": "HIGH",
        "role": "PARTY_PRIMARY",
    },
    "MONEY": {
        "category": "FINANCIAL_DATA",
        "subcategory": "RENT",
        "sensitivity": "HIGH",
        "role": "PARTY_PRIMARY",
    },
    "CARDINAL": None,      # skip
    "ORDINAL": None,       # skip
    "QUANTITY": None,       # skip
    "PERCENT": None,        # skip
    "TIME": None,           # skip
    "WORK_OF_ART": None,   # skip
    "LAW": None,            # skip (legal references, not PII)
    "LANGUAGE": None,       # skip
    "EVENT": None,          # skip
    "FAC": None,            # skip (facilities)
    "PRODUCT": None,        # skip
    "NORP": None,           # skip (nationalities, religious/political groups)
    "LOC": {
        "category": "QUASI_IDENTIFIER",
        "subcategory": "ADDRESS",
        "sensitivity": "HIGH",
        "role": "PARTY_PRIMARY",
    },
}

# ---------------------------------------------------------------------------
# Regex patterns for additional PII detection
# ---------------------------------------------------------------------------

_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_PATTERN = re.compile(r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}")
_DOB_KEYWORDS = {"date of birth", "dob", "born on", "birthdate", "birth date"}
_VIN_PATTERN = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
# Account number patterns — multiple formats:
#   "Account Number 44-8827561", "Account Numbers ending in 7834 and 7841",
#   "account ending in 6847", "Account No. 4471-8829-3156-7742"
_ACCOUNT_FULL_PATTERN = re.compile(
    r"(?:account|acct)\.?\s*(?:numbers?|no\.?|#)\s*[:.]?\s*"
    r"\*{0,2}([\d][\d\-]{3,}[\d])\*{0,2}",
    re.IGNORECASE,
)
_ACCOUNT_ENDING_PATTERN = re.compile(
    r"(?:account|acct)\.?\s*(?:numbers?|no\.?|#)?\s*"
    r"ending\s+in\s*:?\s*"
    r"(\*{0,}\d{4})\*{0,2}(?:\s+and\s+(\*{0,}\d{4}))?",
    re.IGNORECASE,
)
# Labeled numeric identifiers: "Commission Number: 598714", "Case Reference No.: SPM-2024-EV-003891",
# "State Filing Number: 202409847521", "Employer Identification Number: 84-3927156"
# Handles markdown bold markers (**) around labels and values
_LABELED_ID_PATTERN = re.compile(
    r"\*{0,2}(?:commission|case\s+reference|filing|employer\s+identification|federal\s+tax(?:\s+id)?|"
    r"ein|state\s+filing|docket|policy|permit|registration|certificate)"
    r"\s*(?:numbers?|no\.?|id|#)\s*[:.]?\s*\*{0,2}\s*"
    r"\*{0,2}([\w][\w\-]{2,}[\w])\*{0,2}",
    re.IGNORECASE,
)
_LICENSE_PLATE_PATTERN = re.compile(r"\b\d[A-Z]{3}\d{3}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Street address pattern: number + street name + street suffix
# Captures: "4782 Copper Ridge Lane", "892 Windmill Drive", "5521 North Hayden Road"
# Optionally followed by unit/apt/suite info
_STREET_SUFFIXES_RE = (
    r"(?:Lane|Ln|Drive|Dr|Way|Road|Rd|Street|St|Avenue|Ave|Boulevard|Blvd|"
    r"Court|Ct|Place|Pl|Circle|Cir|Terrace|Ter|Ridge|Trail|Trl|"
    r"Parkway|Pkwy|Highway|Hwy|Path|Loop|Run|Crossing|Pike|Square|Alley|Row)"
)
_STREET_ADDRESS_PATTERN = re.compile(
    r"\b(\d{1,6}\s+"                      # Street number
    r"(?:[A-Z][a-zA-Z]*\s+){1,4}"        # Street name words (1-4 words, capitalized)
    r"" + _STREET_SUFFIXES_RE + r")\b"    # Street suffix + word boundary (prevents "St" matching "State")
    r"(?:"                                 # Optional unit/apt/suite
    r"(?:\s*,)?\s*"
    r"(?:(?:Unit|Apt\.?|Suite|Ste\.?|Apartment|#)\s*[\w-]+)"
    r")?",
    re.MULTILINE
)

# State+ZIP pattern: captures full "Arizona 85254" or "AZ 85254" as one element
# Group 1: full "State ZIP" string, Group 2: state name/code
_STATE_ZIP_PATTERN = re.compile(
    r"(?:,?\s*)"
    r"((Alabama|Alaska|Arizona|Arkansas|California|Colorado|"
    r"Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|"
    r"Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|"
    r"Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|"
    r"New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|"
    r"Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|"
    r"South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|"
    r"West\s+Virginia|Wisconsin|Wyoming|"
    r"[A-Z]{2})"
    r"\s+\d{5}(?:-\d{4})?)\b"
)


# ---------------------------------------------------------------------------
# Known legal boilerplate phrases
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [
    r"(?i)this agreement shall be governed by.*?(?:\.|\n)",
    r"(?i)in witness whereof.*?(?:\.|\n)",
    r"(?i)now,?\s+therefore.*?(?:\.|\n)",
    r"(?i)whereas,.*?(?:;|\n)",
]

# ---------------------------------------------------------------------------
# Duration-like DATE strings that are NOT calendar dates
# ---------------------------------------------------------------------------

_DURATION_PATTERN = re.compile(
    r"^\d+\s*(days?|weeks?|months?|years?|hours?|minutes?)$", re.IGNORECASE
)

# Text-number durations: "twenty-three years", "the age of eighteen", etc.
_TEXT_DURATION_PATTERN = re.compile(
    r"(?:the\s+age\s+of\s+)?\b(?:one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    r"thousand|million|billion)"
    r"(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?",
    re.IGNORECASE,
)

# Dollar amounts in words that spaCy may tag as DATE
_DOLLAR_WORDS_PATTERN = re.compile(
    r"(?:hundred|thousand|million|billion)\s+dollars?", re.IGNORECASE
)

# ZIP code pattern (5-digit numbers that spaCy may tag as DATE)
_ZIP_CODE_PATTERN = re.compile(r"^\d{5}(?:-\d{4})?$")

# ---------------------------------------------------------------------------
# Allowlisted entities that should NOT be classified as PII
# ---------------------------------------------------------------------------

_SKIP_ENTITIES = {
    # Country references
    "United States", "United States of America", "U.S.", "USA", "U.S.A.",

    # Legal role terms
    "Tenant", "Landlord", "Lessor", "Lessee", "Buyer", "Seller",
    "Borrower", "Lender", "Mother", "Father", "Parents",
    "Property", "Premises", "Testator", "Executor", "Guarantor",
    "Principal", "Agent", "Attorney", "Witness", "Notary",
    "Grantor", "Grantee", "Trustee", "Beneficiary", "Assignor", "Assignee",
    "Contractor", "Client", "Company", "Member", "Manager",
    "Disclosing Party", "Receiving Party",

    # Legal terms spaCy may tag
    "Social Security Number", "Social Security",
    "Confidential Information", "Agreement", "Testament",
    "Notary Public", "Commission", "Intellectual Property",

    # Medical / healthcare terms spaCy may tag as ORG
    "Healthcare", "Health Care", "Healthcare Agent",
    "Primary Healthcare Agent", "Alternate Healthcare Agent",
    "Primary Healthcare", "Alternate Healthcare",
    "Medical Power", "Power of Attorney", "HIPAA",
    "Living Will", "Independent Contractor",

    # Document / structural terms
    "PRIVACY ENGINEERING", "ENTITY REPLACEMENT",
    "SYNTHETIC TEST DOCUMENT", "Party",
    "DELIVERABLES",
}

# State name -> 2-letter code mapping (for jurisdiction detection)
_STATE_TO_CODE: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC",
}

# Reverse mapping: 2-letter code -> full name
_CODE_TO_STATE = {v: k for k, v in _STATE_TO_CODE.items()}

# States to skip (they appear in governing law clauses, addresses, etc.)
_US_STATES = set(_STATE_TO_CODE.keys())

# Street / road suffixes — entities ending with these are address components, not ORGs
_STREET_SUFFIXES = {
    "lane", "ln", "drive", "dr", "way", "road", "rd", "street", "st",
    "avenue", "ave", "boulevard", "blvd", "court", "ct", "place", "pl",
    "circle", "cir", "terrace", "ter", "ridge", "trail", "trl",
    "parkway", "pkwy", "highway", "hwy", "path", "loop", "run",
    "crossing", "xing", "pike", "square", "sq", "alley", "row",
}

# Automotive / product brands spaCy may tag as ORG
_PRODUCT_BRANDS = {
    "toyota", "honda", "ford", "chevrolet", "chevy", "nissan", "hyundai",
    "kia", "subaru", "mazda", "bmw", "mercedes", "audi", "volkswagen",
    "lexus", "acura", "infiniti", "tesla", "jeep", "dodge", "ram",
    "chrysler", "buick", "cadillac", "gmc", "lincoln", "volvo", "porsche",
    "jaguar", "land rover", "mini", "fiat", "mitsubishi", "suzuki",
}

# Common US city names (that may appear as PERSON in address context)
# These are checked only when the entity appears in clear address context.
_COMMON_CITIES_AS_NAMES = {
    "gilbert", "chandler", "mesa", "glendale", "florence", "jackson",
    "madison", "clinton", "taylor", "lincoln", "austin", "tyler",
    "norman", "jordan", "brandon", "mason", "carson", "edison",
    "hamilton", "jefferson", "troy", "warren", "chester", "foster",
}


# ---------------------------------------------------------------------------
# Helper: detect if entity is in address context
# ---------------------------------------------------------------------------

def _looks_like_address_org(text: str, ent_text: str, start_char: int) -> bool:
    """Check if an ORG entity looks like it's part of an address."""
    ent_lower = ent_text.lower()

    # If it ends with a street suffix, it's an address component
    words = ent_lower.split()
    if words and words[-1].rstrip(".,") in _STREET_SUFFIXES:
        return True

    # If it starts with a number (e.g., "1156 Sycamore Terrace"), it's an address
    if words and words[0].isdigit():
        return True

    # If a known automotive brand
    if ent_lower in _PRODUCT_BRANDS:
        return True

    return False


def _is_person_in_address_context(text: str, ent_text: str, start_char: int, end_char: int) -> bool:
    """Check if a single-word PERSON entity looks like a city name in an address."""
    if " " in ent_text.strip():
        return False  # multi-word names are real names

    ent_lower = ent_text.lower().strip()
    if ent_lower not in _COMMON_CITIES_AS_NAMES:
        return False

    # Check the surrounding context for address patterns (state name/abbreviation after it)
    after_text = text[end_char:end_char + 40].strip()
    # Pattern: "Gilbert, Arizona" or "Gilbert, AZ" or "Arizona 85297"
    if re.match(r"^,?\s*(?:[A-Z][a-z]+|[A-Z]{2})\s+\d{5}", after_text):
        return True

    return False


def _is_false_date(ent_text: str) -> bool:
    """Check if a DATE entity is actually a non-date (zip, dollar words, text duration, etc.)."""
    stripped = ent_text.strip()

    # ZIP codes
    if _ZIP_CODE_PATTERN.match(stripped):
        return True

    # Dollar amounts in words
    if _DOLLAR_WORDS_PATTERN.search(stripped):
        return True

    # Numeric durations ("30 days", "12 months")
    if _DURATION_PATTERN.match(stripped):
        return True

    # Text durations and ages
    lower = stripped.lower()
    if any(kw in lower for kw in [
        "years", "months", "weeks", "days",
        "the age of", "age of majority",
    ]):
        return True

    # Single words that aren't dates
    non_date_words = {"testament", "annual", "monthly", "weekly", "daily", "quarterly"}
    if lower in non_date_words:
        return True

    return False


# ---------------------------------------------------------------------------
# Jurisdiction detection helpers
# ---------------------------------------------------------------------------

def _detect_gpe_jurisdiction(text: str, char_start: int, char_end: int) -> str | None:
    """Detect jurisdiction for a GPE entity from immediate surrounding context.

    Looks for a US state name or 2-letter abbreviation near the entity.
    Returns 'US_XX' if found, or None if no state is detectable.
    """
    # Look at text after the entity (e.g., "Scottsdale, Arizona 85254")
    after = text[char_end:min(len(text), char_end + 80)]

    # Check for full state name: ", Arizona" or " Arizona"
    m = re.match(r"\s*,?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", after)
    if m:
        candidate = re.sub(r"\s+", " ", m.group(1).strip())
        if candidate in _STATE_TO_CODE:
            return f"US_{_STATE_TO_CODE[candidate]}"

    # Check for 2-letter state code: ", AZ" or " AZ"
    m = re.match(r"\s*,?\s*([A-Z]{2})\b", after)
    if m and m.group(1) in _CODE_TO_STATE:
        return f"US_{m.group(1)}"

    # Check text before the entity for state context
    before = text[max(0, char_start - 80):char_start]
    for state_name in sorted(_STATE_TO_CODE.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(state_name) + r"\b", before):
            return f"US_{_STATE_TO_CODE[state_name]}"
    for m_iter in re.finditer(r"\b([A-Z]{2})\b", before):
        if m_iter.group(1) in _CODE_TO_STATE:
            return f"US_{m_iter.group(1)}"

    return None


def _detect_document_jurisdiction(text: str) -> str:
    """Detect the primary US state jurisdiction for the entire document.

    Scans the full text for state names and abbreviations, returning the
    most-frequently referenced state as 'US_XX'.  Falls back to 'US_CA'.
    """
    state_counts: dict[str, int] = {}

    # Count full state name occurrences
    for state_name, code in _STATE_TO_CODE.items():
        count = len(re.findall(r"\b" + re.escape(state_name) + r"\b", text))
        if count > 0:
            state_counts[code] = state_counts.get(code, 0) + count

    # Count 2-letter codes that appear before ZIP codes (strong address signal)
    for m in re.finditer(r"\b([A-Z]{2})\s+\d{5}", text):
        code = m.group(1)
        if code in _CODE_TO_STATE:
            state_counts[code] = state_counts.get(code, 0) + 2  # weight higher

    if state_counts:
        primary_code = max(state_counts, key=state_counts.get)
        return f"US_{primary_code}"

    return "US_CA"  # ultimate fallback


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_metadata(text: str, doc_id: str | None = None) -> dict:
    """Scan a document with spaCy and produce Cyera-compatible metadata."""
    import spacy
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)

    raw = text.encode("utf-8")
    doc_jurisdiction = _detect_document_jurisdiction(text)
    elements: list[dict] = []
    seen_spans: set[tuple[int, int]] = set()
    person_count = 0

    # Track PERSON names for deduplication with ORG
    person_texts: set[str] = set()
    person_surnames: set[str] = set()
    known_name_parts: set[str] = set()  # Individual name parts (uppercase) for span expansion
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name_upper = ent.text.strip().upper()
            person_texts.add(name_upper)
            # Extract surname (last word) for partial matching
            parts = name_upper.split()
            if len(parts) >= 2:
                person_surnames.add(parts[-1])
            # Collect individual name parts for span expansion
            for part in parts:
                if len(part) >= 3:
                    known_name_parts.add(part)

    # --- 0. High-confidence regex patterns (run BEFORE NER to claim spans first) ---
    # VIN, account numbers, and labeled IDs have keyword context that makes them
    # high-confidence.  Running them first prevents spaCy from mis-classifying
    # account numbers like "8351" as dates.

    # --- VIN detection ---
    for match in _VIN_PATTERN.finditer(text):
        vin_text = match.group()
        prefix_bytes = text[:match.start()].encode("utf-8")
        vin_bytes = vin_text.encode("utf-8")
        byte_start = len(prefix_bytes)
        byte_end = byte_start + len(vin_bytes)
        span_key = (byte_start, byte_end)
        overlap = False
        for (es, ee) in seen_spans:
            if byte_start < ee and byte_end > es:
                overlap = True
                break
        if not overlap:
            seen_spans.add(span_key)
            elements.append({
                "element_id": str(uuid.uuid4()),
                "category": "DIRECT_IDENTIFIER",
                "subcategory": "VIN",
                "sensitivity": "HIGH",
                "byte_offset_start": byte_start,
                "byte_offset_end": byte_end,
                "original_text": vin_text,
                "context": {
                    "data_subject_role": "PARTY_PRIMARY",
                    "jurisdiction": doc_jurisdiction,
                    "applicable_regulations": ["CCPA", "SOC2"],
                },
            })

    # --- Account number detection (full account numbers) ---
    for match in _ACCOUNT_FULL_PATTERN.finditer(text):
        acct_num = match.group(1)
        prefix_bytes = text[:match.start(1)].encode("utf-8")
        acct_bytes = acct_num.encode("utf-8")
        byte_start = len(prefix_bytes)
        byte_end = byte_start + len(acct_bytes)
        span_key = (byte_start, byte_end)
        overlap = False
        for (es, ee) in seen_spans:
            if byte_start < ee and byte_end > es:
                overlap = True
                break
        if not overlap:
            seen_spans.add(span_key)
            elements.append({
                "element_id": str(uuid.uuid4()),
                "category": "FINANCIAL_DATA",
                "subcategory": "ACCOUNT_NUMBER",
                "sensitivity": "CRITICAL",
                "byte_offset_start": byte_start,
                "byte_offset_end": byte_end,
                "original_text": acct_num,
                "context": {
                    "data_subject_role": "PARTY_PRIMARY",
                    "jurisdiction": doc_jurisdiction,
                    "applicable_regulations": ["CCPA", "SOC2"],
                },
            })

    # --- Account number detection (partial "ending in" format) ---
    for match in _ACCOUNT_ENDING_PATTERN.finditer(text):
        for grp_idx in (1, 2):
            acct_num = match.group(grp_idx)
            if acct_num is None:
                continue
            grp_start = match.start(grp_idx)
            prefix_bytes = text[:grp_start].encode("utf-8")
            acct_bytes = acct_num.encode("utf-8")
            byte_start = len(prefix_bytes)
            byte_end = byte_start + len(acct_bytes)
            span_key = (byte_start, byte_end)
            overlap = False
            for (es, ee) in seen_spans:
                if byte_start < ee and byte_end > es:
                    overlap = True
                    break
            if not overlap:
                seen_spans.add(span_key)
                elements.append({
                    "element_id": str(uuid.uuid4()),
                    "category": "FINANCIAL_DATA",
                    "subcategory": "ACCOUNT_NUMBER",
                    "sensitivity": "HIGH",
                    "byte_offset_start": byte_start,
                    "byte_offset_end": byte_end,
                    "original_text": acct_num,
                    "context": {
                        "data_subject_role": "PARTY_PRIMARY",
                        "jurisdiction": doc_jurisdiction,
                        "applicable_regulations": ["CCPA", "SOC2"],
                    },
                })

    # --- Labeled identifier detection (commission numbers, case numbers, EINs, filing numbers) ---
    for match in _LABELED_ID_PATTERN.finditer(text):
        id_text = match.group(1)
        prefix_bytes = text[:match.start(1)].encode("utf-8")
        id_bytes = id_text.encode("utf-8")
        byte_start = len(prefix_bytes)
        byte_end = byte_start + len(id_bytes)
        span_key = (byte_start, byte_end)
        overlap = False
        for (es, ee) in seen_spans:
            if byte_start < ee and byte_end > es:
                overlap = True
                break
        if not overlap:
            seen_spans.add(span_key)
            label_text = match.group(0).lower()
            if "employer identification" in label_text or "ein" in label_text or "federal tax" in label_text:
                subcat = "EIN"
                cat = "FINANCIAL_DATA"
            elif "case" in label_text or "docket" in label_text:
                subcat = "CASE_NUMBER"
                cat = "DIRECT_IDENTIFIER"
            else:
                subcat = "LICENSE_NUMBER"
                cat = "DIRECT_IDENTIFIER"
            elements.append({
                "element_id": str(uuid.uuid4()),
                "category": cat,
                "subcategory": subcat,
                "sensitivity": "HIGH",
                "byte_offset_start": byte_start,
                "byte_offset_end": byte_end,
                "original_text": id_text,
                "context": {
                    "data_subject_role": "PARTY_PRIMARY",
                    "jurisdiction": doc_jurisdiction,
                    "applicable_regulations": ["CCPA", "SOC2"],
                },
            })

    # --- 1. spaCy NER entities ---
    for ent in doc.ents:
        mapping = _LABEL_MAP.get(ent.label_)
        if mapping is None:
            continue

        ent_text = ent.text.strip()

        # Skip allowlisted / state entities (case-insensitive for states)
        if ent_text in _SKIP_ENTITIES or ent_text in _US_STATES or ent_text.title() in _US_STATES:
            continue

        # Skip very short entities (likely noise)
        if len(ent_text) < 2:
            continue

        # Skip if any word in entity is in _SKIP_ENTITIES
        if any(word in _SKIP_ENTITIES for word in [ent_text]):
            continue

        # --- ORG-specific filtering ---
        reclassified_as_person = False
        if ent.label_ == "ORG":
            # Skip if entity text was already detected as PERSON
            if ent_text.upper() in person_texts:
                continue

            # Reclassify as PERSON if entity shares a surname with a known PERSON
            # (spaCy sometimes tags all-caps names as ORG)
            ent_words = ent_text.strip().upper().split()
            if len(ent_words) >= 2 and ent_words[-1] in person_surnames:
                mapping = _LABEL_MAP["PERSON"]
                reclassified_as_person = True
                person_texts.add(ent_text.strip().upper())
            else:
                # Skip if it looks like an address component
                if _looks_like_address_org(text, ent_text, ent.start_char):
                    continue

                # Skip if entity contains line breaks (multi-line noise)
                if "\n" in ent_text:
                    continue

                # Skip legal / document terms tagged as ORG (substring match)
                # NOTE: Only use substring matching for terms that won't appear
                # inside real company names.  Medical/legal terms like "healthcare"
                # must use the exact-match check below to avoid filtering out
                # companies like "Meridian Healthcare Solutions".
                ent_lower = ent_text.lower()
                if any(skip.lower() in ent_lower for skip in [
                    "social security", "notary public", "commission",
                    "privacy engineering", "entity replacement",
                    "synthetic test", "department of",
                ]):
                    continue

                # Exact-match skip for ORG entities that are legal/medical terms
                if ent_lower in {
                    "healthcare", "health care", "healthcare agent",
                    "primary healthcare agent", "alternate healthcare agent",
                    "primary healthcare", "alternate healthcare",
                    "medical power", "power of attorney",
                    "hipaa", "living will", "independent contractor",
                    "state", "county", "the state",
                    "intellectual property", "confidential information",
                    "party",
                }:
                    continue

                # Skip product brands
                if ent_lower in _PRODUCT_BRANDS:
                    continue

        # --- DATE-specific filtering ---
        if ent.label_ == "DATE":
            if _is_false_date(ent_text):
                continue

        # --- PERSON-specific filtering ---
        if ent.label_ == "PERSON":
            # Skip single-word names that look like cities in address context
            if _is_person_in_address_context(text, ent_text, ent.start_char, ent.end_char):
                continue

        # --- PERSON span expansion: capture adjacent name parts ---
        # spaCy sometimes splits multi-word names (e.g., detects "DOROTHY ELLEN"
        # but misses "HUTCHINSON").  Expand the span to include adjacent
        # ALL-CAPS or known name-part words.
        char_start = ent.start_char
        char_end = ent.end_char
        if ent.label_ == "PERSON" or reclassified_as_person:
            exp_start, exp_end = char_start, char_end
            # Expand forward: include adjacent capitalized name words
            while exp_end < len(text):
                fwd = re.match(r'[ \t]+([A-Z][a-zA-Z]+)\b', text[exp_end:])
                if not fwd:
                    break
                word = fwd.group(1)
                if len(word) < 2:
                    break
                # Stop at known non-name entities (legal roles, etc.)
                if word in _SKIP_ENTITIES or word.title() in _SKIP_ENTITIES:
                    break
                # State name check: skip only if followed by ZIP (address context)
                # Otherwise allow — it's likely a surname (e.g., "WASHINGTON")
                is_state = word in _US_STATES or word.title() in _US_STATES
                if is_state:
                    after_word = text[exp_end + len(fwd.group(0)):]
                    if re.match(r'\s+\d{5}', after_word):
                        break  # address context — it's a state reference
                    # Not followed by ZIP — treat as surname, fall through
                # Accept ALL-CAPS words (>= 3 chars) or known name parts
                if (word.isupper() and len(word) >= 3) or word.upper() in known_name_parts:
                    exp_end += len(fwd.group(0))
                else:
                    break

            # Expand backward: include adjacent capitalized name words
            while exp_start > 0:
                bwd = re.search(r'\b([A-Z][a-zA-Z]+)[ \t]+$', text[:exp_start])
                if not bwd:
                    break
                word = bwd.group(1)
                if len(word) < 2:
                    break
                if word in _SKIP_ENTITIES or word.title() in _SKIP_ENTITIES:
                    break
                # State name check: skip only in address/legal context
                is_state = word in _US_STATES or word.title() in _US_STATES
                if is_state:
                    before_word = text[:bwd.start(1)].rstrip()
                    if re.search(r'(?:,\s*|state\s+of\s+|of\s+)$', before_word, re.IGNORECASE):
                        break  # address/legal context — it's a state reference
                    # Otherwise treat as surname, fall through
                if (word.isupper() and len(word) >= 3) or word.upper() in known_name_parts:
                    exp_start = bwd.start(1)
                else:
                    break

            if exp_start != char_start or exp_end != char_end:
                char_start = exp_start
                char_end = exp_end
                ent_text = text[char_start:char_end].strip()
                # Update tracking sets with the expanded name
                person_texts.add(ent_text.upper())
                for w in ent_text.upper().split():
                    if len(w) >= 3:
                        known_name_parts.add(w)

        # Compute byte offsets from (possibly expanded) character offsets
        prefix = text[:char_start].encode("utf-8")
        entity_bytes = text[char_start:char_end].encode("utf-8")
        byte_start = len(prefix)
        byte_end = byte_start + len(entity_bytes)

        # Deduplicate by span (check for overlaps, not just exact matches)
        span_key = (byte_start, byte_end)
        overlap_with_existing = False
        for (es, ee) in seen_spans:
            if byte_start < ee and byte_end > es:
                overlap_with_existing = True
                break
        if overlap_with_existing:
            continue
        seen_spans.add(span_key)

        # Determine role for PERSON entities (including reclassified ones)
        role = mapping["role"]
        if ent.label_ == "PERSON" or reclassified_as_person:
            person_count += 1
            role = "PARTY_PRIMARY" if person_count % 2 == 1 else "PARTY_SECONDARY"

        # Check if this DATE entity looks like a DOB
        subcategory = mapping["subcategory"]
        if ent.label_ == "DATE":
            context_start = max(0, ent.start_char - 80)
            context_text = text[context_start:ent.start_char].lower()
            if any(kw in context_text for kw in _DOB_KEYWORDS):
                subcategory = "DATE_OF_BIRTH"

        # Check if MONEY entity is income vs rent
        if ent.label_ == "MONEY":
            context_start = max(0, ent.start_char - 100)
            context_text = text[context_start:ent.start_char].lower()
            if "income" in context_text or "salary" in context_text or "earn" in context_text:
                subcategory = "INCOME"
            elif "deposit" in context_text:
                subcategory = "RENT"
            elif "support" in context_text:
                subcategory = "RENT"
            elif "price" in context_text or "purchase" in context_text:
                subcategory = "PURCHASE_PRICE"
            elif "settlement" in context_text:
                subcategory = "SETTLEMENT"
            elif "valued" in context_text or "value" in context_text:
                subcategory = "RENT"  # estate values

        # Determine jurisdiction — GPE/LOC entities use local context detection
        if ent.label_ in ("GPE", "LOC"):
            ent_jurisdiction = _detect_gpe_jurisdiction(text, ent.start_char, ent.end_char) or doc_jurisdiction
        else:
            ent_jurisdiction = doc_jurisdiction

        elements.append({
            "element_id": str(uuid.uuid4()),
            "category": mapping["category"],
            "subcategory": subcategory,
            "sensitivity": mapping["sensitivity"],
            "byte_offset_start": byte_start,
            "byte_offset_end": byte_end,
            "original_text": text[char_start:char_end],
            "context": {
                "data_subject_role": role,
                "jurisdiction": ent_jurisdiction,
                "applicable_regulations": ["CCPA", "SOC2"],
            },
        })

    # --- 2. Regex-based detection (SSN, phone, VIN, license plate, email) ---
    for match in _SSN_PATTERN.finditer(text):
        prefix_bytes = text[:match.start()].encode("utf-8")
        ssn_bytes = match.group().encode("utf-8")
        byte_start = len(prefix_bytes)
        byte_end = byte_start + len(ssn_bytes)
        span_key = (byte_start, byte_end)
        if span_key not in seen_spans:
            seen_spans.add(span_key)
            elements.append({
                "element_id": str(uuid.uuid4()),
                "category": "DIRECT_IDENTIFIER",
                "subcategory": "SSN",
                "sensitivity": "CRITICAL",
                "byte_offset_start": byte_start,
                "byte_offset_end": byte_end,
                "original_text": match.group(),
                "context": {
                    "data_subject_role": "PARTY_PRIMARY",
                    "jurisdiction": doc_jurisdiction,
                    "applicable_regulations": ["CCPA", "SOC2"],
                },
            })

    for match in _PHONE_PATTERN.finditer(text):
        prefix_bytes = text[:match.start()].encode("utf-8")
        phone_bytes = match.group().encode("utf-8")
        byte_start = len(prefix_bytes)
        byte_end = byte_start + len(phone_bytes)
        span_key = (byte_start, byte_end)
        if span_key not in seen_spans:
            seen_spans.add(span_key)
            elements.append({
                "element_id": str(uuid.uuid4()),
                "category": "DIRECT_IDENTIFIER",
                "subcategory": "PHONE_NUMBER",
                "sensitivity": "HIGH",
                "byte_offset_start": byte_start,
                "byte_offset_end": byte_end,
                "original_text": match.group(),
                "context": {
                    "data_subject_role": "PARTY_PRIMARY",
                    "jurisdiction": doc_jurisdiction,
                    "applicable_regulations": ["CCPA", "SOC2"],
                },
            })

    for match in _EMAIL_PATTERN.finditer(text):
        prefix_bytes = text[:match.start()].encode("utf-8")
        email_bytes = match.group().encode("utf-8")
        byte_start = len(prefix_bytes)
        byte_end = byte_start + len(email_bytes)
        span_key = (byte_start, byte_end)
        if span_key not in seen_spans:
            seen_spans.add(span_key)
            elements.append({
                "element_id": str(uuid.uuid4()),
                "category": "DIRECT_IDENTIFIER",
                "subcategory": "EMAIL",
                "sensitivity": "HIGH",
                "byte_offset_start": byte_start,
                "byte_offset_end": byte_end,
                "original_text": match.group(),
                "context": {
                    "data_subject_role": "PARTY_PRIMARY",
                    "jurisdiction": doc_jurisdiction,
                    "applicable_regulations": ["CCPA", "SOC2"],
                },
            })

    for match in _LICENSE_PLATE_PATTERN.finditer(text):
        prefix_bytes = text[:match.start()].encode("utf-8")
        lp_bytes = match.group().encode("utf-8")
        byte_start = len(prefix_bytes)
        byte_end = byte_start + len(lp_bytes)
        span_key = (byte_start, byte_end)
        if span_key not in seen_spans:
            seen_spans.add(span_key)
            elements.append({
                "element_id": str(uuid.uuid4()),
                "category": "DIRECT_IDENTIFIER",
                "subcategory": "LICENSE_PLATE",
                "sensitivity": "HIGH",
                "byte_offset_start": byte_start,
                "byte_offset_end": byte_end,
                "original_text": match.group(),
                "context": {
                    "data_subject_role": "PARTY_PRIMARY",
                    "jurisdiction": doc_jurisdiction,
                    "applicable_regulations": ["CCPA", "SOC2"],
                },
            })

    # --- Street address detection ---
    for match in _STREET_ADDRESS_PATTERN.finditer(text):
        addr_text = match.group(0)
        prefix_bytes = text[:match.start()].encode("utf-8")
        addr_bytes = addr_text.encode("utf-8")
        byte_start = len(prefix_bytes)
        byte_end = byte_start + len(addr_bytes)
        span_key = (byte_start, byte_end)
        # Check for overlaps with existing spans
        overlap = False
        for (es, ee) in seen_spans:
            if byte_start < ee and byte_end > es:
                overlap = True
                break
        if not overlap:
            seen_spans.add(span_key)
            elements.append({
                "element_id": str(uuid.uuid4()),
                "category": "QUASI_IDENTIFIER",
                "subcategory": "STREET_ADDRESS",
                "sensitivity": "HIGH",
                "byte_offset_start": byte_start,
                "byte_offset_end": byte_end,
                "original_text": addr_text,
                "context": {
                    "data_subject_role": "PARTY_PRIMARY",
                    "jurisdiction": doc_jurisdiction,
                    "applicable_regulations": ["CCPA", "SOC2"],
                },
            })

    # --- State+ZIP detection (captures full "Arizona 85254" or "AZ 85254") ---
    for match in _STATE_ZIP_PATTERN.finditer(text):
        full_text = match.group(1)   # "Arizona 85254" (state + ZIP)
        state_part = match.group(2)  # "Arizona" or "AZ"

        # Determine jurisdiction from the state part
        state_normalized = re.sub(r"\s+", " ", state_part.strip())
        if state_normalized in _STATE_TO_CODE:
            zip_jurisdiction = f"US_{_STATE_TO_CODE[state_normalized]}"
        elif state_normalized in _CODE_TO_STATE:
            zip_jurisdiction = f"US_{state_normalized}"
        else:
            zip_jurisdiction = doc_jurisdiction

        # Byte offsets for the FULL "State ZIP" string (group 1)
        prefix_bytes = text[:match.start(1)].encode("utf-8")
        full_bytes = full_text.encode("utf-8")
        byte_start = len(prefix_bytes)
        byte_end = byte_start + len(full_bytes)
        span_key = (byte_start, byte_end)
        overlap = False
        for (es, ee) in seen_spans:
            if byte_start < ee and byte_end > es:
                overlap = True
                break
        if not overlap:
            seen_spans.add(span_key)
            elements.append({
                "element_id": str(uuid.uuid4()),
                "category": "QUASI_IDENTIFIER",
                "subcategory": "ZIP_CODE",
                "sensitivity": "HIGH",
                "byte_offset_start": byte_start,
                "byte_offset_end": byte_end,
                "original_text": full_text,
                "context": {
                    "data_subject_role": "PARTY_PRIMARY",
                    "jurisdiction": zip_jurisdiction,
                    "applicable_regulations": ["CCPA", "SOC2"],
                },
            })

    # --- Standalone state names in address context ---
    # Catches state names like ", Arizona" or ", AZ" that appear after city names
    # but are NOT followed by a ZIP code (those are already captured above).
    # Excludes legal clause contexts: "laws of California", "State of Texas", etc.
    _LEGAL_CLAUSE_KEYWORDS = [
        "laws of", "state of", "governed by", "jurisdiction of",
        "county of", "court of", "pursuant to", "resident of",
        "filed in", "located in the state",
    ]
    for state_name, code in _STATE_TO_CODE.items():
        # Match ", StateName" NOT followed by a ZIP code
        pattern = re.compile(
            r",\s*\b(" + re.escape(state_name) + r")\b(?!\s+\d{5})"
        )
        for m in pattern.finditer(text):
            state_text = m.group(1)
            state_char_start = m.start(1)
            state_char_end = m.end(1)

            # Skip if in a legal clause context
            before_context = text[max(0, state_char_start - 120):state_char_start].lower()
            if any(kw in before_context for kw in _LEGAL_CLAUSE_KEYWORDS):
                continue

            # Compute byte offsets
            prefix_bytes = text[:state_char_start].encode("utf-8")
            state_bytes = state_text.encode("utf-8")
            byte_start = len(prefix_bytes)
            byte_end = byte_start + len(state_bytes)
            span_key = (byte_start, byte_end)

            overlap = False
            for (es, ee) in seen_spans:
                if byte_start < ee and byte_end > es:
                    overlap = True
                    break
            if not overlap:
                seen_spans.add(span_key)
                elements.append({
                    "element_id": str(uuid.uuid4()),
                    "category": "QUASI_IDENTIFIER",
                    "subcategory": "ADDRESS",
                    "sensitivity": "HIGH",
                    "byte_offset_start": byte_start,
                    "byte_offset_end": byte_end,
                    "original_text": state_text,
                    "context": {
                        "data_subject_role": "PARTY_PRIMARY",
                        "jurisdiction": f"US_{code}",
                        "applicable_regulations": ["CCPA", "SOC2"],
                    },
                })

    # Also detect 2-letter state abbreviations in address context: ", AZ" (not followed by ZIP)
    _STATE_ABBREV_ADDR_PATTERN = re.compile(
        r",\s*\b([A-Z]{2})\b(?!\s+\d{5})"
    )
    for m in _STATE_ABBREV_ADDR_PATTERN.finditer(text):
        abbrev = m.group(1)
        if abbrev not in _CODE_TO_STATE:
            continue

        abbrev_char_start = m.start(1)
        abbrev_char_end = m.end(1)

        # Skip if in a legal clause context
        before_context = text[max(0, abbrev_char_start - 120):abbrev_char_start].lower()
        if any(kw in before_context for kw in _LEGAL_CLAUSE_KEYWORDS):
            continue

        # Compute byte offsets
        prefix_bytes = text[:abbrev_char_start].encode("utf-8")
        abbrev_bytes = abbrev.encode("utf-8")
        byte_start = len(prefix_bytes)
        byte_end = byte_start + len(abbrev_bytes)
        span_key = (byte_start, byte_end)

        overlap = False
        for (es, ee) in seen_spans:
            if byte_start < ee and byte_end > es:
                overlap = True
                break
        if not overlap:
            seen_spans.add(span_key)
            elements.append({
                "element_id": str(uuid.uuid4()),
                "category": "QUASI_IDENTIFIER",
                "subcategory": "ADDRESS",
                "sensitivity": "HIGH",
                "byte_offset_start": byte_start,
                "byte_offset_end": byte_end,
                "original_text": abbrev,
                "context": {
                    "data_subject_role": "PARTY_PRIMARY",
                    "jurisdiction": f"US_{abbrev}",
                    "applicable_regulations": ["CCPA", "SOC2"],
                },
            })

    # --- 3. Name propagation: find all occurrences of detected PERSON names ---
    # spaCy often misses names in ALL CAPS or in certain syntactic positions.
    # We search for every occurrence of each detected name in the full text.
    person_elements = [e for e in elements if e["subcategory"] == "FULL_NAME"]
    known_names: set[str] = set()
    for pe in person_elements:
        name = pe["original_text"].strip()
        if len(name) > 2:
            known_names.add(name)

    for name in known_names:
        # Search for both original case and upper case
        for variant in {name, name.upper(), name.title()}:
            for m in re.finditer(re.escape(variant), text):
                prefix_bytes = text[:m.start()].encode("utf-8")
                name_bytes = m.group().encode("utf-8")
                byte_start = len(prefix_bytes)
                byte_end = byte_start + len(name_bytes)
                span_key = (byte_start, byte_end)
                if span_key not in seen_spans:
                    # Check for overlapping spans
                    overlap = False
                    for (es, ee) in seen_spans:
                        if byte_start < ee and byte_end > es:
                            overlap = True
                            break
                    if overlap:
                        continue
                    seen_spans.add(span_key)
                    person_count += 1
                    elements.append({
                        "element_id": str(uuid.uuid4()),
                        "category": "DIRECT_IDENTIFIER",
                        "subcategory": "FULL_NAME",
                        "sensitivity": "CRITICAL",
                        "byte_offset_start": byte_start,
                        "byte_offset_end": byte_end,
                        "original_text": m.group(),
                        "context": {
                            "data_subject_role": "PARTY_PRIMARY" if person_count % 2 == 1 else "PARTY_SECONDARY",
                            "jurisdiction": doc_jurisdiction,
                            "applicable_regulations": ["CCPA", "SOC2"],
                        },
                    })

    # --- 3b. Name-part propagation: find orphaned first/last name parts ---
    # After name propagation, some individual name parts may still be orphaned
    # (e.g., spaCy detected "DOROTHY ELLEN" but not "HUTCHINSON", and full-name
    # propagation didn't cover the standalone word).  Extract individual words
    # from all known multi-word names and search for them adjacent to existing
    # FULL_NAME spans.
    all_name_parts: set[str] = set()
    for e in elements:
        if e["subcategory"] == "FULL_NAME":
            words = e["original_text"].strip().split()
            if len(words) >= 2:
                for w in words:
                    w_clean = w.strip()
                    if len(w_clean) >= 3:
                        all_name_parts.add(w_clean.upper())

    # Current FULL_NAME byte spans (for adjacency check)
    name_byte_spans = [(e["byte_offset_start"], e["byte_offset_end"])
                        for e in elements if e["subcategory"] == "FULL_NAME"]

    for part_upper in all_name_parts:
        for variant in {part_upper, part_upper.title()}:
            for m in re.finditer(r'\b' + re.escape(variant) + r'\b', text):
                prefix_bytes = text[:m.start()].encode("utf-8")
                part_bytes = m.group().encode("utf-8")
                byte_start = len(prefix_bytes)
                byte_end = byte_start + len(part_bytes)
                span_key = (byte_start, byte_end)
                if span_key in seen_spans:
                    continue
                # Check for overlap with existing spans
                overlap = False
                for (es, ee) in seen_spans:
                    if byte_start < ee and byte_end > es:
                        overlap = True
                        break
                if overlap:
                    continue
                # Only include if adjacent (within 2 bytes) to a FULL_NAME span
                is_adjacent = False
                for (ps, pe) in name_byte_spans:
                    if 0 <= (ps - byte_end) <= 2 or 0 <= (byte_start - pe) <= 2:
                        is_adjacent = True
                        break
                if not is_adjacent:
                    continue
                seen_spans.add(span_key)
                person_count += 1
                elements.append({
                    "element_id": str(uuid.uuid4()),
                    "category": "DIRECT_IDENTIFIER",
                    "subcategory": "FULL_NAME",
                    "sensitivity": "CRITICAL",
                    "byte_offset_start": byte_start,
                    "byte_offset_end": byte_end,
                    "original_text": m.group(),
                    "context": {
                        "data_subject_role": "PARTY_PRIMARY" if person_count % 2 == 1 else "PARTY_SECONDARY",
                        "jurisdiction": doc_jurisdiction,
                        "applicable_regulations": ["CCPA", "SOC2"],
                    },
                })
                # Update spans so subsequent parts can chain
                name_byte_spans.append((byte_start, byte_end))

    # --- 3c. Signature-line name detection ---
    # ALL-CAPS person names that appear on their own line after signature lines
    # ("_____") are frequently missed by spaCy, especially with unusual surnames
    # or middle initials (e.g., "CAROL S. DIMITRIOU").  Detect these patterns.
    _SIG_LINE = re.compile(r'_{10,}', re.MULTILINE)
    _CAPS_NAME = re.compile(
        r'^[ \t]*([A-Z]{2,}(?:\s+[A-Z]\.?)?\s+[A-Z]{2,}(?:\s+[A-Z]{2,})*)[ \t]*$',
        re.MULTILINE,
    )
    sig_ends = [m.end() for m in _SIG_LINE.finditer(text)]

    for m in _CAPS_NAME.finditer(text):
        name_text = m.group(1).strip()
        name_pos = m.start()
        # Must be within 50 chars after a signature line
        near_sig = any(0 <= (name_pos - se) <= 50 for se in sig_ends)
        if not near_sig:
            continue
        # Skip if any substantive word matches a skip entity or state name
        words = [w for w in name_text.split() if len(w.rstrip('.')) >= 2]
        if len(words) < 2:
            continue
        if any(w.title() in _SKIP_ENTITIES or w.title() in _US_STATES for w in words):
            continue
        # Compute byte offsets
        name_char_start = m.start(1)
        name_char_end = m.end(1)
        prefix_bytes = text[:name_char_start].encode("utf-8")
        name_bytes = name_text.encode("utf-8")
        byte_start = len(prefix_bytes)
        byte_end = byte_start + len(name_bytes)
        span_key = (byte_start, byte_end)
        if span_key in seen_spans:
            continue
        overlap = False
        for (es, ee) in seen_spans:
            if byte_start < ee and byte_end > es:
                overlap = True
                break
        if overlap:
            continue
        seen_spans.add(span_key)
        person_count += 1
        elements.append({
            "element_id": str(uuid.uuid4()),
            "category": "DIRECT_IDENTIFIER",
            "subcategory": "FULL_NAME",
            "sensitivity": "CRITICAL",
            "byte_offset_start": byte_start,
            "byte_offset_end": byte_end,
            "original_text": name_text,
            "context": {
                "data_subject_role": "PARTY_PRIMARY" if person_count % 2 == 1 else "PARTY_SECONDARY",
                "jurisdiction": doc_jurisdiction,
                "applicable_regulations": ["CCPA", "SOC2"],
            },
        })

    # --- 4. Detect boilerplate sections ---
    for pattern in _BOILERPLATE_PATTERNS:
        for match in re.finditer(pattern, text):
            prefix_bytes = text[:match.start()].encode("utf-8")
            bp_bytes = match.group().encode("utf-8")
            byte_start = len(prefix_bytes)
            byte_end = byte_start + len(bp_bytes)
            span_key = (byte_start, byte_end)
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                elements.append({
                    "element_id": str(uuid.uuid4()),
                    "category": "BOILERPLATE",
                    "subcategory": "LEGAL_CLAUSE",
                    "sensitivity": "LOW",
                    "byte_offset_start": byte_start,
                    "byte_offset_end": byte_end,
                    "original_text": match.group(),
                    "context": {
                        "data_subject_role": "PARTY_PRIMARY",
                        "jurisdiction": doc_jurisdiction,
                        "applicable_regulations": [],
                    },
                })

    # --- Build summary ---
    critical = sum(1 for e in elements if e["sensitivity"] == "CRITICAL")
    high = sum(1 for e in elements if e["sensitivity"] == "HIGH")
    low = sum(1 for e in elements if e["sensitivity"] == "LOW")
    has_minor = any(e["category"] == "MINOR_INFORMATION" for e in elements)
    has_privilege = any(e["category"] == "PRIVILEGE_INDICATOR" for e in elements)

    return {
        "document_id": doc_id or str(uuid.uuid4()),
        "tenant_id": str(uuid.uuid4()),
        "classification_timestamp": "2026-03-07T12:00:00Z",
        "classifier_version": "simulated-spacy-1.0",
        "elements": elements,
        "summary": {
            "total_elements": len(elements),
            "critical_count": critical,
            "high_count": high,
            "low_count": low,
            "privilege_flag": has_privilege,
            "minor_data_present": has_minor,
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate simulated Cyera metadata from a plain text document."
    )
    parser.add_argument("document", help="Path to the plain text document")
    parser.add_argument("-o", "--output", help="Output JSON path (default: <document>_metadata.json)")
    parser.add_argument("--run", action="store_true", help="Also run the pipeline after generating metadata")
    parser.add_argument("--id", help="Document ID (default: auto-generated UUID)")
    args = parser.parse_args()

    doc_path = Path(args.document)
    if not doc_path.exists():
        print(f"Error: {doc_path} not found")
        sys.exit(1)

    text = doc_path.read_text(encoding="utf-8")
    print(f"Scanning {doc_path.name} ({len(text)} chars, {len(text.encode('utf-8'))} bytes)...")

    metadata = generate_metadata(text, doc_id=args.id)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = doc_path.with_name(doc_path.stem + "_metadata.json")

    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Generated {len(metadata['elements'])} elements -> {out_path.name}")

    # Print summary
    s = metadata["summary"]
    print(f"  Critical: {s['critical_count']}  High: {s['high_count']}  Low: {s['low_count']}")

    # Optionally run the pipeline
    if args.run:
        print("\nRunning pipeline...\n")
        from run_pipeline import run_one
        run_one(str(doc_path), str(out_path))


if __name__ == "__main__":
    main()
