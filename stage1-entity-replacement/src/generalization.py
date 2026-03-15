"""Generalization functions for reducing PII to safe categorical tokens."""

from __future__ import annotations

import math
import os
import re
from datetime import date

import yaml
from dateutil import parser as dateutil_parser

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "config")

# ---------------------------------------------------------------------------
# Config loading (lazy, cached)
# ---------------------------------------------------------------------------

_metro_cities: dict[str, list[str]] | None = None
_gen_rules: dict | None = None


def _load_metro_cities() -> dict[str, list[str]]:
    global _metro_cities
    if _metro_cities is None:
        path = os.path.join(_CONFIG_DIR, "us_metro_cities.yaml")
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        # Normalize city names to lowercase for matching
        _metro_cities = {}
        for state, cities in raw.items():
            _metro_cities[state] = [c.lower() for c in (cities or [])]
    return _metro_cities


def _load_gen_rules() -> dict:
    global _gen_rules
    if _gen_rules is None:
        path = os.path.join(_CONFIG_DIR, "generalization_rules.yaml")
        with open(path, encoding="utf-8") as f:
            _gen_rules = yaml.safe_load(f)
    return _gen_rules


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generalize_date(date_str: str) -> str:
    """Reduce a date to its quarter.

    Returns ``[DATE_REDACTED]`` for strings that cannot be parsed as
    calendar dates (e.g. durations like "30 days").

    >>> generalize_date("March 15, 2026")
    '[DATE_2026_Q1]'
    >>> generalize_date("11/03/2025")
    '[DATE_2025_Q4]'
    >>> generalize_date("30 days")
    '[DATE_REDACTED]'
    """
    try:
        dt = dateutil_parser.parse(date_str, dayfirst=False)
    except (ValueError, OverflowError):
        return "[DATE_REDACTED]"
    quarter = (dt.month - 1) // 3 + 1
    return f"[DATE_{dt.year}_Q{quarter}]"


def generalize_dob(dob_str: str, reference_date: date | None = None) -> str:
    """Convert a date of birth to a 5-year age range.

    >>> generalize_dob("03/15/1988", reference_date=date(2026, 1, 1))
    '[AGE_RANGE_35_40]'
    """
    try:
        dt = dateutil_parser.parse(dob_str, dayfirst=False).date()
    except (ValueError, OverflowError):
        return "[AGE_REDACTED]"
    ref = reference_date or date.today()
    age = ref.year - dt.year - ((ref.month, ref.day) < (dt.month, dt.day))
    rules = _load_gen_rules()
    bucket = rules["age_range"]["bucket_size"]
    lower = (age // bucket) * bucket
    upper = lower + bucket
    return f"[AGE_RANGE_{lower}_{upper}]"


def generalize_address(address: str, jurisdiction: str) -> str:
    """Reduce an address to a state + metro/rural token.

    The state code is extracted from the *jurisdiction* field (e.g. ``US_CA``
    → ``CA``), **not** parsed from the address string.  Metro vs rural is
    determined by checking whether any city in the address matches the static
    metro-cities lookup.

    >>> generalize_address("142 Oak St, Sacramento, CA 95814", "US_CA")
    '[LOCATION_CA_METRO]'
    >>> generalize_address("12 Elm Rd, Smalltown, WV 25000", "US_WV")
    '[LOCATION_WV_RURAL]'
    """
    # Extract state code from jurisdiction
    state = _extract_state(jurisdiction)
    metro_cities = _load_metro_cities()
    city_list = metro_cities.get(state, [])

    # Try to match any metro city name in the address (case-insensitive)
    addr_lower = address.lower()
    is_metro = any(city in addr_lower for city in city_list)
    designation = "METRO" if is_metro else "RURAL"
    return f"[LOCATION_{state}_{designation}]"


def bucket_financial(value: float | str, category: str) -> str:
    """Bucket a financial value into a range.

    >>> bucket_financial("$4,200/month", "RENT")
    '[RENT_4000_5000]'
    >>> bucket_financial(375000, "PURCHASE_PRICE")
    '[PRICE_350K_400K]'
    """
    numeric = _parse_currency(value)
    rules = _load_gen_rules()
    bucket_cfg = rules["financial_buckets"].get(category)
    if bucket_cfg is None:
        # Fallback: $1000 buckets
        bucket_size = 1000
        lower = (int(numeric) // bucket_size) * bucket_size
        upper = lower + bucket_size
        return f"[FINANCIAL_{lower}_{upper}]"

    bucket_size = bucket_cfg["bucket_size"]
    fmt = bucket_cfg["format"]
    lower = (int(numeric) // bucket_size) * bucket_size
    upper = lower + bucket_size

    # Format string substitution
    token = fmt.format(
        lower=lower,
        upper=upper,
        lower_k=lower // 1000,
        upper_k=upper // 1000,
    )
    return f"[{token}]"


def generalize_employer(employer: str, industry: str | None = None) -> str:
    """Replace an employer with an industry token or numbered fallback.

    >>> generalize_employer("Acme Corp", "TECH")
    '[EMPLOYER_TECH]'
    >>> generalize_employer("Some Firm", None)
    '[EMPLOYER_UNKNOWN]'
    """
    if industry:
        return f"[EMPLOYER_{industry.upper()}]"
    return "[EMPLOYER_UNKNOWN]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_state(jurisdiction: str) -> str:
    """Extract a two-letter US state code from a jurisdiction string.

    Handles formats like ``US_CA``, ``US-CA``, ``CA``.
    """
    parts = re.split(r"[_\-]", jurisdiction)
    if len(parts) >= 2:
        return parts[-1].upper()
    return jurisdiction.upper()


def _parse_currency(value: float | str) -> float:
    """Parse a currency string like ``$4,200/month`` into a float."""
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^\d.]", "", str(value))
    if not cleaned:
        return 0.0
    return float(cleaned)
