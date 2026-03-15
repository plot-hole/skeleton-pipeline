"""Tests for generalization functions."""

from datetime import date

import pytest

from src.generalization import (
    bucket_financial,
    generalize_address,
    generalize_date,
    generalize_dob,
    generalize_employer,
)


# ---------------------------------------------------------------------------
# generalize_date
# ---------------------------------------------------------------------------

class TestGeneralizeDate:
    def test_march_is_q1(self):
        assert generalize_date("March 15, 2026") == "[DATE_2026_Q1]"

    def test_november_is_q4(self):
        assert generalize_date("November 3, 2025") == "[DATE_2025_Q4]"

    def test_january_is_q1(self):
        assert generalize_date("01/01/2024") == "[DATE_2024_Q1]"

    def test_april_is_q2(self):
        assert generalize_date("April 1, 2026") == "[DATE_2026_Q2]"

    def test_june_is_q2(self):
        assert generalize_date("06/30/2025") == "[DATE_2025_Q2]"

    def test_july_is_q3(self):
        assert generalize_date("July 4, 2026") == "[DATE_2026_Q3]"

    def test_september_is_q3(self):
        assert generalize_date("09/30/2025") == "[DATE_2025_Q3]"

    def test_october_is_q4(self):
        assert generalize_date("10/15/2024") == "[DATE_2024_Q4]"

    def test_iso_format(self):
        assert generalize_date("2026-03-15") == "[DATE_2026_Q1]"


# ---------------------------------------------------------------------------
# generalize_dob
# ---------------------------------------------------------------------------

class TestGeneralizeDob:
    def test_age_37_bucket_35_40(self):
        # Born 1988, reference 2026 -> age 37 -> bucket 35-40
        assert generalize_dob("03/15/1988", reference_date=date(2026, 1, 1)) == "[AGE_RANGE_35_40]"

    def test_age_0_bucket_0_5(self):
        assert generalize_dob("06/01/2025", reference_date=date(2026, 1, 1)) == "[AGE_RANGE_0_5]"

    def test_age_exact_boundary(self):
        # Born 1986, reference 2026 -> age 40 -> bucket 40-45
        assert generalize_dob("01/01/1986", reference_date=date(2026, 6, 1)) == "[AGE_RANGE_40_45]"

    def test_age_before_birthday(self):
        # Born Dec 1988, reference Jan 2026 -> still 37 -> bucket 35-40
        assert generalize_dob("12/15/1988", reference_date=date(2026, 1, 1)) == "[AGE_RANGE_35_40]"

    def test_age_65_bucket_65_70(self):
        assert generalize_dob("05/20/1960", reference_date=date(2026, 6, 1)) == "[AGE_RANGE_65_70]"


# ---------------------------------------------------------------------------
# generalize_address
# ---------------------------------------------------------------------------

class TestGeneralizeAddress:
    def test_sacramento_metro(self):
        assert generalize_address(
            "142 Oak St, Sacramento, CA 95814", "US_CA"
        ) == "[LOCATION_CA_METRO]"

    def test_smalltown_rural(self):
        assert generalize_address(
            "12 Elm Rd, Smalltown, WV 25000", "US_WV"
        ) == "[LOCATION_WV_RURAL]"

    def test_springfield_il_metro(self):
        assert generalize_address(
            "89 Main St, Springfield, IL 62701", "US_IL"
        ) == "[LOCATION_IL_METRO]"

    def test_chicago_metro(self):
        assert generalize_address(
            "1 N State St, Chicago, IL 60602", "US_IL"
        ) == "[LOCATION_IL_METRO]"

    def test_jurisdiction_dash_format(self):
        assert generalize_address(
            "100 Main St, Austin, TX 78701", "US-TX"
        ) == "[LOCATION_TX_METRO]"

    def test_bare_state_code(self):
        assert generalize_address(
            "100 Main St, Austin, TX 78701", "TX"
        ) == "[LOCATION_TX_METRO]"

    def test_unknown_city_rural(self):
        assert generalize_address(
            "1 Road, Nowhere, MT 59000", "US_MT"
        ) == "[LOCATION_MT_RURAL]"


# ---------------------------------------------------------------------------
# bucket_financial
# ---------------------------------------------------------------------------

class TestBucketFinancial:
    def test_rent_4200(self):
        assert bucket_financial("$4,200/month", "RENT") == "[RENT_4000_5000]"

    def test_rent_numeric(self):
        assert bucket_financial(4200, "RENT") == "[RENT_4000_5000]"

    def test_rent_low(self):
        assert bucket_financial(800, "RENT") == "[RENT_0_1000]"

    def test_income_7500(self):
        assert bucket_financial("$7,500", "INCOME") == "[INCOME_7000_8000]"

    def test_purchase_price_375k(self):
        assert bucket_financial(375000, "PURCHASE_PRICE") == "[PRICE_350K_400K]"

    def test_purchase_price_string(self):
        assert bucket_financial("$375,000", "PURCHASE_PRICE") == "[PRICE_350K_400K]"

    def test_settlement_60k(self):
        assert bucket_financial(60000, "SETTLEMENT") == "[SETTLEMENT_50K_75K]"

    def test_unknown_category_fallback(self):
        result = bucket_financial(1500, "OTHER")
        assert result == "[FINANCIAL_1000_2000]"


# ---------------------------------------------------------------------------
# generalize_employer
# ---------------------------------------------------------------------------

class TestGeneralizeEmployer:
    def test_with_industry(self):
        assert generalize_employer("Acme Corp", "TECH") == "[EMPLOYER_TECH]"

    def test_with_lowercase_industry(self):
        assert generalize_employer("Acme Corp", "tech") == "[EMPLOYER_TECH]"

    def test_without_industry(self):
        assert generalize_employer("Some Firm") == "[EMPLOYER_UNKNOWN]"

    def test_none_industry(self):
        assert generalize_employer("Some Firm", None) == "[EMPLOYER_UNKNOWN]"

    def test_empty_industry(self):
        assert generalize_employer("Some Firm", "") == "[EMPLOYER_UNKNOWN]"
