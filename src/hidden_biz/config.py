"""Paths, constants and the domain dictionary of MCC groups.

MCC groups are defined from the ISO 18245 meaning of each code, not from the
target. That keeps the features explainable to a business audience and avoids
leaking the business/consumer label into the feature definitions.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

BUSINESS_FILE = DATA_RAW / "business_cards_MDQ.parquet"
CONSUMER_FILE = DATA_RAW / "consumer_cards_MDQ.parquet"
MERCHANTS_FILE = DATA_RAW / "merchants_reference.parquet"

RANDOM_STATE = 42
HOME_COUNTRY = "Kazakhstan"
LARGE_TICKET_KZT = 100_000
VERY_LARGE_TICKET_KZT = 500_000
BUSINESS_HOURS = (9, 18)  # [start, end) local time, Monday to Friday

# --- Commercial footprint: spend that a household rarely needs -------------
COMMERCIAL_GROUPS = {
    # paid traffic: Google/Meta/TikTok/Yandex ads, Instagram promote
    "ads": {"7311"},
    # software, cloud and hosting used to run a business
    "saas_cloud": {"7372", "5968", "4816"},
    # B2B distribution (ISO 50xx/51xx wholesale ranges) and fuel wholesale
    "wholesale": {"5021", "5044", "5045", "5046", "5065", "5085", "5099",
                  "5111", "5131", "5172", "5199"},
    # retail categories used to buy stock for resale (building materials, food)
    "stock_retail": {"5211", "5499"},
    # freight, courier and warehousing
    "logistics": {"4214", "4215", "4225"},
    # professional services: consulting, legal, accounting, staffing, IT repair
    "prof_services": {"7338", "7361", "7379", "7392", "7399", "8111", "8911", "8931"},
    # office supplies and telecom equipment
    "office": {"5943", "4812"},
}

# --- Personal life: used for archetype profiling and the "full" benchmark --
PERSONAL_GROUPS = {
    "grocery": {"5411", "5422", "5441", "5451", "5462"},
    "eating_out": {"5812", "5813", "5814"},
    "fuel_retail": {"5541", "5542"},
    "transport": {"4111", "4121"},
    "pharmacy_health": {"5912", "8011", "8021", "8062", "8099"},
    "clothing": {"5611", "5621", "5631", "5641", "5651", "5655", "5661", "5691", "5699"},
    "entertainment": {"7832", "7841", "7995", "7996", "7997"},
    "travel": {"4511", "7011", "4722", "7512"},
    "telecom_utilities": {"4814", "4899", "4900"},
}

# Columns that must never reach a model: they identify the product, not the behaviour
LEAKAGE_COLUMNS = ["card_number", "card_tier", "bank_name"]
