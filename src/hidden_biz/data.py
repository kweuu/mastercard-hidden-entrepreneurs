"""Loading, validation and cleaning of the transaction files."""
import logging

import pandas as pd

from . import config

log = logging.getLogger(__name__)

TX_COLUMNS = [
    "transaction_timestamp", "transaction_amount_kzt", "mcc", "merchant_id",
    "channel", "country", "card_number", "tokenized", "is_recurring",
]


def load_merchants() -> pd.DataFrame:
    merchants = pd.read_parquet(config.MERCHANTS_FILE)
    merchants["mcc"] = merchants["mcc"].astype(str).str.zfill(4)
    return merchants


def _load_segment(path, is_business: int) -> pd.DataFrame:
    tx = pd.read_parquet(path, columns=TX_COLUMNS)
    tx["is_business"] = is_business
    return tx


def clean_transactions(tx: pd.DataFrame, merchants: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the data-quality rules and return the cleaned frame plus a log of fixes."""
    report = {"rows_in": len(tx)}

    # 1. Exact duplicates (none expected, checked anyway)
    dup = tx.duplicated()
    report["duplicates_removed"] = int(dup.sum())
    tx = tx.loc[~dup]

    # 2. Missing values: drop rows without the keys, fill booleans
    key_cols = ["card_number", "merchant_id", "transaction_timestamp", "transaction_amount_kzt"]
    missing = tx[key_cols].isna().any(axis=1)
    report["rows_missing_keys_removed"] = int(missing.sum())
    tx = tx.loc[~missing]
    for col in ["tokenized", "is_recurring"]:
        report[f"{col}_nulls_filled"] = int(tx[col].isna().sum())
        tx[col] = tx[col].fillna(False).astype(bool)

    # 3. Non-positive amounts are reversals or errors, not spend
    bad_amount = tx["transaction_amount_kzt"] <= 0
    report["non_positive_amounts_removed"] = int(bad_amount.sum())
    tx = tx.loc[~bad_amount]

    # 4. MCC reconciliation. Some ad-platform rows carry MCC 7012 (timeshares)
    #    although the merchant master says 7311. Merchant master wins.
    ref = merchants.set_index("merchant_id")
    ref_mcc = tx["merchant_id"].map(ref["mcc"])
    mismatch = ref_mcc.notna() & (ref_mcc != tx["mcc"])
    report["mcc_mismatches_fixed"] = int(mismatch.sum())
    report["mcc_mismatch_examples"] = (
        tx.loc[mismatch, "mcc"].astype(str) + "->" + ref_mcc[mismatch].astype(str)
    ).value_counts().head(5).to_dict()
    tx["mcc"] = ref_mcc.fillna(tx["mcc"])
    tx["merchant_country"] = tx["merchant_id"].map(ref["merchant_country"]).fillna("Unknown")
    report["unknown_merchants"] = int((tx["merchant_country"] == "Unknown").sum())

    # 5. Outliers: keep the rows (a 40M KZT purchase is plausible for a business)
    #    but flag them; card-level features use medians, quantiles and log scales.
    cap = tx["transaction_amount_kzt"].quantile(0.999)
    report["amount_p999_kzt"] = float(cap)
    report["rows_above_p999"] = int((tx["transaction_amount_kzt"] > cap).sum())

    # 6. Midnight-only timestamps (no time part) would distort hour features
    ts = tx["transaction_timestamp"]
    report["midnight_only_timestamps"] = int(((ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)).sum())

    report["rows_out"] = len(tx)
    return tx.reset_index(drop=True), report


def load_transactions() -> tuple[pd.DataFrame, dict]:
    """Business + consumer transactions, cleaned, with an is_business label per row."""
    merchants = load_merchants()
    business = _load_segment(config.BUSINESS_FILE, 1)
    consumer = _load_segment(config.CONSUMER_FILE, 0)

    overlap = set(business["card_number"]) & set(consumer["card_number"])
    if overlap:
        log.warning("%d cards appear in both files; keeping the business label", len(overlap))
        consumer = consumer[~consumer["card_number"].isin(overlap)]

    tx = pd.concat([business, consumer], ignore_index=True)
    for col in ["mcc", "merchant_id", "channel", "country", "card_number"]:
        tx[col] = tx[col].astype(str)
    tx, report = clean_transactions(tx, merchants)
    report["business_cards"] = int(business["card_number"].nunique())
    report["consumer_cards"] = int(consumer["card_number"].nunique())
    return tx, report
