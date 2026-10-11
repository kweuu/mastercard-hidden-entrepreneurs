"""Card-level behavioural features.

One row per card. Every feature describes how the card is used, never which
product it is (tier, bank and BIN are excluded on purpose).
"""
import numpy as np
import pandas as pd

from . import config

ALL_GROUPS = {**config.COMMERCIAL_GROUPS, **config.PERSONAL_GROUPS}


def _mcc_to_group(mcc: pd.Series) -> pd.Series:
    lookup = {code: name for name, codes in ALL_GROUPS.items() for code in codes}
    return mcc.map(lookup).fillna("other")


def _add_row_flags(tx: pd.DataFrame) -> pd.DataFrame:
    """Row-level helper columns that the card aggregation sums or averages."""
    ts = tx["transaction_timestamp"]
    amount = tx["transaction_amount_kzt"].astype(float)
    weekday = ts.dt.dayofweek < 5
    hour = ts.dt.hour

    flags = pd.DataFrame({
        "card_number": tx["card_number"].values,
        "amount": amount.values,
        "log_amount": np.log1p(amount.values),
        "group": _mcc_to_group(tx["mcc"]).values,
        "is_online": (tx["channel"].str.lower() == "online").values,
        "is_tokenized": tx["tokenized"].values,
        "is_recurring": tx["is_recurring"].values,
        "is_foreign_country": (tx["country"] != config.HOME_COUNTRY).values,
        "is_foreign_merchant": (~tx["merchant_country"].isin([config.HOME_COUNTRY, "Unknown"])).values,
        "is_weekday": weekday.values,
        "is_business_hours": (weekday & hour.between(config.BUSINESS_HOURS[0], config.BUSINESS_HOURS[1] - 1)).values,
        "is_evening": hour.between(18, 23).values,
        "is_large": (amount >= config.LARGE_TICKET_KZT).values,
        "is_very_large": (amount >= config.VERY_LARGE_TICKET_KZT).values,
        "date": ts.dt.normalize().values,
        "month": ts.dt.to_period("M").astype(str).values,
        "merchant_id": tx["merchant_id"].values,
        "mcc": tx["mcc"].values,
        "country": tx["country"].values,
    })
    flags["is_commercial"] = flags["group"].isin(config.COMMERCIAL_GROUPS).values
    return flags


def _merchant_structure(f: pd.DataFrame) -> pd.DataFrame:
    """Concentration and repeat-supplier features from card x merchant counts."""
    cm = f.groupby(["card_number", "merchant_id"], observed=True).agg(
        n=("amount", "size"), commercial=("is_commercial", "first"))
    totals = cm.groupby(level=0)["n"].transform("sum")
    share = cm["n"] / totals
    out = pd.DataFrame({
        "top_merchant_share": share.groupby(level=0).max(),
        "merchant_hhi": (share ** 2).groupby(level=0).sum(),
        # share of transactions at merchants the card visits 3+ times
        "repeat_merchant_tx_share": (cm["n"].where(cm["n"] >= 3, 0).groupby(level=0).sum()
                                     / cm["n"].groupby(level=0).sum()),
        # commercial merchants visited 3+ times: suppliers, ad accounts, couriers
        "n_repeat_commercial_merchants": ((cm["n"] >= 3) & cm["commercial"]).groupby(level=0).sum(),
        "n_commercial_merchants": cm["commercial"].groupby(level=0).sum(),
    })
    return out


def _group_shares(f: pd.DataFrame) -> pd.DataFrame:
    """Share of transactions and of spend per MCC group."""
    by_count = pd.crosstab(f["card_number"], f["group"], normalize="index")
    spend = f.pivot_table(index="card_number", columns="group", values="amount",
                          aggfunc="sum", fill_value=0)
    by_spend = spend.div(spend.sum(axis=1), axis=0)
    groups = list(ALL_GROUPS)
    by_count = by_count.reindex(columns=groups, fill_value=0).add_prefix("txshare_")
    by_spend = by_spend.reindex(columns=groups, fill_value=0).add_prefix("spendshare_")
    return by_count.join(by_spend)


def _monthly_regularity(f: pd.DataFrame) -> pd.DataFrame:
    monthly = f.pivot_table(index="card_number", columns="month", values="amount",
                            aggfunc="sum", fill_value=0)
    mean = monthly.mean(axis=1)
    cols = sorted(monthly.columns)
    half = len(cols) // 2
    return pd.DataFrame({
        "active_months": (monthly > 0).sum(axis=1),
        "monthly_spend_cv": monthly.std(axis=1) / mean.replace(0, np.nan),
        # >1 means spend is growing in the second half of the period
        "spend_trend_ratio": (monthly[cols[half:]].sum(axis=1) + 1) / (monthly[cols[:half]].sum(axis=1) + 1),
    })


def build_card_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned transactions to one feature row per card."""
    f = _add_row_flags(tx)
    g = f.groupby("card_number", observed=True)

    base = g.agg(
        n_tx=("amount", "size"),
        total_spend=("amount", "sum"),
        mean_ticket=("amount", "mean"),
        median_ticket=("amount", "median"),
        p90_ticket=("amount", lambda s: s.quantile(0.9)),
        max_ticket=("amount", "max"),
        log_ticket_std=("log_amount", "std"),
        large_tx_share=("is_large", "mean"),
        very_large_tx_share=("is_very_large", "mean"),
        active_days=("date", "nunique"),
        n_merchants=("merchant_id", "nunique"),
        n_mcc=("mcc", "nunique"),
        n_countries=("country", "nunique"),
        online_share=("is_online", "mean"),
        tokenized_share=("is_tokenized", "mean"),
        recurring_share=("is_recurring", "mean"),
        foreign_country_share=("is_foreign_country", "mean"),
        foreign_merchant_share=("is_foreign_merchant", "mean"),
        weekday_share=("is_weekday", "mean"),
        business_hours_share=("is_business_hours", "mean"),
        evening_share=("is_evening", "mean"),
        commercial_tx_count=("is_commercial", "sum"),
        commercial_tx_share=("is_commercial", "mean"),
    )

    com = f[f["is_commercial"]]
    gc = com.groupby("card_number", observed=True)
    commercial = pd.DataFrame({
        "commercial_spend": gc["amount"].sum(),
        "commercial_median_ticket": gc["amount"].median(),
        "commercial_business_hours_share": gc["is_business_hours"].mean(),
        "n_commercial_groups": gc["group"].nunique(),
        "n_recurring_commercial_merchants": com[com["is_recurring"]].groupby("card_number")["merchant_id"].nunique(),
    })

    feats = (base.join(commercial)
             .join(_merchant_structure(f))
             .join(_group_shares(f))
             .join(_monthly_regularity(f)))

    # cards with no commercial spend get zeros, not NaN
    zero_fill = ["commercial_spend", "commercial_median_ticket", "n_commercial_groups",
                 "n_recurring_commercial_merchants"]
    feats[zero_fill] = feats[zero_fill].fillna(0)
    feats["commercial_business_hours_share"] = feats["commercial_business_hours_share"].fillna(0)
    feats["monthly_spend_cv"] = feats["monthly_spend_cv"].fillna(0)
    feats["log_ticket_std"] = feats["log_ticket_std"].fillna(0)

    feats["tx_per_active_day"] = feats["n_tx"] / feats["active_days"]
    feats["commercial_spend_share"] = feats["commercial_spend"] / feats["total_spend"]
    for col in ["total_spend", "commercial_spend", "mean_ticket", "median_ticket",
                "p90_ticket", "max_ticket", "commercial_median_ticket"]:
        feats[f"log_{col}"] = np.log1p(feats[col])

    label = tx.groupby("card_number", observed=True)["is_business"].first()
    feats["is_business"] = label.reindex(feats.index).astype(int)
    return feats


# --- Feature sets ------------------------------------------------------------
# "Commercial footprint" set: describes how much business-like activity a card
# shows. It deliberately ignores how much a card spends on groceries, cafes or
# clothes: a hidden entrepreneur still lives a normal life on the same card,
# so "no groceries" must not be what the model learns.
COMMERCIAL_FEATURES = [
    "n_tx", "tx_per_active_day", "active_days", "active_months",
    "log_total_spend", "log_median_ticket", "log_p90_ticket", "log_max_ticket", "log_ticket_std",
    "large_tx_share", "very_large_tx_share",
    "n_merchants", "top_merchant_share", "merchant_hhi", "repeat_merchant_tx_share",
    "online_share", "tokenized_share", "recurring_share", "foreign_merchant_share", "n_countries",
    "weekday_share", "business_hours_share", "evening_share",
    "commercial_tx_count", "commercial_tx_share", "log_commercial_spend", "commercial_spend_share",
    "log_commercial_median_ticket", "commercial_business_hours_share", "n_commercial_groups",
    "n_commercial_merchants", "n_repeat_commercial_merchants", "n_recurring_commercial_merchants",
    "monthly_spend_cv", "spend_trend_ratio",
] + [f"spendshare_{g}" for g in config.COMMERCIAL_GROUPS] + [f"txshare_{g}" for g in config.COMMERCIAL_GROUPS]

PERSONAL_FEATURES = [f"spendshare_{g}" for g in config.PERSONAL_GROUPS] + \
                    [f"txshare_{g}" for g in config.PERSONAL_GROUPS] + ["n_mcc", "foreign_country_share"]

# Benchmark set with personal-life features: scores near-perfectly on
# business vs consumer, but learns the "no groceries" shortcut.
FULL_FEATURES = COMMERCIAL_FEATURES + PERSONAL_FEATURES

# The model uses the full commercial footprint. Dropping the spend-share
# features (the ones the labelling rule reads) was tested: it removes the
# circularity but makes the model blind to cards with 10-20% business
# activity (spike-in detection fell from 66% to 0%). Circularity is checked
# instead by the MCC-blind model below.
MODEL_FEATURES = COMMERCIAL_FEATURES

# MCC-blind set: no information about what is bought, only how. Used as a
# sanity check that the labels describe real behaviour, not an MCC artefact.
MCC_BLIND_FEATURES = [
    "n_tx", "tx_per_active_day", "active_days", "active_months", "log_total_spend",
    "log_median_ticket", "log_p90_ticket", "log_max_ticket", "log_ticket_std",
    "large_tx_share", "very_large_tx_share", "n_merchants", "top_merchant_share", "merchant_hhi",
    "repeat_merchant_tx_share", "online_share", "tokenized_share", "recurring_share",
    "foreign_merchant_share", "n_countries", "weekday_share", "business_hours_share",
    "evening_share", "monthly_spend_cv", "spend_trend_ratio",
]
