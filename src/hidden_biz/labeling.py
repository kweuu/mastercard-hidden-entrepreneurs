"""Weak supervision: turn unlabeled consumer cards into silver labels.

There is no "hidden entrepreneur" flag in the data, so labels are built from
two independent views of the same card and kept only where both agree:

  View A (unsupervised): K-Means personas on the full behavioural profile.
                         A persona is commercial if its median card sends at
                         least 30% of spend into commercial MCC groups.
  View B (expert rule):  at least 25% of spend in commercial MCC groups and
                         2+ commercial merchants used repeatedly (suppliers,
                         ad accounts, couriers). Below 10% = clearly personal.

Cards where the views disagree stay unlabeled ("ambiguous"); the model scores
them later. This is the confident-learning idea: train on clean labels only.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FULL_FEATURES

LOG_COLUMNS = [
    "n_tx", "active_days", "n_merchants", "commercial_tx_count", "n_commercial_merchants",
    "n_repeat_commercial_merchants", "n_recurring_commercial_merchants", "n_countries",
    "n_mcc", "tx_per_active_day", "spend_trend_ratio",
]
PERSONA_COMMERCIAL_SHARE = 0.30
RULE_POSITIVE_SHARE = 0.25
RULE_NEGATIVE_SHARE = 0.10
RULE_MIN_REPEAT_SUPPLIERS = 2

GROUP_TITLES = {
    "ads": "Online seller / digital marketer",
    "saas_cloud": "IT freelancer",
    "wholesale": "Wholesale trader",
    "stock_retail": "Reseller buying stock at retail",
    "logistics": "Delivery & logistics trader",
    "prof_services": "Service business",
    "office": "Office-based business",
}


def _scaled_matrix(cards: pd.DataFrame) -> np.ndarray:
    X = cards[FULL_FEATURES].copy()
    X[LOG_COLUMNS] = np.log1p(X[LOG_COLUMNS])
    return StandardScaler().fit_transform(X)


def choose_k(Z: np.ndarray, candidates=range(6, 13), sample=15_000) -> pd.Series:
    """Silhouette on a sample for each k (reported, used to pick k)."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(len(Z), min(sample, len(Z)), replace=False)
    scores = {}
    for k in candidates:
        labels = KMeans(k, n_init=5, random_state=config.RANDOM_STATE).fit_predict(Z)
        scores[k] = silhouette_score(Z[idx], labels[idx])
    return pd.Series(scores, name="silhouette")


def segment_consumers(consumers: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """K-Means personas for consumer cards plus an auto-generated persona name."""
    Z = _scaled_matrix(consumers)
    labels = KMeans(k, n_init=10, random_state=config.RANDOM_STATE).fit_predict(Z)
    out = pd.DataFrame({"persona_id": labels}, index=consumers.index)

    profile = consumers.groupby(labels).median(numeric_only=True)
    commercial = profile["commercial_spend_share"] >= PERSONA_COMMERCIAL_SHARE
    names = {}
    for pid, row in profile.iterrows():
        if not commercial[pid]:
            names[pid] = "Traveller / affluent" if row["foreign_country_share"] > 0.35 else "Everyday consumer"
            continue
        top_group = max(config.COMMERCIAL_GROUPS, key=lambda g: row[f"txshare_{g}"])
        personal = row[[f"txshare_{g}" for g in config.PERSONAL_GROUPS]].sum()
        style = "dedicated card" if personal < 0.05 else "mixed with personal spend"
        names[pid] = f"{GROUP_TITLES[top_group]} ({style})"
    # several clusters can share a name ("Everyday consumer"): suffix A, B, C by size
    sizes = pd.Series(labels).value_counts()
    for name in set(names.values()):
        same = sorted([p for p, n in names.items() if n == name], key=lambda p: -sizes[p])
        if len(same) > 1:
            for letter, pid in zip("ABCDEFGH", same):
                names[pid] = f"{name} {letter}"
    out["persona"] = out["persona_id"].map(names)
    out["persona_is_commercial"] = out["persona_id"].map(commercial)
    return out


def make_silver_labels(consumers: pd.DataFrame, personas: pd.DataFrame) -> pd.DataFrame:
    rule_pos = ((consumers["commercial_spend_share"] >= RULE_POSITIVE_SHARE)
                & (consumers["n_repeat_commercial_merchants"] >= RULE_MIN_REPEAT_SUPPLIERS))
    rule_neg = consumers["commercial_spend_share"] < RULE_NEGATIVE_SHARE
    view_a = personas["persona_is_commercial"]

    silver = pd.Series(np.nan, index=consumers.index, name="silver_label")
    silver[view_a & rule_pos] = 1
    silver[~view_a & rule_neg] = 0
    out = personas.join(silver)
    out["rule_view"] = np.select([rule_pos, rule_neg], ["commercial", "personal"], "in-between")
    return out
