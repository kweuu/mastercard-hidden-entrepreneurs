"""Checks that do not depend on the silver labels.

Silver labels come from our own rules, so a high test score only proves the
model reproduces them. These three checks test the model against evidence it
never saw:

1. Real business cards: the model is trained on consumers only. A good
   commercial-footprint detector should still flag almost every business card.
2. Spike-in: take an ordinary consumer and mix in a share of a real business
   card's transactions. This simulates "running a business through a personal
   card" and measures how much business activity the model needs to notice it.
3. Temporal stability: score Oct-Dec and Jan-Mar separately. A real
   entrepreneur stays an entrepreneur; a noisy model reshuffles its list.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config
from .features import build_card_features


def business_card_recall(model, business_features: pd.DataFrame, feature_cols, threshold) -> dict:
    proba = model.predict_proba(business_features[feature_cols])[:, 1]
    return {"business_cards": len(proba), "flagged_share": float((proba >= threshold).mean()),
            "median_score": float(np.median(proba))}


def spike_in_test(tx, model, feature_cols, threshold, consumer_ids, business_ids,
                  fractions=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5), n_pairs=1000) -> pd.DataFrame:
    """Detection rate of hybrid cards as a function of the business share of transactions."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    c_ids = rng.choice(np.asarray(consumer_ids), n_pairs, replace=False)
    b_ids = rng.choice(np.asarray(business_ids), n_pairs, replace=False)
    c_tx = dict(tuple(tx[tx["card_number"].isin(c_ids)].groupby("card_number")))
    b_tx = dict(tuple(tx[tx["card_number"].isin(b_ids)].groupby("card_number")))

    pieces = []
    for frac in fractions:
        for i, (c, b) in enumerate(zip(c_ids, b_ids)):
            own = c_tx[c]
            n_extra = int(round(frac / (1 - frac) * len(own)))
            extra = b_tx[b].sample(n_extra, replace=n_extra > len(b_tx[b]),
                                   random_state=int(rng.integers(1e9))) if n_extra else own.iloc[:0]
            hybrid = pd.concat([own, extra]).assign(card_number=f"H{frac:.2f}_{i}", is_business=0)
            pieces.append(hybrid)
    hybrids = pd.concat(pieces, ignore_index=True)
    feats = build_card_features(hybrids)
    proba = model.predict_proba(feats[feature_cols])[:, 1]
    result = pd.DataFrame({"fraction": feats.index.str.slice(1, 5).astype(float), "proba": proba})
    return (result.groupby("fraction")
            .agg(detection_rate=("proba", lambda p: (p >= threshold).mean()),
                 median_score=("proba", "median"), cards=("proba", "size"))
            .reset_index())


def temporal_stability(tx, model, feature_cols, threshold, n_flagged: int,
                       split_date="2026-01-01") -> dict:
    """Score each half-year separately and compare the two flagged lists."""
    consumers = tx[tx["is_business"] == 0]
    first = consumers[consumers["transaction_timestamp"] < split_date]
    second = consumers[consumers["transaction_timestamp"] >= split_date]
    f1, f2 = build_card_features(first), build_card_features(second)
    s1 = pd.Series(model.predict_proba(f1[feature_cols])[:, 1], index=f1.index)
    s2 = pd.Series(model.predict_proba(f2[feature_cols])[:, 1], index=f2.index)
    common = s1.index.intersection(s2.index)
    s1, s2 = s1[common], s2[common]

    # Half-year cards have ~half the transactions, so compare ranks: top-N lists
    top1 = set(s1.nlargest(n_flagged).index)
    top2 = set(s2.nlargest(n_flagged).index)
    flag1, flag2 = s1 >= threshold, s2 >= threshold
    return {
        "cards_in_both_halves": int(len(s1)),
        "spearman_rank_corr": float(spearmanr(s1, s2).statistic),
        "top_n": int(n_flagged),
        "top_n_overlap": len(top1 & top2) / n_flagged,
        "flag_agreement": float((flag1 == flag2).mean()),
        "flagged_h1": int(flag1.sum()), "flagged_h2": int(flag2.sum()),
        "flagged_in_both": int((flag1 & flag2).sum()),
    }
