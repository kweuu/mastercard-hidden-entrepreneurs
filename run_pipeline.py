"""Hidden entrepreneurs in consumer card data: end-to-end pipeline.

Mastercard Data Quest 2026 (AIESEC Almaty). Run:

    python run_pipeline.py            # uses cached features if present
    python run_pipeline.py --rebuild  # recompute features from raw parquet
    python run_pipeline.py --fast     # smaller search and validation, ~3x quicker

Outputs land in reports/ (metrics, tables, figures, flagged-card list).
"""
import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from hidden_biz import config, data, explain, labeling, modeling, plots, validation
from hidden_biz.features import COMMERCIAL_FEATURES, MCC_BLIND_FEATURES, MODEL_FEATURES, build_card_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pipeline")

# Illustrative business-case assumptions (not from the data; change to the bank's own figures)
CONVERSION_RATE = 0.15          # share of flagged clients who accept a business product
FEE_UPLIFT_ON_COMMERCIAL = 0.01  # extra revenue per KZT of commercial spend once on a business tariff


def mask_card(card: str) -> str:
    return f"{card[:6]}******{card[-4:]}"


def main(rebuild: bool, fast: bool):
    t0 = time.time()
    config.REPORTS.mkdir(exist_ok=True)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    metrics = {}

    # 1. Data ---------------------------------------------------------------
    log.info("Loading and cleaning transactions")
    tx, dq = data.load_transactions()
    metrics["data_quality"] = dq
    log.info("Data quality: %s", {k: v for k, v in dq.items() if k != "mcc_mismatch_examples"})

    # 2. Features -----------------------------------------------------------
    feat_path = config.DATA_PROCESSED / "card_features.parquet"
    if feat_path.exists() and not rebuild:
        cards = pd.read_parquet(feat_path)
    else:
        log.info("Building card-level features")
        t_feat = time.time()
        cards = build_card_features(tx)
        metrics["feature_build_seconds"] = round(time.time() - t_feat, 1)
        cards.to_parquet(feat_path)
    business = cards[cards["is_business"] == 1]
    consumers = cards[cards["is_business"] == 0]
    log.info("Cards: %d business, %d consumer, %d model features", len(business), len(consumers),
             len(MODEL_FEATURES))

    # 3. The naive approach and why it fails --------------------------------
    log.info("Naive benchmark: business vs consumer classifier")
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    naive = cross_val_predict(HistGradientBoostingClassifier(random_state=config.RANDOM_STATE),
                              cards[COMMERCIAL_FEATURES], cards["is_business"], cv=cv,
                              method="predict_proba")[:, 1]
    naive = pd.Series(naive, index=cards.index)
    naive_eval = modeling.evaluate(cards["is_business"], naive, 0.5)
    metrics["naive_business_vs_consumer"] = {
        **naive_eval, "consumers_flagged": int((naive[consumers.index] >= 0.5).sum())}
    plots.naive_trap(naive, cards["is_business"])

    # 4. Personas and silver labels ----------------------------------------
    log.info("Segmenting consumers")
    Z = labeling._scaled_matrix(consumers)
    metrics["kmeans_silhouette"] = labeling.choose_k(Z, range(6, 13, 2) if fast else range(6, 13)).round(4).to_dict()
    personas = labeling.segment_consumers(consumers, k=10)
    silver = labeling.make_silver_labels(consumers, personas)
    persona_table = (consumers.join(silver).groupby("persona")
                     .agg(cards=("n_tx", "size"), is_commercial=("persona_is_commercial", "first"),
                          commercial_spend_share=("commercial_spend_share", "median"),
                          median_ticket_kzt=("median_ticket", "median"),
                          business_hours_share=("business_hours_share", "median"),
                          n_merchants=("n_merchants", "median"))
                     .reset_index().rename(columns={"persona": "label"})
                     .sort_values("cards", ascending=False))
    persona_table.to_csv(config.REPORTS / "personas.csv", index=False)
    plots.personas(persona_table)
    counts = silver["silver_label"].value_counts(dropna=False)
    metrics["silver_labels"] = {"positive": int(counts.get(1.0, 0)), "negative": int(counts.get(0.0, 0)),
                                "ambiguous": int(silver["silver_label"].isna().sum())}
    log.info("Silver labels: %s", metrics["silver_labels"])

    # 5. Train / test split on confidently labelled consumers ----------------
    labelled = consumers.join(silver["silver_label"]).dropna(subset=["silver_label"])
    y = labelled["silver_label"].astype(int)
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        labelled, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)
    X, X_train, X_test = labelled[MODEL_FEATURES], X_train_full[MODEL_FEATURES], X_test_full[MODEL_FEATURES]

    # 6. Model comparison with hyper-parameter search ----------------------
    log.info("Tuning and comparing models (5-fold CV on train)")
    fitted, cv_table = modeling.tune_and_compare(X_train, y_train, n_iter=6 if fast else 12)
    # Business cards are split in two halves: one builds validation hybrids for
    # model selection, the other builds the hybrids reported as the final test.
    rng = np.random.default_rng(config.RANDOM_STATE)
    business_ids = rng.permutation(business.index.to_numpy())
    business_val, business_test = business_ids[: len(business_ids) // 2], business_ids[len(business_ids) // 2:]
    train_negatives = y_train[y_train == 0].index

    rows = []
    for name, model in fitted.items():
        thr = modeling.pick_threshold(model, X_train, y_train)
        m = modeling.evaluate(y_test, model.predict_proba(X_test)[:, 1], thr)
        val_spike = validation.spike_in_test(tx, model, MODEL_FEATURES, thr, train_negatives, business_val,
                                             fractions=(0.0, 0.05, 0.1, 0.2, 0.3), n_pairs=200 if fast else 400)
        rate = val_spike.set_index("fraction")["detection_rate"]
        # mean detection on hybrids, with a heavy penalty for false alarms on clean cards
        sensitivity = rate.drop(0.0).mean() - 10 * rate.loc[0.0]
        rows.append({"model": name, "val_spike_sensitivity": sensitivity,
                     **{f"test_{k}": v for k, v in m.items()}})
    rule = modeling.evaluate(y_test, modeling.rule_baseline_scores(X_test_full), labeling.RULE_POSITIVE_SHARE)
    rows.append({"model": "rule: commercial share >= 25%", **{f"test_{k}": v for k, v in rule.items()}})
    comparison = cv_table.merge(pd.DataFrame(rows), on="model", how="right")
    comparison.to_csv(config.REPORTS / "model_comparison.csv", index=False)

    # Champion: best CV PR-AUC, near-ties broken by sensitivity to partial business activity
    best_name = modeling.choose_champion(comparison.dropna(subset=["cv_pr_auc"]))
    best = fitted[best_name]
    plots.model_comparison(comparison, best_name)
    best_row = comparison.set_index("model").loc[best_name]
    threshold = float(best_row["test_threshold"])
    champion = {k.replace("test_", ""): best_row[k] for k in best_row.index if k.startswith("test_")}
    metrics["champion"] = {"model": best_name, **champion, "best_params": best_row["best_params"]}
    plots.confusion(champion)
    log.info("Champion %s: %s", best_name, {k: round(float(v), 4) for k, v in champion.items()})

    # MCC-blind check: can a model that never sees *what* is bought recover the
    # labels from *how* it is bought (tickets, timing, merchant concentration)?
    blind = HistGradientBoostingClassifier(random_state=config.RANDOM_STATE).fit(X_train_full[MCC_BLIND_FEATURES], y_train)
    metrics["mcc_blind_check"] = modeling.evaluate(y_test, blind.predict_proba(X_test_full[MCC_BLIND_FEATURES])[:, 1], 0.5)

    # 7. Score every consumer card --------------------------------------------
    # Labelled cards get out-of-fold scores (never scored by a model that saw them);
    # ambiguous cards are scored by the champion refit on all labelled cards.
    final = clone(best).fit(X, y)
    t_score = time.time()
    scores = pd.Series(final.predict_proba(consumers[MODEL_FEATURES])[:, 1], index=consumers.index)
    metrics["scoring_seconds_80k_cards"] = round(time.time() - t_score, 3)
    oof = cross_val_predict(clone(best), X, y, cv=cv, method="predict_proba")[:, 1]
    scores[labelled.index] = oof
    plots.score_distribution(scores, threshold)

    # 8. Label-independent validation ---------------------------------------
    log.info("Validation: business cards, spike-in, temporal stability")
    metrics["business_card_recall"] = validation.business_card_recall(final, business, MODEL_FEATURES, threshold)
    negatives_in_test = y_test[y_test == 0].index
    spike = validation.spike_in_test(tx, final, MODEL_FEATURES, threshold,
                                     consumer_ids=negatives_in_test, business_ids=business_test,
                                     n_pairs=300 if fast else 1000)
    spike.to_csv(config.REPORTS / "spike_in.csv", index=False)
    metrics["spike_in"] = spike.round(4).to_dict(orient="records")
    plots.spike_in(spike)
    n_flagged = int((scores >= threshold).sum())
    metrics["temporal_stability"] = validation.temporal_stability(tx, final, MODEL_FEATURES, threshold, n_flagged)

    # 9. Explanations -------------------------------------------------------
    log.info("Explaining the model")
    ambiguous = silver.index[silver["silver_label"].isna()]
    imp = explain.global_importance(final, consumers.loc[ambiguous, MODEL_FEATURES], n_repeats=2 if fast else 3)
    imp.to_csv(config.REPORTS / "feature_importance.csv", header=["mean_abs_score_shift"])
    plots.importance(imp)

    flagged = consumers.loc[scores >= threshold].join(silver[["persona", "persona_is_commercial", "rule_view", "silver_label"]])
    flagged["score"] = scores[flagged.index]
    # Tier = how many independent views agree with the model (scores saturate,
    # so the evidence count ranks leads better than the raw probability)
    evidence = flagged["persona_is_commercial"].astype(int) + (flagged["rule_view"] == "commercial").astype(int)
    flagged["tier"] = evidence.map({2: "1-confirmed", 1: "2-likely", 0: "3-watchlist"})
    flagged["reasons"] = explain.reason_codes(flagged, X[y == 0], X[y == 1], imp)
    out = flagged.reset_index()
    out["card"] = out["card_number"].map(mask_card)
    out[["card", "score", "tier", "persona", "rule_view", "silver_label", "reasons",
         "commercial_spend", "commercial_spend_share", "n_tx", "median_ticket"]] \
        .sort_values("score", ascending=False) \
        .to_csv(config.REPORTS / "flagged_consumers.csv", index=False)
    # small committed sample: 20 highest-scoring cards per tier
    (pd.read_csv(config.REPORTS / "flagged_consumers.csv").groupby("tier").head(20)
     .to_csv(config.REPORTS / "flagged_consumers_sample.csv", index=False))

    commercial_personas = flagged[flagged["persona_is_commercial"]]
    profile = (commercial_personas.groupby("persona")[[f"spendshare_{g}" for g in config.COMMERCIAL_GROUPS]]
               .median())
    plots.persona_profiles(profile)

    # 10. Business sizing -----------------------------------------------------
    # 6 months of data -> year; sizing uses confirmed + likely leads only
    annual_commercial = float(flagged.loc[flagged["tier"] != "3-watchlist", "commercial_spend"].sum() * 2)
    metrics["business_case"] = {
        "consumers_scored": int(len(scores)),
        "flagged": n_flagged,
        "flagged_share": n_flagged / len(scores),
        "flagged_from_ambiguous_zone": int(flagged["silver_label"].isna().sum()),
        "tiers": flagged["tier"].value_counts().sort_index().to_dict(),
        "annual_commercial_spend_by_tier_kzt": (flagged.groupby("tier")["commercial_spend"].sum() * 2).to_dict(),
        "by_persona": flagged["persona"].value_counts().to_dict(),
        "annual_commercial_spend_confirmed_likely_kzt": annual_commercial,
        "assumed_conversion": CONVERSION_RATE,
        "assumed_fee_uplift": FEE_UPLIFT_ON_COMMERCIAL,
        "illustrative_annual_uplift_kzt": annual_commercial * CONVERSION_RATE * FEE_UPLIFT_ON_COMMERCIAL,
    }
    metrics["runtime_seconds"] = round(time.time() - t0, 1)
    with open(config.REPORTS / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log.info("Done in %.0fs. Flagged %d of %d consumer cards.", metrics["runtime_seconds"], n_flagged, len(scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="recompute features from raw data")
    parser.add_argument("--fast", action="store_true", help="smaller search and validation")
    args = parser.parse_args()
    main(args.rebuild, args.fast)
