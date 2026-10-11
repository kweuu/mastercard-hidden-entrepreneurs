"""Global and per-card explanations."""
import numpy as np
import pandas as pd

from . import config

READABLE = {
    "commercial_spend_share": "{:.0%} of spend goes to business inputs",
    "log_commercial_spend": "commercial spend of {:,.0f} KZT",
    "commercial_tx_count": "{:.0f} purchases in business categories",
    "n_repeat_commercial_merchants": "{:.0f} suppliers/platforms used repeatedly",
    "n_commercial_merchants": "{:.0f} different business merchants",
    "spendshare_ads": "{:.0%} of spend on advertising",
    "txshare_ads": "{:.0%} of purchases are ad payments",
    "spendshare_saas_cloud": "{:.0%} of spend on software/cloud",
    "spendshare_wholesale": "{:.0%} of spend at wholesalers",
    "txshare_wholesale": "{:.0%} of purchases at wholesalers",
    "spendshare_stock_retail": "{:.0%} of spend on stock (building materials, food)",
    "txshare_stock_retail": "{:.0%} of purchases are stock buys",
    "spendshare_logistics": "{:.0%} of spend on delivery/freight",
    "business_hours_share": "{:.0%} of purchases in weekday office hours",
    "commercial_business_hours_share": "{:.0%} of business purchases in office hours",
    "log_median_ticket": "median ticket {:,.0f} KZT",
    "large_tx_share": "{:.0%} of purchases above 100k KZT",
    "top_merchant_share": "{:.0%} of purchases at one merchant",
    "n_merchants": "only {:.0f} merchants in 6 months",
    "online_share": "{:.0%} of purchases online",
    "commercial_tx_share": "{:.0%} of purchases in business categories",
    "n_commercial_groups": "{:.0f} types of business inputs",
    "n_recurring_commercial_merchants": "{:.0f} business subscriptions",
}
LOG_FEATURES = {"log_commercial_spend", "log_median_ticket"}


def global_importance(model, X_eval: pd.DataFrame, n_repeats=3) -> pd.Series:
    """Mean absolute change in P(entrepreneur) when one feature is shuffled.

    Measured on the ambiguous cards, where the model actually makes decisions.
    On the clean silver test set one feature already separates the classes,
    so classic permutation importance (drop in PR-AUC) is zero for the rest.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    base = model.predict_proba(X_eval)[:, 1]
    result = {}
    for col in X_eval.columns:
        shifts = []
        for _ in range(n_repeats):
            shuffled = X_eval.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            shifts.append(np.abs(model.predict_proba(shuffled)[:, 1] - base).mean())
        result[col] = float(np.mean(shifts))
    return pd.Series(result).sort_values(ascending=False)


def reason_codes(cards: pd.DataFrame, reference_neg: pd.DataFrame, reference_pos: pd.DataFrame,
                 importance: pd.Series, top_k_features=15, n_reasons=3) -> pd.Series:
    """Three plain-language reasons per flagged card.

    For the most important features, measure how far the card sits from a
    typical ordinary consumer (robust z-score), in the direction that points
    toward commercial behaviour, and print the strongest deviations.
    """
    feats = [f for f in importance.index[:top_k_features] if f in READABLE]
    med_neg = reference_neg[feats].median()
    iqr = (reference_neg[feats].quantile(0.75) - reference_neg[feats].quantile(0.25)).replace(0, np.nan)
    iqr = iqr.fillna(reference_neg[feats].std()).replace(0, 1)
    direction = np.sign(reference_pos[feats].median() - med_neg).replace(0, 1)
    z = (cards[feats] - med_neg) / iqr * direction

    def describe(row):
        top = z.loc[row.name].nlargest(n_reasons)
        texts = []
        for feat in top.index:
            value = np.expm1(row[feat]) if feat in LOG_FEATURES else row[feat]
            texts.append(READABLE[feat].format(value))
        return "; ".join(texts)

    return cards.apply(describe, axis=1)
