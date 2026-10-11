"""Model zoo, hyper-parameter search, threshold choice and evaluation."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from . import config


def _signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


def candidate_models() -> dict:
    """Three model families of rising complexity, each with a search space.

    - Logistic regression: transparent coefficients, the baseline a risk team trusts.
    - Random forest: non-linear, robust to outliers, little tuning.
    - Histogram gradient boosting: strongest on tabular data, handles skew,
      fast enough to score millions of cards (the case asks for "fast").
    """
    return {
        "logistic_regression": (
            make_pipeline(FunctionTransformer(_signed_log), StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced")),
            {"logisticregression__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
        ),
        "random_forest": (
            RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample",
                                   random_state=config.RANDOM_STATE),
            {"n_estimators": [200, 300], "max_depth": [8, 12, 16, None],
             "min_samples_leaf": [1, 5, 20], "max_features": ["sqrt", 0.3]},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=config.RANDOM_STATE, class_weight="balanced"),
            {"learning_rate": [0.03, 0.05, 0.1], "max_leaf_nodes": [15, 31, 63],
             "min_samples_leaf": [20, 50, 100], "l2_regularization": [0.0, 0.1, 1.0],
             "max_iter": [200, 400]},
        ),
    }


def tune_and_compare(X: pd.DataFrame, y: pd.Series, n_iter: int = 12) -> tuple[dict, pd.DataFrame]:
    """5-fold randomized search per family, scored by PR-AUC (average precision).

    PR-AUC is used instead of accuracy because positives are a minority (~16%)
    and the business cares about the quality of the flagged list.
    """
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    fitted, rows = {}, []
    for name, (model, grid) in candidate_models().items():
        n_combos = int(np.prod([len(v) for v in grid.values()]))
        search = RandomizedSearchCV(model, grid, n_iter=min(n_iter, n_combos), scoring="average_precision",
                                    cv=cv, n_jobs=-1, random_state=config.RANDOM_STATE, refit=True)
        search.fit(X, y)
        fitted[name] = search.best_estimator_
        rows.append({"model": name, "cv_pr_auc": search.best_score_,
                     "cv_pr_auc_std": search.cv_results_["std_test_score"][search.best_index_],
                     "best_params": search.best_params_})
    return fitted, pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False)


def pick_threshold(model, X: pd.DataFrame, y: pd.Series, beta: float = 0.5) -> float:
    """Threshold that maximises out-of-fold F-beta on the training folds.

    beta = 0.5 weighs precision twice as much as recall: a relationship
    manager pays for every false lead, while a missed entrepreneur can still
    be caught next month. Never chosen on the test set.
    """
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, oof)
    p, r = precision[:-1], recall[:-1]
    fbeta = (1 + beta**2) * p * r / np.clip(beta**2 * p + r, 1e-12, None)
    # On well-separated data many thresholds tie for the best score. Taking the
    # edge of that plateau makes the cut-off brittle; the median is not.
    plateau = thresholds[fbeta >= fbeta.max() - 1e-4]
    return float(np.median(plateau))


def choose_champion(table: pd.DataFrame, tolerance: float = 0.001) -> str:
    """Best CV PR-AUC; near-ties are broken by sensitivity on validation hybrids.

    Silver labels are clean, so CV PR-AUC saturates near 1.0 for every model.
    What separates the models is how they treat partial business activity,
    measured by `val_spike_sensitivity` (built from training cards only).
    """
    best = table["cv_pr_auc"].max()
    near = table[table["cv_pr_auc"] >= best - tolerance]
    return near.sort_values(["val_spike_sensitivity", "cv_pr_auc"], ascending=False)["model"].iloc[0]


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def rule_baseline_scores(X: pd.DataFrame) -> np.ndarray:
    """Single-feature baseline: share of spend in commercial MCC groups."""
    return X["commercial_spend_share"].to_numpy()
