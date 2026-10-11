"""Static figures for the README and the deck (one consistent style)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b4b2a9"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e4dd", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def _title(fig, text):
    """Figure-level title aligned to the left edge, so long tick labels never clip it."""
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)


def _save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES / name, dpi=180)
    plt.close(fig)


def naive_trap(naive_scores: pd.Series, is_business: pd.Series):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0, 1, 41)
    ax.hist(naive_scores[is_business == 0], bins=bins, color=BLUE, label="Consumer cards (80,000)",
            edgecolor=SURFACE, linewidth=1)
    ax.hist(naive_scores[is_business == 1], bins=bins, color=ORANGE, label="Business cards (25,000)",
            edgecolor=SURFACE, linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("P(business card)")
    ax.set_ylabel("Cards (log scale)")
    _title(fig, "Naive model: business vs consumer separates perfectly")
    n_flag = int((naive_scores[is_business == 0] >= 0.5).sum())
    ax.annotate(f"only {n_flag} consumers score above 0.5", xy=(0.5, 3), xytext=(0.3, 300),
                color=INK_2, arrowprops={"arrowstyle": "->", "color": INK_2})
    ax.legend(frameon=False)
    _save(fig, "01_naive_trap.png")


def personas(persona_table: pd.DataFrame):
    t = persona_table.sort_values("cards")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    colors = [ORANGE if c else GRAY for c in t["is_commercial"]]
    ax.barh(t["label"], t["cards"], color=colors, height=0.7, edgecolor=SURFACE, linewidth=2)
    for y, (n, share) in enumerate(zip(t["cards"], t["commercial_spend_share"])):
        ax.text(n + 300, y, f"{n:,} cards · {share:.0%}", va="center", color=INK_2, fontsize=8.5)
    ax.set_xlabel("cards · median share of spend on business inputs")
    ax.set_xticks([])
    _title(fig, "Unsupervised personas: 6 of 10 clusters run a business")
    ax.set_xlim(0, t["cards"].max() * 1.45)
    ax.grid(axis="y", visible=False)
    _save(fig, "02_personas.png")


def model_comparison(comparison: pd.DataFrame, champion: str):
    """Two small multiples: test PR-AUC on silver labels, and spike-in sensitivity."""
    t = comparison.copy()
    t["val_spike_sensitivity"] = t["val_spike_sensitivity"].fillna(np.nan)
    t = t.sort_values("test_pr_auc")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    colors = [ORANGE if m == champion else BLUE for m in t["model"]]
    axes[0].barh(t["model"], t["test_pr_auc"], color=colors, height=0.6)
    for y, v in enumerate(t["test_pr_auc"]):
        axes[0].text(v - 0.002, y, f"{v:.4f}", va="center", ha="right", fontsize=8.5, color="white")
    axes[0].set_xlim(0.95, 1.0)
    axes[0].set_title("Test PR-AUC (silver labels)")
    sens = t["val_spike_sensitivity"].fillna(0)
    axes[1].barh(t["model"], sens, color=colors, height=0.6)
    for y, (v, m) in enumerate(zip(sens, t["model"])):
        axes[1].text(v + 0.01, y, "n/a" if m.startswith("rule") else f"{v:.2f}", va="center",
                     fontsize=8.5, color=INK_2)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Sensitivity to partial business activity")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    _save(fig, "03_model_comparison.png")


def confusion(m: dict):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#eef4fc", BLUE]))
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13,
                color="white" if v > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["Predicted\npersonal", "Predicted\nentrepreneur"])
    ax.set_yticks([0, 1], ["Silver: personal", "Silver: entrepreneur"])
    ax.grid(False)
    _title(fig, f"Confusion matrix (threshold {m['threshold']:.2f})")
    _save(fig, "04_confusion_matrix.png")


def importance(imp: pd.Series, top=12):
    t = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(t.index, t.values, color=BLUE, height=0.65)
    ax.set_xlabel("Mean shift in P(entrepreneur) when the feature is shuffled")
    _title(fig, "What drives decisions on the 19k ambiguous cards")
    ax.grid(axis="y", visible=False)
    _save(fig, "05_feature_importance.png")


def spike_in(curve: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(curve["fraction"] * 100, curve["detection_rate"] * 100, color=BLUE, linewidth=2,
            marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    for x, y in zip(curve["fraction"], curve["detection_rate"]):
        ax.annotate(f"{y:.0%}", (x * 100, y * 100), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color=INK_2)
    ax.set_xlabel("Share of the card's transactions taken from a real business card, %")
    ax.set_ylabel("Flagged, %")
    ax.set_ylim(-5, 110)
    _title(fig, "Spike-in test: how much business the model needs to notice")
    _save(fig, "06_spike_in.png")


def score_distribution(scores: pd.Series, threshold: float):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.linspace(0, 1, 51)
    ax.hist(scores, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.axvline(threshold, color=ORANGE, linewidth=2)
    ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.6, f"threshold {threshold:.2f}\n"
            f"{(scores >= threshold).sum():,} flagged", color=INK_2, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("P(hidden entrepreneur)")
    ax.set_ylabel("Consumer cards (log)")
    _title(fig, "Final score across all 80,000 consumer cards")
    _save(fig, "07_score_distribution.png")


def persona_profiles(profile: pd.DataFrame):
    """Heatmap: commercial-group spend shares per flagged persona."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#104281"])
    im = ax.imshow(profile.values, cmap=cmap, aspect="auto", vmin=0, vmax=max(0.6, profile.values.max()))
    ax.set_xticks(range(profile.shape[1]), [c.replace("spendshare_", "") for c in profile.columns],
                  rotation=30, ha="right")
    ax.set_yticks(range(profile.shape[0]), profile.index)
    for (i, j), v in np.ndenumerate(profile.values):
        if v >= 0.05:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.35 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, label="Median share of spend")
    _title(fig, "Where each entrepreneur persona spends")
    _save(fig, "08_persona_profiles.png")
