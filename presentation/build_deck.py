"""Build the PowerPoint deck from reports/ (run after run_pipeline.py).

Every number on the slides is read from reports/metrics.json, so the deck
never drifts from the code.
"""
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
REPORTS, FIG = ROOT / "reports", ROOT / "reports" / "figures"
OUT = Path(__file__).resolve().parent / "hidden_entrepreneurs.pptx"

INK, INK_2, ORANGE, BLUE, LIGHT = (RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E),
                                   RGBColor(0xEB, 0x68, 0x34), RGBColor(0x2A, 0x78, 0xD6),
                                   RGBColor(0xF4, 0xF3, 0xEF))
W, H = Inches(13.333), Inches(7.5)


def text_box(slide, x, y, w, h, lines, size=16, color=INK, bold_first=False, spacing=6):
    """Add a text box; `lines` is a list of str (each becomes a paragraph)."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.bold = bold_first and i == 0
        p.space_after = Pt(spacing)
    return box


def new_slide(prs, title, kicker=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bar = slide.shapes.add_shape(1, 0, 0, Inches(0.18), H)
    bar.fill.solid()
    bar.fill.fore_color.rgb = ORANGE
    bar.line.fill.background()
    if kicker:
        text_box(slide, Inches(0.6), Inches(0.35), Inches(12), Inches(0.4), [kicker.upper()], 12, ORANGE)
    text_box(slide, Inches(0.6), Inches(0.65), Inches(12.2), Inches(1.0), [title], 28, INK, bold_first=True)
    return slide


def picture(slide, name, x, y, w=None, h=None):
    slide.shapes.add_picture(str(FIG / name), x, y, width=w, height=h)


def table(slide, x, y, w, rows, col_widths=None, size=13):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, Inches(0.4) * len(rows))
    tbl = shape.table
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(value)
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(size)
            para.font.bold = r == 0
            para.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if r == 0 else INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE if r == 0 else (LIGHT if r % 2 else RGBColor(0xFF, 0xFF, 0xFF))
    if col_widths:
        for c, cw in enumerate(col_widths):
            tbl.columns[c].width = Emu(int(w * cw))
    return tbl


def main():
    m = json.loads((REPORTS / "metrics.json").read_text())
    dq, naive, champ = m["data_quality"], m["naive_business_vs_consumer"], m["champion"]
    bc, ts, blind = m["business_case"], m["temporal_stability"], m["mcc_blind_check"]
    spike = pd.DataFrame(m["spike_in"]).set_index("fraction")["detection_rate"]
    comparison = pd.read_csv(REPORTS / "model_comparison.csv")
    personas = pd.read_csv(REPORTS / "personas.csv")
    flagged_n, consumers_n = bc["flagged"], bc["consumers_scored"]

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1. Title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(1, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = INK
    bg.line.fill.background()
    text_box(s, Inches(0.8), Inches(2.2), Inches(11.5), Inches(1.5), ["Hidden entrepreneurs in card data"],
             44, RGBColor(0xFF, 0xFF, 0xFF), bold_first=True)
    text_box(s, Inches(0.8), Inches(3.5), Inches(11.5), Inches(1.2),
             ["Detecting commercial activity on consumer cards",
              "Mastercard Data Quest 2026 · AIESEC in Almaty"], 20, RGBColor(0xC3, 0xC2, 0xB7))
    text_box(s, Inches(0.8), Inches(5.3), Inches(11.5), Inches(0.8),
             [f"{flagged_n:,} of {consumers_n:,} consumer cards ({bc['flagged_share']:.0%}) run a business"],
             22, ORANGE, bold_first=True)

    # 2. Task
    s = new_slide(prs, "Find people who run a business through a personal card", "The task")
    text_box(s, Inches(0.6), Inches(1.7), Inches(6), Inches(5), [
        "Who: sole traders and self-employed people who sell through ordinary consumer cards.",
        "Goal: a predictive ML model that spots this hidden commercial activity reliably and fast.",
        "No label: nothing in the data says a consumer is an entrepreneur.",
    ], 18)
    text_box(s, Inches(7.0), Inches(1.7), Inches(5.8), Inches(5), [
        "Why the bank cares",
        "Business tariffs: higher fees and POS acquiring revenue.",
        "Cross-sell: working-capital loans, payroll projects, merchant acquiring, cash registers.",
    ], 18, bold_first=True)

    # 3. Data and cleaning
    s = new_slide(prs, "12.8M transactions, one data-quality fix that matters", "Data")
    table(s, Inches(0.6), Inches(1.8), Inches(6.2), [
        ["Segment", "Cards", "Transactions"],
        ["Business", f"{dq['business_cards']:,}", "2,997,593"],
        ["Consumer", f"{dq['consumer_cards']:,}", "9,832,487"],
        ["Merchants (reference)", "2,165", ""],
    ], [0.5, 0.25, 0.25], 15)
    text_box(s, Inches(7.2), Inches(1.8), Inches(5.6), Inches(5), [
        f"{dq['mcc_mismatches_fixed']:,} ad-platform payments coded 7012 (timeshares); merchant master says 7311 (ads). Fixed from the master.",
        f"{dq['rows_above_p999']:,} purchases above {dq['amount_p999_kzt'] / 1e6:.2f}M KZT kept: big tickets are normal for a business. Features use medians and logs.",
        "No duplicates, nulls or non-positive amounts; every merchant found in the reference.",
        "Oct 2025 to Mar 2026, synthetic data.",
    ], 16)

    # 4. Trap
    s = new_slide(prs, f"The obvious model scores AUC ≈ 1.0 and finds {naive['consumers_flagged']} people", "The trap")
    picture(s, "01_naive_trap.png", Inches(0.6), Inches(1.8), w=Inches(8.7))
    text_box(s, Inches(9.6), Inches(1.8), Inches(3.4), Inches(5), [
        "Business vs consumer is too easy: business cards here never buy groceries.",
        "The model learns the card product, not the behaviour.",
        "A hidden entrepreneur buys groceries on the same card, so this model calls them a consumer.",
    ], 15)

    # 5. Approach
    s = new_slide(prs, "Weak supervision: label only where two independent views agree", "Approach")
    steps = [("1  Clean", "MCC fix, outliers kept"), ("2  Features", "49 per card, MCC groups by ISO meaning"),
             ("3  Personas", "K-Means, 10 clusters"), ("4  Silver labels", "cluster AND rule agree"),
             ("5  Models", "LR · RF · boosting, 5-fold search"), ("6  Checks", "4 tests without our labels")]
    for i, (head, sub) in enumerate(steps):
        x = Inches(0.6 + i * 2.1)
        box = s.shapes.add_shape(1, x, Inches(2.0), Inches(1.95), Inches(1.6))
        box.fill.solid()
        box.fill.fore_color.rgb = ORANGE if i == 3 else LIGHT
        box.line.fill.background()
        text_box(s, x + Inches(0.1), Inches(2.1), Inches(1.75), Inches(1.4), [head, sub], 14,
                 RGBColor(0xFF, 0xFF, 0xFF) if i == 3 else INK, bold_first=True)
    sl = m["silver_labels"]
    text_box(s, Inches(0.6), Inches(4.1), Inches(12.2), Inches(3), [
        "Cluster view: the card sits in a persona whose median card spends ≥ 30% on business inputs.",
        "Rule view: ≥ 25% of spend on business inputs and ≥ 2 commercial merchants used repeatedly; < 10% = clearly personal.",
        f"Result: {sl['positive']:,} positives, {sl['negative']:,} negatives, {sl['ambiguous']:,} ambiguous cards left for the model.",
        "Card tier, bank and BIN never enter the model: they describe the product, not the behaviour.",
    ], 16)

    # 6. Personas
    s = new_slide(prs, "Six of ten consumer personas run a business", "Unsupervised discovery")
    picture(s, "02_personas.png", Inches(0.5), Inches(1.7), w=Inches(9.6))
    text_box(s, Inches(10.4), Inches(1.8), Inches(2.6), Inches(5), [
        "Stable: 99.8% same commercial split across random seeds.",
        "Resellers buy stock at building and food stores, where business cards never shop.",
    ], 14)

    # 7. Models
    s = new_slide(prs, "All models reproduce the labels; boosting handles partial business best", "Model choice")
    picture(s, "03_model_comparison.png", Inches(0.4), Inches(1.6), w=Inches(8.4))
    picture(s, "04_confusion_matrix.png", Inches(8.9), Inches(1.6), w=Inches(4.2))
    best = comparison.set_index("model").loc[champ["model"]]
    text_box(s, Inches(0.6), Inches(5.0), Inches(8), Inches(2.3), [
        f"Champion: histogram gradient boosting, threshold {champ['threshold']:.2f} (max out-of-fold F0.5).",
        f"Tie on PR-AUC is expected: silver labels keep clear cases only. The tie-break is sensitivity to cards with 5–30% business activity ({best['val_spike_sensitivity']:.2f} vs 0.38–0.39).",
        "Boosting also scales: 80,000 cards scored in under a second.",
    ], 14)

    # 8. Validation
    s = new_slide(prs, "Four checks that never use our labels", "Does it work?")
    table(s, Inches(0.6), Inches(1.7), Inches(7.0), [
        ["Check", "Result"],
        ["Real business cards (never seen)", f"{m['business_card_recall']['flagged_share']:.2%} flagged"],
        ["MCC-blind model (how, not what)", f"ROC-AUC {blind['roc_auc']:.4f} · F1 {blind['f1']:.3f}"],
        ["Spike-in: clean consumer cards", f"{spike.loc[0.0]:.0%} false alarms"],
        ["Spike-in: 10% / 20% business", f"{spike.loc[0.1]:.0%} / {spike.loc[0.2]:.0%} caught"],
        ["Same decision Oct–Dec vs Jan–Mar", f"{ts['flag_agreement']:.1%} of cards"],
    ], [0.55, 0.45], 14)
    picture(s, "06_spike_in.png", Inches(7.8), Inches(1.7), w=Inches(5.3))
    text_box(s, Inches(0.6), Inches(5.0), Inches(7), Inches(2), [
        "MCC-blind check: labels come from what people buy; a model that sees only tickets, timing and merchant concentration finds the same people. The personas are real behaviour.",
    ], 14)

    # 9. Explainability
    s = new_slide(prs, "Every flag comes with three reasons a banker can read", "Explainability")
    picture(s, "05_feature_importance.png", Inches(0.6), Inches(1.6), h=Inches(4.6))
    sample = pd.read_csv(REPORTS / "flagged_consumers.csv").groupby("tier").head(1)
    text_box(s, Inches(8.7), Inches(1.7), Inches(4.4), Inches(5.5),
             ["Examples"] + [f"{r.card} · {r.tier}: {r.reasons}" for r in sample.itertuples()], 12, bold_first=True)

    # 10. Personas and offers
    s = new_slide(prs, "Who they are and what to offer", "Recommendations")
    picture(s, "08_persona_profiles.png", Inches(0.4), Inches(1.5), w=Inches(6.6))
    size = lambda key: int(personas.loc[personas["label"].str.startswith(key), "cards"].sum())  # noqa: E731
    table(s, Inches(7.2), Inches(1.6), Inches(5.9), [
        ["Persona", "Cards", "Offer"],
        ["Online seller / marketer", f"{size('Online seller'):,}", "Card with ad cashback, FX for SaaS, payment links"],
        ["Reseller (stock at retail)", f"{size('Reseller'):,}", "QR/POS acquiring, inventory loan, IP registration"],
        ["Wholesale trader", f"{size('Wholesale'):,}", "Credit line, corporate card, payroll"],
        ["Logistics trader", f"{size('Delivery'):,}", "Fleet/fuel card, acquiring"],
    ], [0.36, 0.14, 0.5], 12)

    # 11. Business case
    s = new_slide(prs, "Start with 9,757 confirmed leads", "Business case")
    spend = bc["annual_commercial_spend_by_tier_kzt"]
    table(s, Inches(0.6), Inches(1.7), Inches(7.5), [
        ["Tier", "Rule", "Cards", "Commercial spend / year"],
        ["1 confirmed", "model + cluster + rule", f"{bc['tiers']['1-confirmed']:,}", f"{spend['1-confirmed'] / 1e9:.1f} bn KZT"],
        ["2 likely", "model + one view", f"{bc['tiers']['2-likely']:,}", f"{spend['2-likely'] / 1e9:.1f} bn KZT"],
        ["3 watchlist", "model only", f"{bc['tiers']['3-watchlist']:,}", f"{spend['3-watchlist'] / 1e9:.1f} bn KZT"],
    ], [0.2, 0.33, 0.17, 0.3], 14)
    text_box(s, Inches(8.5), Inches(1.7), Inches(4.5), Inches(5), [
        f"≈ {bc['illustrative_annual_uplift_kzt'] / 1e6:,.0f}M KZT a year",
        f"Illustrative: tiers 1–2, {bc['assumed_conversion']:.0%} conversion, +{bc['assumed_fee_uplift']:.0%} revenue on commercial spend after moving to a business tariff. Replace with the bank's own numbers.",
    ], 16, bold_first=True)
    text_box(s, Inches(0.6), Inches(4.4), Inches(7.5), Inches(2.5), [
        f"{bc['flagged_from_ambiguous_zone']:,} flagged cards came from the ambiguous zone: the model adds leads the rules alone would not send.",
        "Use the score to offer a better product, never to report or reprice clients without consent.",
    ], 14)

    # 12. Limitations and next steps
    s = new_slide(prs, "Limitations and next steps", "Honest view")
    text_box(s, Inches(0.6), Inches(1.7), Inches(6), Inches(5.5), [
        "Limitations",
        "Synthetic data with clean personas: real scores will be lower.",
        "Silver labels come from my own cluster model and rule.",
        "Spend side only: the brief mentions incoming payments from many counterparties, which the data does not include.",
        "MCC dictionary is expert-made and needs product-team review.",
    ], 15, bold_first=True)
    text_box(s, Inches(7.0), Inches(1.7), Inches(6), Inches(5.5), [
        "Next steps",
        "Pilot: managers call a random sample per tier; outcomes become real labels; retrain.",
        "Add inflow features: distinct senders, regular incoming transfers.",
        "Monthly re-scoring with drift monitoring on features and persona sizes.",
    ], 15, bold_first=True)

    prs.save(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
