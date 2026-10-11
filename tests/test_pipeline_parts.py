"""Unit tests on a tiny hand-made dataset (no raw data needed)."""
import pandas as pd
import pytest

from hidden_biz import data, features


@pytest.fixture
def merchants():
    return pd.DataFrame({
        "merchant_id": ["MER_1", "MER_2", "MER_3"],
        "merchant_name": ["Meta Ads", "Magnum", "Wholesale LLP"],
        "mcc": ["7311", "5411", "5045"],
        "merchant_country": ["Ireland", "Kazakhstan", "Kazakhstan"],
        "recurring_capable": [True, False, False],
    })


@pytest.fixture
def tx():
    rows = [
        # card A: ordinary consumer, groceries in the evening
        ("A", "2025-10-06 19:00", 5_000, "5411", "MER_2", "POS", "Kazakhstan", 0),
        ("A", "2025-10-07 20:00", 7_000, "5411", "MER_2", "POS", "Kazakhstan", 0),
        # card B: ads (miscoded as 7012) and wholesale in office hours
        ("B", "2025-10-06 10:00", 150_000, "7012", "MER_1", "online", "Ireland", 0),
        ("B", "2025-10-07 11:00", 400_000, "5045", "MER_3", "POS", "Kazakhstan", 0),
        ("B", "2025-10-08 12:00", 420_000, "5045", "MER_3", "POS", "Kazakhstan", 0),
        ("B", "2025-11-08 12:00", 380_000, "5045", "MER_3", "POS", "Kazakhstan", 0),
        ("B", "2025-11-09 20:00", 6_000, "5411", "MER_2", "POS", "Kazakhstan", 0),
    ]
    df = pd.DataFrame(rows, columns=["card_number", "transaction_timestamp", "transaction_amount_kzt",
                                     "mcc", "merchant_id", "channel", "country", "is_business"])
    df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"])
    df["tokenized"] = False
    df["is_recurring"] = False
    return df


def test_cleaning_fixes_mcc_from_merchant_master(tx, merchants):
    clean, report = data.clean_transactions(tx, merchants)
    assert report["mcc_mismatches_fixed"] == 1
    assert clean.loc[clean["merchant_id"] == "MER_1", "mcc"].eq("7311").all()


def test_card_features_capture_commercial_footprint(tx, merchants):
    clean, _ = data.clean_transactions(tx, merchants)
    f = features.build_card_features(clean)
    assert f.loc["A", "commercial_spend_share"] == 0
    assert f.loc["B", "commercial_spend_share"] > 0.9
    assert f.loc["B", "n_repeat_commercial_merchants"] == 1  # MER_3 visited 3 times
    # 8 Nov 2025 is a Saturday, so only the three October purchases count
    assert f.loc["B", "business_hours_share"] == pytest.approx(3 / 5)
    assert f.loc["B", "spendshare_ads"] > 0


def test_model_features_exclude_product_identifiers():
    banned = {"card_number", "card_tier", "bank_name", "is_business"}
    assert banned.isdisjoint(features.MODEL_FEATURES)
    assert banned.isdisjoint(features.MCC_BLIND_FEATURES)
    assert not any(f.startswith(("spendshare_", "txshare_")) for f in features.MCC_BLIND_FEATURES)
