import math

import pandas as pd
import pytest

from conftest import make_funnel
from funnel import (
    clean_funnel,
    compute_metrics,
    largest_leakage,
    step_rates,
    transition_table,
    validate_funnel,
)


# ---------- validation ----------

def test_sample_is_valid(sample_df):
    result = validate_funnel(sample_df)
    assert result.ok
    assert result.errors == []
    assert result.warnings == []


def test_increasing_stage_is_flagged_and_explained():
    df = make_funnel([("A", 100), ("B", 50), ("C", 80)])
    result = validate_funnel(df)
    assert not result.ok
    assert len(result.errors) == 1
    msg = result.errors[0]
    assert "'C' has 80 users" in msg and "'B', has only 50" in msg
    assert "160% step conversion" in msg
    assert "only stay flat or fall" in msg


def test_every_increase_is_reported():
    df = make_funnel([("A", 10), ("B", 20), ("C", 5), ("D", 9)])
    errors = validate_funnel(df).errors
    assert sum("Non-increasing" in e for e in errors) == 2


def test_flat_stages_are_allowed():
    assert validate_funnel(make_funnel([("A", 100), ("B", 100), ("C", 0)])).ok


def test_missing_columns():
    result = validate_funnel(pd.DataFrame({"step": ["A"], "count": [1]}))
    assert not result.ok
    assert "stage" in result.errors[0] and "users" in result.errors[0]


def test_column_names_are_case_and_space_insensitive():
    df = pd.DataFrame({" Stage ": ["A", "B"], "USERS": [10, 5]})
    assert validate_funnel(df).ok


def test_too_few_stages():
    assert not validate_funnel(make_funnel([("A", 10)])).ok


def test_non_numeric_negative_blank_and_duplicate():
    df = make_funnel([("A", 100), ("B", "lots"), ("", 10), ("A", -1)])
    errors = " | ".join(validate_funnel(df).errors)
    assert "must be a number" in errors
    assert "blank" in errors
    assert "more than once" in errors
    assert "cannot be negative" in errors


def test_zero_top_stage_is_error():
    assert not validate_funnel(make_funnel([("A", 0), ("B", 0)])).ok


def test_zero_mid_funnel_is_warning():
    result = validate_funnel(make_funnel([("A", 100), ("B", 0), ("C", 0)]))
    assert result.ok
    assert any("0 users" in w for w in result.warnings)


def test_blank_editor_rows_are_ignored():
    df = pd.DataFrame({"stage": ["A", "B", None], "users": [10, 5, None]})
    assert validate_funnel(df).ok
    assert len(clean_funnel(df)) == 2


# ---------- metrics ----------

def test_step_rates_handles_zero_denominator():
    assert step_rates([100, 0, 0]) == [0.0, 0.0]


def test_compute_metrics_on_sample(sample_df):
    m = compute_metrics(sample_df)
    assert list(m.columns) == [
        "stage", "users", "step_conversion", "drop_off", "users_lost", "cumulative_conversion"
    ]
    assert math.isnan(m.at[0, "step_conversion"])
    assert m.at[0, "cumulative_conversion"] == 1.0
    assert m.at[0, "users_lost"] == 0

    assert m.at[1, "step_conversion"] == pytest.approx(0.25)
    assert m.at[1, "drop_off"] == pytest.approx(0.75)
    assert m.at[1, "users_lost"] == 7500
    assert m.at[2, "step_conversion"] == pytest.approx(0.6)
    assert m.at[3, "step_conversion"] == pytest.approx(0.4)
    assert m.at[4, "step_conversion"] == pytest.approx(0.3)
    assert m.at[5, "step_conversion"] == pytest.approx(40 / 180)
    assert m.at[5, "cumulative_conversion"] == pytest.approx(0.004)


def test_cumulative_equals_product_of_step_rates(sample_df):
    m = compute_metrics(sample_df)
    assert m["step_conversion"].iloc[1:].prod() == pytest.approx(m["cumulative_conversion"].iloc[-1])


def test_users_lost_sum_to_top_minus_bottom(sample_df):
    m = compute_metrics(sample_df)
    assert m["users_lost"].sum() == 10000 - 40


def test_transition_table(sample_df):
    t = transition_table(sample_df)
    assert len(t) == 5
    assert t.at[0, "transition"] == "Landing Page → Form Start"
    assert t.at[4, "from_stage"] == "Opportunity"
    assert t.at[4, "to_stage"] == "Customer"
    assert t.at[2, "rate"] == pytest.approx(0.4)


# ---------- leakage ----------

def test_largest_leakage_by_drop_off(sample_df):
    # Opportunity → Customer loses 77.8%, more than Landing Page → Form Start (75%).
    t = transition_table(sample_df)
    assert t.at[largest_leakage(sample_df, "drop_off"), "transition"] == "Opportunity → Customer"


def test_largest_leakage_by_users_lost(sample_df):
    t = transition_table(sample_df)
    assert t.at[largest_leakage(sample_df, "users_lost"), "transition"] == "Landing Page → Form Start"


def test_largest_leakage_tie_takes_topmost():
    df = make_funnel([("A", 100), ("B", 50), ("C", 25)])
    assert largest_leakage(df, "drop_off") == 0


def test_largest_leakage_rejects_unknown_metric(sample_df):
    with pytest.raises(ValueError):
        largest_leakage(sample_df, "vibes")
