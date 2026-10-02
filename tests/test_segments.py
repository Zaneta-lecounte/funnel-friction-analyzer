from pathlib import Path

import pandas as pd
import pytest

from funnel import (
    BLENDED,
    blended_funnel,
    segment_comparison,
    segment_divergence,
    segment_insights,
    split_segments,
    validate_segments,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def seg_df():
    return pd.read_csv(ROOT / "sample_segments.csv")


def seg(rows):
    return pd.DataFrame(rows, columns=["segment", "stage", "users"])


def test_sample_segments_valid(seg_df):
    assert validate_segments(seg_df).ok


def test_sample_segments_blend_to_sample_funnel(seg_df, sample_df):
    assert blended_funnel(seg_df)["users"].tolist() == sample_df["users"].tolist()


def test_split_preserves_order(seg_df):
    parts = split_segments(seg_df)
    assert list(parts) == ["Paid", "Organic"]
    assert parts["Paid"]["stage"].iloc[0] == "Landing Page"


def test_missing_segment_column(sample_df):
    result = validate_segments(sample_df)
    assert not result.ok and "segment" in result.errors[0]


def test_single_segment_is_error():
    assert not validate_segments(seg([("A", "x", 10), ("A", "y", 5)])).ok


def test_mismatched_stages_is_error():
    df = seg([("A", "x", 10), ("A", "y", 5), ("B", "y", 10), ("B", "x", 5)])
    result = validate_segments(df)
    assert not result.ok
    assert "same stages in the same order" in result.errors[0]


def test_segment_violation_is_prefixed():
    df = seg([("A", "x", 10), ("A", "y", 5), ("B", "x", 10), ("B", "y", 20)])
    errors = validate_segments(df).errors
    assert len(errors) == 1 and errors[0].startswith("[B] Non-increasing")


def test_segment_comparison_includes_blended(seg_df):
    comp = segment_comparison(seg_df)
    assert set(comp["segment"]) == {"Paid", "Organic", BLENDED}
    paid = comp[comp["segment"] == "Paid"].reset_index(drop=True)
    assert paid.at[0, "rate"] == pytest.approx(0.20)
    assert paid.at[0, "rate_vs_blended_pp"] == pytest.approx(-5.0)
    blended = comp[comp["segment"] == BLENDED]
    assert (blended["rate_vs_blended_pp"].abs() < 1e-9).all()


def test_divergence_sorted_by_spread(seg_df):
    div = segment_divergence(seg_df)
    assert div.at[0, "transition"] == "Form Complete → Qualified Lead"
    assert div.at[0, "best_segment"] == "Organic"
    assert div.at[0, "worst_segment"] == "Paid"
    assert div.at[0, "spread_pp"] == pytest.approx((384 / 780 - 0.30) * 100)
    assert div["spread_pp"].is_monotonic_decreasing
    # Form Start → Form Complete is 60% for both segments.
    same = div[div["transition"] == "Form Start → Form Complete"]
    assert same["spread_pp"].iloc[0] == pytest.approx(0)


def test_segment_insights(seg_df):
    lines = segment_insights(seg_df)
    assert lines[0].startswith("Segments diverge most at Form Complete → Qualified Lead")
    assert "Paid at 30.0%" in lines[0]
    assert lines[1].startswith("Paid leaks most at Opportunity → Customer")
    assert lines[2].startswith("Organic leaks most at")
    assert len(lines) == 3
