import math

import pytest

from funnel import benchmark_gap
from opportunity_model import benchmark_opportunity, benchmark_total_upside

TARGETS = [0.30, 0.60, 0.35, None, 0.25]


def test_benchmark_gap(sample_df):
    gap = benchmark_gap(sample_df, TARGETS)
    assert gap["status"].tolist() == ["Below", "At/above", "At/above", "No target", "Below"]
    assert gap.at[0, "gap_pp"] == pytest.approx(-5.0)
    assert gap.at[0, "users_gap"] == pytest.approx(500)       # 10,000*30% - 2,500
    assert gap.at[2, "gap_pp"] == pytest.approx(5.0)
    assert gap.at[2, "users_gap"] == pytest.approx(-75)       # 1,500*35% - 600 (ahead)
    assert math.isnan(gap.at[3, "gap_pp"])
    assert gap.at[4, "users_gap"] == pytest.approx(5)         # 180*25% - 40


def test_benchmark_gap_accepts_nan_as_no_target(sample_df):
    gap = benchmark_gap(sample_df, [float("nan")] * 5)
    assert (gap["status"] == "No target").all()


@pytest.mark.parametrize("targets", [[0.3] * 4, [1.5, None, None, None, None]])
def test_benchmark_gap_rejects_bad_targets(sample_df, targets):
    with pytest.raises(ValueError):
        benchmark_gap(sample_df, targets)


def test_benchmark_opportunity_propagates(sample_df):
    opp = benchmark_opportunity(sample_df, TARGETS, customer_value=1200)
    # 25% -> 30% is a 20% relative lift, so customers rise 40 * 0.2 = 8.
    assert opp.at[0, "incremental_final"] == pytest.approx(8)
    assert opp.at[0, "incremental_revenue"] == pytest.approx(9600)
    # Steps at/above or without target add nothing.
    assert opp.loc[1:3, "incremental_final"].tolist() == [0, 0, 0]
    # Last step: 180 * 25% = 45 -> +5.
    assert opp.at[4, "incremental_final"] == pytest.approx(5)


def test_total_upside_compounds(sample_df):
    total = benchmark_total_upside(sample_df, TARGETS)
    # 10,000 * .30 * .60 * .40 * .30 * .25 = 54 -> +14, more than 8 + 5.
    assert total == pytest.approx(14)
    assert total > sum(benchmark_opportunity(sample_df, TARGETS)["incremental_final"])


def test_total_upside_zero_when_all_ahead(sample_df):
    assert benchmark_total_upside(sample_df, [0.1] * 5) == 0
