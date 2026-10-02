import pytest

from conftest import make_funnel
from funnel import compute_metrics
from opportunity_model import (
    apply_lift,
    approx,
    build_summary,
    describe_opportunity,
    pluralize,
    propagate,
    rank_order,
    ranked,
    size_opportunities,
)

SAMPLE_EFFORT = [1, 2, 3, 3, 4]


# ---------- propagation ----------

def test_propagate_reproduces_funnel(sample_df):
    rates = compute_metrics(sample_df)["step_conversion"].iloc[1:].tolist()
    users = propagate(10000, rates)
    assert users == pytest.approx(sample_df["users"].tolist())


def test_apply_lift_only_changes_target_rate():
    assert apply_lift([0.5, 0.5, 0.5], 1, 0.10) == pytest.approx([0.5, 0.55, 0.5])


def test_apply_lift_caps_at_100_percent():
    assert apply_lift([0.95], 0, 0.10) == [1.0]


def test_apply_lift_never_reduces_invalid_rate():
    assert apply_lift([1.2], 0, 0.10) == [1.2]


def test_apply_lift_rejects_negative():
    with pytest.raises(ValueError):
        apply_lift([0.5], 0, -0.1)


# ---------- sizing ----------

def test_worked_example_from_brief(sample_df):
    """10% at Landing Page → Form Start: ~250 form starts, ~4 customers, ~$4,800 at $1,200."""
    opps = size_opportunities(sample_df, 0.10, SAMPLE_EFFORT, customer_value=1200)
    first = opps.iloc[0]
    assert first["transition"] == "Landing Page → Form Start"
    assert first["current_rate"] == pytest.approx(0.25)
    assert first["lifted_rate"] == pytest.approx(0.275)
    assert first["incremental_next"] == pytest.approx(250)
    assert first["incremental_final"] == pytest.approx(4)
    assert first["incremental_revenue"] == pytest.approx(4800)
    assert first["priority_score"] == pytest.approx(4)
    assert first["rank"] == 1


def test_uniform_lift_gives_equal_final_impact(sample_df):
    opps = size_opportunities(sample_df, 0.10)
    assert opps["incremental_final"].tolist() == pytest.approx([4.0] * 5)


def test_incremental_next_is_downstream_users_times_lift(sample_df):
    opps = size_opportunities(sample_df, 0.10)
    assert opps["incremental_next"].tolist() == pytest.approx([250, 150, 60, 18, 4])


def test_propagation_matches_closed_form_with_mixed_lifts(sample_df):
    lifts = [0.05, 0.20, 0.0, 0.50, 0.10]
    opps = size_opportunities(sample_df, lifts)
    for i, lift in enumerate(lifts):
        assert opps.at[i, "incremental_final"] == pytest.approx(40 * lift)


def test_zero_lift_means_zero_impact(sample_df):
    opps = size_opportunities(sample_df, 0.0)
    assert (opps["incremental_final"] == 0).all()


def test_cap_limits_impact_and_is_flagged():
    df = make_funnel([("A", 1000), ("B", 950), ("C", 95)])
    opps = size_opportunities(df, 0.10)
    assert bool(opps.at[0, "capped"]) is True
    assert opps.at[0, "lifted_rate"] == 1.0
    assert opps.at[0, "incremental_next"] == pytest.approx(50)
    assert opps.at[0, "incremental_final"] == pytest.approx(5)  # 1000*1.0*0.1 - 95
    assert bool(opps.at[1, "capped"]) is False
    assert opps.at[1, "incremental_final"] == pytest.approx(9.5)


def test_zero_mid_funnel_produces_zero_downstream_impact():
    df = make_funnel([("A", 100), ("B", 0), ("C", 0)])
    opps = size_opportunities(df, 0.10)
    assert opps["incremental_final"].tolist() == [0.0, 0.0]


def test_priority_is_customers_over_effort(sample_df):
    opps = size_opportunities(sample_df, 0.10, SAMPLE_EFFORT)
    assert opps["priority_score"].tolist() == pytest.approx([4 / e for e in SAMPLE_EFFORT])


def test_ranking_prefers_low_effort_then_funnel_order(sample_df):
    opps = size_opportunities(sample_df, 0.10, [3, 1, 3, 1, 5])
    order = ranked(opps)["transition"].tolist()
    assert order == [
        "Form Start → Form Complete",
        "Qualified Lead → Opportunity",
        "Landing Page → Form Start",
        "Form Complete → Qualified Lead",
        "Opportunity → Customer",
    ]


def test_ranking_uses_lift_when_effort_equal(sample_df):
    opps = size_opportunities(sample_df, [0.1, 0.1, 0.1, 0.1, 0.3], 3)
    assert ranked(opps).at[0, "transition"] == "Opportunity → Customer"


def test_rank_order_is_a_permutation(sample_df):
    opps = size_opportunities(sample_df, 0.1, SAMPLE_EFFORT)
    assert sorted(rank_order(opps)) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("kwargs", [
    {"lifts": [0.1, 0.1]},
    {"lifts": -0.1},
    {"efforts": 0},
    {"efforts": 6},
    {"customer_value": -1},
])
def test_invalid_inputs_raise(sample_df, kwargs):
    with pytest.raises(ValueError):
        size_opportunities(sample_df, **kwargs)


# ---------- summaries ----------

@pytest.mark.parametrize("stage,expected", [
    ("Form Start", "form starts"),
    ("Opportunity", "opportunities"),
    ("Customer", "customers"),
    ("Qualified Lead", "qualified leads"),
    ("Sales", "sales"),
    ("Day", "days"),
    ("Checkout Match", "checkout matches"),
])
def test_pluralize(stage, expected):
    assert pluralize(stage) == expected


def test_approx():
    assert approx(249.6) == "~250"
    assert approx(4.0) == "~4"
    assert approx(0.42) == "~0.4"
    assert approx(12345) == "~12,345"


def test_describe_matches_brief(sample_df):
    opps = size_opportunities(sample_df, 0.10, SAMPLE_EFFORT, customer_value=1200)
    text = describe_opportunity(opps.iloc[0], "Customer", 1200, len(opps))
    assert text == (
        "A 10% lift at Landing Page → Form Start yields ~250 more form starts and "
        "~4 more customers (~$4,800). Highest priority given effort."
    )


def test_describe_last_step_and_no_revenue(sample_df):
    opps = size_opportunities(sample_df, 0.10, SAMPLE_EFFORT)
    text = describe_opportunity(opps.iloc[4], "Customer", 0, len(opps))
    assert text == "A 10% lift at Opportunity → Customer yields ~4 more customers. Priority #5 of 5 (effort 4/5)."


def test_describe_mentions_cap():
    df = make_funnel([("A", 1000), ("B", 950), ("C", 95)])
    opps = size_opportunities(df, 0.10)
    assert "Capped" in describe_opportunity(opps.iloc[0], "C", 0, 2)


def test_build_summary(sample_df):
    opps = size_opportunities(sample_df, 0.10, SAMPLE_EFFORT, customer_value=1200)
    lines = build_summary(sample_df, opps, 1200, "Opportunity → Customer")
    assert lines[0] == ("Of 10,000 users at Landing Page, 40 reach Customer — 0.40% end-to-end conversion.")
    assert "Opportunity → Customer" in lines[1]
    assert "same relative lift" in lines[2]
    assert lines[3].startswith("A 10% lift at Landing Page → Form Start")
    assert len(lines) == 3 + 5


def test_build_summary_skips_uniform_note_when_lifts_differ(sample_df):
    opps = size_opportunities(sample_df, [0.1, 0.2, 0.1, 0.1, 0.1])
    assert not any("same relative lift" in l for l in build_summary(sample_df, opps, 0))
