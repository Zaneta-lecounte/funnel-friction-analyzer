"""Opportunity sizing: lift propagation, incremental impact, priority scoring and summaries.

Model
-----
A funnel with top-of-funnel volume ``U0`` and step rates ``r1..rn`` ends with
``U_n = U0 * r1 * r2 * ... * rn`` final-stage conversions.

Lifting step ``k`` by a relative amount ``L`` changes only ``r_k``:

    r_k' = min(r_k * (1 + L), 1.0)

and every downstream stage is re-propagated with the *unchanged* downstream
rates. So the incremental users at stage ``j >= k`` are

    ΔU_j = U0 * r1 ... r_k' ... r_j  -  U_j  =  U_j * (r_k' / r_k - 1)

which, when the cap does not bind, is simply ``U_j * L``. Incremental revenue is
``ΔU_n * customer_value`` and ``priority = ΔU_n / effort``.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from funnel import benchmark_gap, clean_funnel, step_rates, transition_label

DEFAULT_LIFT = 0.10
DEFAULT_EFFORT = 3
EFFORT_MIN, EFFORT_MAX = 1, 5


def propagate(top_users: float, rates: Sequence[float]) -> list[float]:
    """Users at every stage given top-of-funnel volume and step rates (len = len(rates)+1)."""
    users = [float(top_users)]
    for r in rates:
        users.append(users[-1] * float(r))
    return users


def apply_lift(rates: Sequence[float], index: int, lift: float) -> list[float]:
    """Return rates with ``rates[index]`` raised by relative ``lift``, capped at 100%.

    A rate already above 100% (invalid data) is left unchanged rather than reduced.
    """
    if lift < 0:
        raise ValueError("lift must be >= 0")
    out = [float(r) for r in rates]
    out[index] = max(out[index], min(out[index] * (1 + lift), 1.0))
    return out


def _broadcast(value: float | Sequence[float], n: int, name: str) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * n
    values = [float(v) for v in value]
    if len(values) != n:
        raise ValueError(f"{name} needs {n} values (one per transition), got {len(values)}")
    return values


def size_opportunities(
    df: pd.DataFrame,
    lifts: float | Sequence[float] = DEFAULT_LIFT,
    efforts: float | Sequence[float] = DEFAULT_EFFORT,
    customer_value: float = 0.0,
) -> pd.DataFrame:
    """Size the lift opportunity at every transition and rank by priority.

    Returns one row per transition in funnel order with columns:
    transition, from_stage, to_stage, current_rate, lift, lifted_rate, capped,
    incremental_next, incremental_final, incremental_revenue, effort,
    priority_score, rank (1 = do first).
    """
    df = clean_funnel(df)
    stages = df["stage"].astype(str).tolist()
    users = df["users"].astype(float).tolist()
    rates = step_rates(users)
    n = len(rates)

    lifts_ = _broadcast(lifts, n, "lifts")
    efforts_ = _broadcast(efforts, n, "efforts")
    if any(l < 0 for l in lifts_):
        raise ValueError("lifts must be >= 0")
    if any(not EFFORT_MIN <= e <= EFFORT_MAX for e in efforts_):
        raise ValueError(f"efforts must be between {EFFORT_MIN} and {EFFORT_MAX}")
    if customer_value < 0:
        raise ValueError("customer_value must be >= 0")

    baseline = propagate(users[0], rates)
    rows = []
    for k in range(n):
        lifted = apply_lift(rates, k, lifts_[k])
        new_users = propagate(users[0], lifted)
        inc_next = new_users[k + 1] - baseline[k + 1]
        inc_final = new_users[-1] - baseline[-1]
        rows.append({
            "transition": transition_label(stages[k], stages[k + 1]),
            "from_stage": stages[k],
            "to_stage": stages[k + 1],
            "current_rate": rates[k],
            "lift": lifts_[k],
            "lifted_rate": lifted[k],
            "capped": rates[k] * (1 + lifts_[k]) > lifted[k] + 1e-12,
            "incremental_next": inc_next,
            "incremental_final": inc_final,
            "incremental_revenue": inc_final * customer_value,
            "effort": efforts_[k],
            "priority_score": inc_final / efforts_[k],
        })

    out = pd.DataFrame(rows)
    out["rank"] = rank_order(out)
    return out


def rank_order(opps: pd.DataFrame) -> list[int]:
    """Rank 1..n: priority desc, then incremental customers desc, then funnel order."""
    order = sorted(
        range(len(opps)),
        key=lambda i: (-round(opps.at[i, "priority_score"], 9),
                       -round(opps.at[i, "incremental_final"], 9), i),
    )
    ranks = [0] * len(opps)
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    return ranks


def ranked(opps: pd.DataFrame) -> pd.DataFrame:
    """Opportunities sorted by rank."""
    return opps.sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------- summaries

def pluralize(stage: str) -> str:
    """Rough English plural of a stage name, lower-cased: 'Form Start' -> 'form starts'."""
    words = stage.strip().lower().split()
    if not words:
        return stage
    last = words[-1]
    if last.endswith("s"):
        pass
    elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        last = last[:-1] + "ies"
    elif last.endswith(("ch", "sh", "x", "z")):
        last += "es"
    else:
        last += "s"
    return " ".join(words[:-1] + [last])


def approx(n: float) -> str:
    """Human-friendly approximate count: '~250', '~4', '~0.4'."""
    if abs(n) >= 1 or n == 0:
        return f"~{round(n):,}"
    return f"~{n:.1f}"


def money(v: float) -> str:
    return f"${v:,.0f}"


def describe_opportunity(row: pd.Series, final_stage: str, customer_value: float,
                         total: int) -> str:
    """One plain-English sentence (plus priority note) for a sized opportunity."""
    pct = row["lift"] * 100
    lift = f"{row['lift']:.0%}" if abs(pct - round(pct)) < 1e-9 else f"{row['lift']:.1%}"
    head = f"A {lift} lift at {row['transition']} yields "
    final_part = f"{approx(row['incremental_final'])} more {pluralize(final_stage)}"
    if customer_value > 0:
        final_part += f" (~{money(row['incremental_revenue'])})"

    if row["to_stage"] == final_stage:
        body = final_part
    else:
        body = f"{approx(row['incremental_next'])} more {pluralize(row['to_stage'])} and {final_part}"

    note = ""
    if row["capped"]:
        note = f" (Capped: this step can't exceed 100%, so the full {lift} isn't achievable.)"

    if row["rank"] == 1:
        priority = " Highest priority given effort."
    else:
        priority = (f" Priority #{int(row['rank'])} of {total} "
                    f"(effort {row['effort']:g}/{EFFORT_MAX}).")
    return head + body + "." + note + priority


def build_summary(df: pd.DataFrame, opps: pd.DataFrame, customer_value: float,
                  leak_transition: str | None = None) -> list[str]:
    """Plain-English bullet points: overall conversion, leak, insight, ranked opportunities."""
    df = clean_funnel(df)
    stages = df["stage"].astype(str).tolist()
    users = df["users"].astype(float).tolist()
    top, final = users[0], users[-1]
    overall = final / top if top else 0.0

    lines = [
        f"Of {top:,.0f} users at {stages[0]}, {final:,.0f} reach {stages[-1]} "
        f"— {overall:.2%} end-to-end conversion."
    ]
    if leak_transition:
        lines.append(f"The largest leakage point is {leak_transition}.")

    uniform = opps["lift"].nunique() == 1 and not opps["capped"].any()
    if uniform and len(opps) > 1:
        lines.append(
            "Because stage rates multiply, the same relative lift at any step adds the same "
            "number of final conversions — so effort is what separates the priorities."
        )

    for _, row in ranked(opps).iterrows():
        lines.append(describe_opportunity(row, stages[-1], customer_value, len(opps)))
    return lines


# ---------------------------------------------------------------- benchmarks

def benchmark_opportunity(df: pd.DataFrame, targets: Sequence[float | None],
                          customer_value: float = 0.0) -> pd.DataFrame:
    """Benchmark gaps plus the final-stage upside of closing each gap on its own.

    Adds to ``funnel.benchmark_gap``: incremental_final (extra final-stage users if only
    this step reached its target, re-propagated through actual downstream rates) and
    incremental_revenue. Steps at/above target or without a target contribute 0.
    """
    gap = benchmark_gap(df, list(targets))
    df = clean_funnel(df)
    users = df["users"].astype(float).tolist()
    rates = step_rates(users)
    baseline_final = propagate(users[0], rates)[-1]

    inc = []
    for k, row in gap.iterrows():
        if row["status"] != "Below":
            inc.append(0.0)
            continue
        new_rates = list(rates)
        new_rates[k] = row["target_rate"]
        inc.append(propagate(users[0], new_rates)[-1] - baseline_final)
    gap["incremental_final"] = inc
    gap["incremental_revenue"] = gap["incremental_final"] * customer_value
    return gap


def benchmark_total_upside(df: pd.DataFrame, targets: Sequence[float | None]) -> float:
    """Extra final-stage users if *every* below-target step reached its target at once.

    Compounds, so it exceeds the sum of single-step upsides.
    """
    df = clean_funnel(df)
    users = df["users"].astype(float).tolist()
    rates = step_rates(users)
    lifted = [
        max(r, float(t)) if t is not None and not pd.isna(t) else r
        for r, t in zip(rates, targets)
    ]
    return propagate(users[0], lifted)[-1] - propagate(users[0], rates)[-1]
