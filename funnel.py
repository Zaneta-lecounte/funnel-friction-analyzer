"""Funnel conversion math and validation.

All functions are pure: they take DataFrames / plain values and return new
objects without side effects. Rates are fractions (0.25 == 25%).

A funnel is a DataFrame ordered top -> bottom with columns ``stage`` and
``users``. A *transition* ``i`` (1..n-1) is the step from stage ``i-1`` to
stage ``i``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = ("stage", "users")
ARROW = "→"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a funnel. ``errors`` block analysis; ``warnings`` don't."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _fmt(n: float) -> str:
    return f"{n:,.0f}" if float(n).is_integer() else f"{n:,.2f}"


def transition_label(from_stage: str, to_stage: str) -> str:
    return f"{from_stage} {ARROW} {to_stage}"


def clean_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalised column names, stripped stage names and numeric users.

    Rows where both stage and users are blank (e.g. empty editor rows) are dropped.
    Non-numeric users become NaN so that validation can report them.
    """
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "stage" in out.columns:
        out["stage"] = out["stage"].astype("string").str.strip()
    if "users" in out.columns:
        out["users"] = pd.to_numeric(out["users"], errors="coerce")
    if set(REQUIRED_COLUMNS).issubset(out.columns):
        blank_stage = out["stage"].isna() | (out["stage"] == "")
        out = out[~(blank_stage & out["users"].isna())]
    return out.reset_index(drop=True)


def validate_funnel(df: pd.DataFrame) -> ValidationResult:
    """Check structure and the non-increasing rule, explaining every violation."""
    errors: list[str] = []
    warnings: list[str] = []

    df = clean_funnel(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return ValidationResult(
            errors=[f"Missing required column(s): {', '.join(missing)}. "
                    "Expected columns `stage, users` ordered top to bottom."]
        )

    if len(df) < 2:
        errors.append("A funnel needs at least 2 stages to compute conversion.")

    stage_blank = df["stage"].isna() | (df["stage"] == "")
    for i in df.index[stage_blank]:
        errors.append(f"Row {i + 1}: stage name is blank.")

    dupes = df.loc[~stage_blank, "stage"]
    for name in dupes[dupes.duplicated()].unique():
        errors.append(f"Stage '{name}' appears more than once; stage names must be unique.")

    users = df["users"]
    for i in df.index[users.isna()]:
        errors.append(f"Row {i + 1} ({df.at[i, 'stage']}): users must be a number.")
    for i in df.index[users < 0]:
        errors.append(f"Row {i + 1} ({df.at[i, 'stage']}): users cannot be negative "
                      f"(got {_fmt(users[i])}).")

    if len(df) >= 1 and pd.notna(users.iloc[0]) and users.iloc[0] == 0:
        errors.append("The top stage has 0 users, so no conversion rates can be computed.")

    for i in range(1, len(df)):
        prev, cur = users.iloc[i - 1], users.iloc[i]
        if pd.isna(prev) or pd.isna(cur):
            continue
        if cur > prev:
            implied = f" ({cur / prev:.0%} step conversion)" if prev > 0 else ""
            errors.append(
                f"Non-increasing rule violated: '{df.at[i, 'stage']}' has {_fmt(cur)} users but "
                f"the stage above it, '{df.at[i - 1, 'stage']}', has only {_fmt(prev)}{implied}. "
                "Every user in a stage must have passed through the stage above, so counts can "
                "only stay flat or fall. Common causes: stages listed out of order, different "
                "date windows per stage, users entering mid-funnel (e.g. direct links to the "
                "form), or mixing units (sessions vs. users)."
            )

    if not errors:
        zero_at = next((i for i in range(1, len(df)) if users.iloc[i] == 0), None)
        if zero_at is not None and zero_at < len(df) - 1:
            warnings.append(
                f"'{df.at[zero_at, 'stage']}' has 0 users, so every stage below it is 0 and "
                "their step conversion rates are treated as 0%."
            )
        if any(not float(u).is_integer() for u in users):
            warnings.append("Some user counts are not whole numbers; results are still computed.")

    return ValidationResult(errors=errors, warnings=warnings)


def _safe_rate(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def step_rates(users: list[float] | pd.Series) -> list[float]:
    """Stage-to-stage conversion rates; length is ``len(users) - 1``. 0/0 is treated as 0."""
    u = [float(x) for x in users]
    return [_safe_rate(u[i], u[i - 1]) for i in range(1, len(u))]


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-stage metrics.

    Columns: stage, users, step_conversion, drop_off, users_lost, cumulative_conversion.
    The top stage has NaN step_conversion/drop_off (there is no previous stage),
    0 users_lost and cumulative_conversion 1.0.
    """
    df = clean_funnel(df)
    users = df["users"].astype(float).tolist()
    top = users[0]
    rates = step_rates(users)

    out = pd.DataFrame({"stage": df["stage"].astype(str).tolist(), "users": users})
    out["step_conversion"] = [float("nan")] + rates
    out["drop_off"] = [float("nan")] + [1 - r for r in rates]
    out["users_lost"] = [0.0] + [users[i - 1] - users[i] for i in range(1, len(users))]
    out["cumulative_conversion"] = [_safe_rate(u, top) for u in users]
    return out


def transition_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per transition (stage i-1 -> stage i).

    Columns: transition, from_stage, to_stage, from_users, to_users, rate, drop_off, users_lost.
    """
    m = compute_metrics(df)
    rows = []
    for i in range(1, len(m)):
        rows.append({
            "transition": transition_label(m.at[i - 1, "stage"], m.at[i, "stage"]),
            "from_stage": m.at[i - 1, "stage"],
            "to_stage": m.at[i, "stage"],
            "from_users": m.at[i - 1, "users"],
            "to_users": m.at[i, "users"],
            "rate": m.at[i, "step_conversion"],
            "drop_off": m.at[i, "drop_off"],
            "users_lost": m.at[i, "users_lost"],
        })
    return pd.DataFrame(rows)


LEAKAGE_METRICS = {
    "drop_off": "Highest drop-off %",
    "users_lost": "Most users lost",
}


def largest_leakage(df: pd.DataFrame, by: str = "drop_off") -> int:
    """Row index (into ``transition_table``) of the leakiest transition.

    ``by`` is ``"drop_off"`` (relative friction) or ``"users_lost"`` (absolute volume).
    Ties resolve to the topmost transition.
    """
    if by not in LEAKAGE_METRICS:
        raise ValueError(f"by must be one of {list(LEAKAGE_METRICS)}, got {by!r}")
    t = transition_table(df)
    return int(t[by].idxmax())
