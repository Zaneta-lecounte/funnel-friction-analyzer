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


# ---------------------------------------------------------------- segments

SEGMENT_COLUMN = "segment"
BLENDED = "All segments"


def split_segments(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a long ``segment, stage, users`` table into one funnel per segment.

    Segments and stages keep their order of first appearance.
    """
    df = clean_funnel(df)
    df[SEGMENT_COLUMN] = df[SEGMENT_COLUMN].astype("string").str.strip()
    return {
        str(seg): grp[["stage", "users"]].reset_index(drop=True)
        for seg, grp in df.groupby(SEGMENT_COLUMN, sort=False)
    }


def validate_segments(df: pd.DataFrame) -> ValidationResult:
    """Validate a segmented funnel: segment column present, same stages in the same
    order for every segment, and every segment funnel valid on its own."""
    cleaned = clean_funnel(df)
    if SEGMENT_COLUMN not in cleaned.columns:
        return ValidationResult(errors=[
            "Missing required column: segment. Expected columns `segment, stage, users`."])
    missing = [c for c in REQUIRED_COLUMNS if c not in cleaned.columns]
    if missing:
        return ValidationResult(errors=[
            f"Missing required column(s): {', '.join(missing)}. "
            "Expected columns `segment, stage, users`."])
    if cleaned[SEGMENT_COLUMN].isna().any():
        return ValidationResult(errors=["Some rows have a blank segment."])

    segments = split_segments(df)
    errors: list[str] = []
    warnings: list[str] = []
    if len(segments) < 2:
        errors.append("Segment comparison needs at least 2 segments.")

    reference_name, reference = next(iter(segments.items()))
    ref_stages = reference["stage"].tolist()
    for name, seg in segments.items():
        if seg["stage"].tolist() != ref_stages:
            errors.append(
                f"Segment '{name}' has stages {seg['stage'].tolist()} but '{reference_name}' "
                f"has {ref_stages}. Every segment must list the same stages in the same order "
                "so transitions can be compared."
            )
            continue
        result = validate_funnel(seg)
        errors += [f"[{name}] {e}" for e in result.errors]
        warnings += [f"[{name}] {w}" for w in result.warnings]
    return ValidationResult(errors=errors, warnings=warnings)


def blended_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Sum users per stage across segments (stage order preserved)."""
    segments = split_segments(df)
    stages = next(iter(segments.values()))["stage"].tolist()
    totals = sum(seg["users"].to_numpy() for seg in segments.values())
    return pd.DataFrame({"stage": stages, "users": totals})


def segment_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Long table of transition metrics per segment, plus the blended funnel.

    Columns: segment, transition, from_stage, to_stage, from_users, to_users, rate,
    drop_off, users_lost, rate_vs_blended_pp (segment rate minus blended rate).
    """
    blended = transition_table(blended_funnel(df))
    frames = []
    for name, seg in [*split_segments(df).items(), (BLENDED, blended_funnel(df))]:
        t = transition_table(seg)
        t.insert(0, "segment", name)
        t["rate_vs_blended_pp"] = (t["rate"] - blended["rate"]) * 100
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def segment_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """Per transition: each segment's rate, the spread between best and worst
    segment (percentage points) and which segments those are. Sorted by spread desc."""
    comp = segment_comparison(df)
    comp = comp[comp["segment"] != BLENDED]
    wide = comp.pivot(index="transition", columns="segment", values="rate")
    order = comp["transition"].drop_duplicates().tolist()
    seg_order = comp["segment"].drop_duplicates().tolist()
    wide = wide.loc[order, seg_order]

    out = wide.copy()
    out.columns = [f"rate · {c}" for c in out.columns]
    out["best_segment"] = wide.idxmax(axis=1)
    out["worst_segment"] = wide.idxmin(axis=1)
    out["spread_pp"] = (wide.max(axis=1) - wide.min(axis=1)) * 100
    out = out.reset_index()
    out["funnel_order"] = range(len(out))
    return (out.sort_values(["spread_pp", "funnel_order"], ascending=[False, True])
               .drop(columns="funnel_order").reset_index(drop=True))


def segment_insights(df: pd.DataFrame, by: str = "drop_off") -> list[str]:
    """Plain-English notes on where each segment leaks and where segments differ most."""
    lines = []
    div = segment_divergence(df)
    top = div.iloc[0]
    best, worst = top["best_segment"], top["worst_segment"]
    lines.append(
        f"Segments diverge most at {top['transition']}: {best} converts "
        f"{top[f'rate · {best}']:.1%} vs {worst} at {top[f'rate · {worst}']:.1%} "
        f"({top['spread_pp']:.1f} pp gap). Fixing {worst}'s experience here is the "
        "segment-specific opportunity."
    )
    for name, seg in split_segments(df).items():
        t = transition_table(seg)
        i = largest_leakage(seg, by)
        overall = seg["users"].iloc[-1] / seg["users"].iloc[0]
        lines.append(
            f"{name} leaks most at {t.at[i, 'transition']} ({t.at[i, 'drop_off']:.1%} drop-off, "
            f"{t.at[i, 'users_lost']:,.0f} users lost); end-to-end conversion {overall:.2%}."
        )
    return lines


# ---------------------------------------------------------------- benchmarks

def benchmark_gap(df: pd.DataFrame, targets: list[float | None]) -> pd.DataFrame:
    """Compare each transition's actual rate with a target rate.

    ``targets`` has one entry per transition (fraction, or None/NaN for "no target").
    Columns: transition, actual_rate, target_rate, gap_pp (actual - target, in
    percentage points), status ("Below", "At/above", "No target") and users_gap —
    users the next stage is short of target given the actual users entering the step
    (negative means ahead of target).
    """
    t = transition_table(df)
    if len(targets) != len(t):
        raise ValueError(f"targets needs {len(t)} values (one per transition), got {len(targets)}")
    target = pd.Series([float("nan") if v is None else float(v) for v in targets], dtype=float)
    if ((target < 0) | (target > 1)).any():
        raise ValueError("target rates must be between 0 and 1")

    out = pd.DataFrame({
        "transition": t["transition"],
        "actual_rate": t["rate"],
        "target_rate": target,
        "gap_pp": (t["rate"] - target) * 100,
        "users_gap": t["from_users"] * target - t["to_users"],
    })
    out["status"] = [
        "No target" if pd.isna(tr) else ("Below" if ar < tr - 1e-12 else "At/above")
        for ar, tr in zip(out["actual_rate"], out["target_rate"])
    ]
    return out
