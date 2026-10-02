# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

## Project
Funnel Friction Analyzer — a Streamlit app that diagnoses where a conversion funnel
leaks and sizes/prioritises the fix opportunities. Portfolio project; **synthetic data only**.

## Layout
- `app.py` — Streamlit UI only. No business logic; it calls the modules below.
- `funnel.py` — validation + conversion math (stage-to-stage rates, drop-off, cumulative,
  leakage, segment comparison, benchmark gaps).
- `opportunity_model.py` — lift propagation, incremental customers/revenue, priority scoring,
  plain-English summaries.
- `tests/` — pytest suite for all logic.
- `sample_funnel.csv`, `sample_segments.csv` — synthetic demo data.

## Conventions
- All logic lives in **pure functions** (inputs in, DataFrame/dataclass out, no I/O, no
  Streamlit). Every new function gets tests.
- Funnel input is ordered top → bottom with columns `stage, users`; segment input adds `segment`.
- Rates are fractions (0.25), not percents, inside the modules; the UI formats them.
- A "transition" is index `i` (1..n-1) meaning stage `i-1 → i`.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), one per feature.

## Commands
```bash
pip install -r requirements.txt
pytest -q              # run tests
streamlit run app.py   # launch the app
```
