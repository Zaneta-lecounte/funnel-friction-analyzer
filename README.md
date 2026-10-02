# Funnel Friction Analyzer

A Streamlit app that finds where a conversion funnel leaks, sizes what fixing each leak is worth, and ranks the fixes by impact per unit of effort.

> All data in this repo is **synthetic**.

## Problem

Growth teams usually know their funnel numbers. They rarely agree on **which step to fix first**. The usual instinct is "fix the biggest drop-off", but that ignores three things:

1. **Propagation.** A lift at one step flows through every downstream rate. What the business feels is the change in *final* conversions and revenue, not the local rate.
2. **Effort.** A 10% lift on landing-page copy and a 10% lift on the sales close rate are very different projects.
3. **Context.** A step can look bad overall and still be fine for one segment, or look fine and still sit well below a benchmark.

## Solution

Funnel Friction Analyzer turns a `stage, users` table into a ranked action list:

- **Diagnose:** stage-to-stage conversion, drop-off, users lost, cumulative conversion, and the single largest leakage point.
- **Size:** apply an assumed relative lift at each transition, push it through every downstream rate, and get incremental customers and revenue.
- **Prioritize:** priority = incremental customers ÷ effort, ranked, with a plain-English summary.
- **Compare:** see where paid and organic (or any other segments) leak differently, and how far each step sits from a target benchmark.

## What I Built

| Area | Details |
|---|---|
| **Input** | Sample funnel, CSV upload, or an editable table (add, delete or edit stages live) |
| **Validation** | Required columns, numeric and non-negative counts, unique stage names, and the **non-increasing rule**. Each violation is explained in plain English, with likely causes (stages out of order, mismatched date windows, mid-funnel entry, sessions vs. users) |
| **Diagnosis** | Step conversion, drop-off %, users lost, cumulative conversion. A Plotly funnel highlights the largest leak, which you can define as highest drop-off % or most users lost |
| **Opportunity sizing** | Lift % and effort (1–5) are editable per transition. Each lift is propagated downstream, capped at 100% |
| **Prioritization** | Ranked table (customers ÷ effort), CSV export, generated summary sentences |
| **Segments** | Long-format `segment, stage, users` CSV. Shows per-segment rates against the blended funnel, the spread between best and worst segment per step, and where each segment leaks most |
| **Benchmark** | Target rate per step: gap in pp, users short at the next stage, and the upside of closing each gap alone and all gaps together |

Example output on the sample funnel ($1,200 per customer, 10% lift, Landing Page effort = 1):

> A 10% lift at Landing Page → Form Start yields ~250 more form starts and ~4 more customers (~$4,800). Highest priority given effort.

### Project structure

```
app.py                  # Streamlit UI only — no business logic
funnel.py               # validation, conversion math, segments, benchmark gaps
opportunity_model.py    # lift propagation, sizing, priority scoring, summaries
tests/                  # pytest suite (logic + Streamlit AppTest smoke tests)
sample_funnel.csv       # synthetic B2B funnel
sample_segments.csv     # synthetic paid vs organic split (sums to sample_funnel.csv)
```

All logic lives in pure functions (DataFrames in, DataFrames out, no I/O), so every number in the UI is covered by a test.

## Methodology: Signal → Diagnosis → Priority → Action

1. **Signal:** load stage counts and validate them. A funnel where a stage has more users than the stage above it is not a funnel; it's a tracking or definition problem. The app stops and explains the problem rather than computing misleading rates.
2. **Diagnosis:** compute step conversion and losses and flag the leakiest transition. Segment and benchmark views add context: *who* leaks there, and is it actually underperforming?
3. **Priority:** size each fix by its effect on the **final stage**, not the local rate, then divide by effort.
4. **Action:** the summary turns the ranking into sentences a PM or stakeholder can act on, and the ranked table exports to CSV for a backlog.

## Opportunity-sizing formula

Let the funnel have top-of-funnel volume $U_0$ and step rates $r_1, \dots, r_n$, where $r_i = U_i / U_{i-1}$. Final conversions are:

$$U_n = U_0 \cdot r_1 \cdot r_2 \cdots r_n$$

A relative lift $L$ at transition $k$ changes only that rate, capped at 100%:

$$r_k' = \min\big(r_k(1 + L),\ 1\big)$$

Every downstream stage is then **re-propagated** with the unchanged downstream rates. The incremental users at any stage $j \ge k$ are:

$$\Delta U_j = U_j \left(\frac{r_k'}{r_k} - 1\right) \quad \xrightarrow{\text{no cap}} \quad U_j \cdot L$$

From there:

- **Incremental next-stage users** = $\Delta U_k$, e.g. $2{,}500 \times 10\% = 250$ form starts
- **Incremental final conversions** = $\Delta U_n$, e.g. $40 \times 10\% = 4$ customers
- **Incremental revenue** = $\Delta U_n \times$ average customer value, e.g. $4 \times \$1{,}200 = \$4{,}800$
- **Priority score** = $\Delta U_n \div \text{effort}$. Ties go to the larger customer gain, then to the step higher in the funnel.

**The key insight:** because the rates multiply, *the same relative lift at any step produces the same number of extra final conversions*. "Fix the biggest leak" is therefore not automatically right. With equal lifts, **effort and achievable lift** decide priority. That's why both are editable per step, and the app says so in the summary. The only exception is the 100% cap: a step already near 100% can't take the full lift, and the app flags it as *capped*.

**Benchmark upside** uses the same propagation. It replaces $r_k$ with the target rate (when the actual rate is below it) and measures $\Delta U_n$. "All targets hit" replaces every below-target rate at once, so the gains compound and exceed the sum of the single-step gains.

## How to Run

```bash
git clone https://github.com/zaneta-lecounte/funnel-friction-analyzer.git
cd funnel-friction-analyzer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py      # open http://localhost:8501
pytest -q                 # run the test suite
```

**CSV formats.** Funnel, ordered top to bottom:

```csv
stage,users
Landing Page,10000
Form Start,2500
```

Segments, with the same stages in the same order for each segment:

```csv
segment,stage,users
Paid,Landing Page,6000
Organic,Landing Page,4000
```

## Tech

- **Python 3.11+**
- **Streamlit** for the UI (data editor, tabs, AppTest for smoke tests)
- **pandas** for funnel math
- **Plotly** for the funnel, segment and benchmark charts
- **pytest** for unit tests on every pure function, plus app smoke tests

## Future Improvements

- **Uncertainty:** treat lift as a range or distribution (Monte Carlo) and rank on expected value with confidence bands, rather than a point estimate.
- **Statistical significance for segments:** add a two-proportion z-test per transition, so small segments don't produce false "divergence".
- **Time series:** upload weekly snapshots to spot when a step started leaking, and alert on regressions.
- **Cost-aware ROI:** replace the 1–5 effort score with engineering-weeks or $ cost to get a true ROI and payback period.
- **Stage-specific value:** some intermediate stages carry their own value (e.g. pipeline $ at Opportunity), not only the final stage.
- **Experiment planning:** estimate the sample size and test duration needed to detect each assumed lift.
- **Data connectors:** read directly from GA4, Amplitude, Mixpanel or a warehouse instead of CSV.
