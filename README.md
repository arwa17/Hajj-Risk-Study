# Hajj Risk Study

Controlled comparison of Single-Agent and Multi-Agent risk classification using simulated Hajj crowd-management data.

## Architecture

- **Single-Agent:** One generalist analyzes crowd, health, and environment observations.
- **Multi-Agent:** Three domain specialists send structured JSON reports to a coordinator.
- **Decision Gate:** Issues a risk decision, verifies one domain at most, or abstains.

Both conditions use the same cases, shared ML predictions, decision policy, Gemini configuration, and seeds (11, 42, 77).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-study.txt
python -m unittest discover -s tests/study -p 'test_*.py'
```

## Run

Download the [Hajj and Umrah Crowd Management dataset](https://www.kaggle.com/datasets/ziya07/hajj-and-umrah-crowd-management-dataset) and set `STUDY_RAW_CSV` to the absolute path of the original CSV. The data are simulated.

```bash
export STUDY_RAW_CSV="/absolute/path/to/original.csv"
python -m hajj_study.audit_data --csv "$STUDY_RAW_CSV"
python -m hajj_study.split_data --csv "$STUDY_RAW_CSV" --out artifacts/row_splits.csv
python -m hajj_study.train_model --csv "$STUDY_RAW_CSV" --splits artifacts/row_splits.csv --out artifacts/model
python -m hajj_study.cases --csv "$STUDY_RAW_CSV" --splits artifacts/row_splits.csv --predictions artifacts/model/predictions.jsonl --out artifacts/g3_cases
```

Set `GEMINI_API_KEY` privately before running live evaluation. Use an unfrozen working directory and run the following commands in order:

```bash
python -m hajj_study.g5 preflight --snapshot-root . --source-csv "$STUDY_RAW_CSV" --private-output artifacts/g5
python -m hajj_study.g5 smoke --private-output artifacts/g5
python -m hajj_study.g5 validation --private-output artifacts/g5
python -m hajj_study.g5 freeze --private-output artifacts/g5
python -m hajj_study.g6 run
python -m hajj_study.score_runs
python -m hajj_study.g7 smoke
python -m hajj_study.g7 run --max-cases 117 --continue-on-failure
python -m hajj_study.m6_results
```

Frozen experiments must not be changed or overwritten. Raw data, gold labels, model artifacts, and traces remain private.

## Results

| Metric | Single-Agent | Multi-Agent |
|---|---:|---:|
| Correct decisions | 37/90 | 31/90 |
| Coverage | 76.67% ± 3.33 pp | 82.22% ± 3.85 pp |
| Selective error | 46.44% ± 3.30 pp | 58.12% ± 0.37 pp |
| API attempts/case | 1.411 ± 0.038 | 5.156 ± 0.506 |
| Recorded execution time | 1.870 ± 0.280 s | 12.853 ± 0.263 s |

Percentages and per-case measures use mean ± sample SD across three seeds; correct decisions are totals. The timing excludes deliberate pacing delays. A complete monetary cost was unavailable because 42 Multi-Agent attempts had incomplete token usage metadata.

## M4–M6 Evidence

- **M4 (C4.1–C4.3):** 10/10 live Gemini smoke calls, 40/40 Validation evaluations, a working Single-Agent baseline, and successful clean-clone offline tests.
- **M5 (C5.1–C5.4):** 30 golden test cases; 90 primary and 27 clean-reference Single-Agent runs (117/117 completed); automated scoring, three-seed variance, and trajectory-safe completion (64.44%).
- **M6 (C6.1–C6.4):** Three specialists and one coordinator, matched cases and seeds, 117 recorded treatment runs (116 completed; one `INVALID_REPORT` failure), variance, cost/latency measures, and trace-based failure analysis.

All incorrect issued decisions were over-escalations relative to reconstructed reference labels. Verification did not correct the HIGH risk prediction in case HJ-T028. The results do not establish real-world safety performance.
