# Hajj Risk Study

Single-Agent baseline for risk classification using simulated Hajj crowd-management data.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-study.txt
python -m unittest discover -s tests/study -p 'test_*.py'
```

## Data and execution

Download the [Hajj and Umrah Crowd Management dataset](https://www.kaggle.com/datasets/ziya07/hajj-and-umrah-crowd-management-dataset). Set `STUDY_RAW_CSV` to its local absolute path, and keep the source data outside Git.

```bash
python -m hajj_study.audit_data --csv "$STUDY_RAW_CSV"
python -m hajj_study.split_data --csv "$STUDY_RAW_CSV" --out artifacts/row_splits.csv
python -m hajj_study.train_model --csv "$STUDY_RAW_CSV" --splits artifacts/row_splits.csv --out artifacts/model
python -m hajj_study.cases --csv "$STUDY_RAW_CSV" --splits artifacts/row_splits.csv --predictions artifacts/model/predictions.jsonl --out artifacts/g3_cases
python -m hajj_study.g5 preflight --snapshot-root . --source-csv "$STUDY_RAW_CSV" --private-output artifacts/g5
```

Set `GEMINI_API_KEY` in the environment before live calls. G5 smoke, validation, and freeze operate on private outputs.

## M4 evidence

Single-Agent implementation, real Gemini smoke (10/10), validation (40/40), and successful clean-clone offline tests. Private data and model outputs are not published.

## M5 evidence

30 frozen golden test cases, seeds 11/42/77: 90 Single-Agent test runs and 27 clean references (117/117 complete). Correct issued decisions: 37/90. Coverage: 76.67% ± 3.33 percentage points; selective error: 46.44% ± 3.30 percentage points (mean ± sample SD across three seeds). Offline scoring covers end-to-end, component, and trajectory-level metrics. Gold labels and traces stay private.

After G5 freeze, run:

```bash
python -m hajj_study.g6 run
python -m hajj_study.score_runs
```
