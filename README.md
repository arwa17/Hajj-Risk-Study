# Hajj Risk Study

Controlled comparison of Single-Agent and Multi-Agent risk classification using simulated Hajj crowd-management data.

## Architecture

- **Single-Agent:** One generalist analyzes crowd, health, and environment observations.
- **Multi-Agent:** Three domain specialists send structured JSON reports to a coordinator.
- **Decision Gate:** Issues a risk decision, verifies one domain at most, or abstains.

Both conditions use the same cases, shared ML predictions, decision policy, Gemini configuration, and seeds (11, 42, 77).

The main experimental difference is the agent organization: one generalist versus three specialists and one coordinator. The Multi-Agent design also entails more role invocations, so operational costs are an observed consequence of the architecture, not a separately controlled factor.

## Setup

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-study.txt
python -m unittest discover -s tests/study -p 'test_*.py'
```

### Windows CMD

Open Command Prompt in the project root directory.

```cmd
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-study.txt
.venv\Scripts\python.exe -m unittest discover -s tests/study -p "test_*.py"
```

If the virtual environment already exists, there is no need to recreate it.

To run the offline tests using the existing environment:

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests/study -p "test_*.py"
```

These tests do not require new Gemini API calls. Tests that depend on unavailable private files may be skipped.

In the reviewed local environment, 134 tests were discovered: 133 passed and one was skipped because private files were unavailable. The skipped class-level integration fixture also prevented its six dependent integration test methods from executing; these methods must not be reported as passed.

## Run

Download the [Hajj and Umrah Crowd Management dataset](https://www.kaggle.com/datasets/ziya07/hajj-and-umrah-crowd-management-dataset) and set `STUDY_RAW_CSV` to the absolute path of the original CSV. The data are simulated.

### Linux / macOS

```bash
export STUDY_RAW_CSV="/absolute/path/to/original.csv"
python -m hajj_study.audit_data --csv "$STUDY_RAW_CSV"
python -m hajj_study.split_data --csv "$STUDY_RAW_CSV" --out artifacts/row_splits.csv
python -m hajj_study.train_model --csv "$STUDY_RAW_CSV" --splits artifacts/row_splits.csv --out artifacts/model
python -m hajj_study.cases --csv "$STUDY_RAW_CSV" --splits artifacts/row_splits.csv --predictions artifacts/model/predictions.jsonl --out artifacts/g3_cases
```

### Windows CMD

For a new, unfrozen working directory, set the path to the original CSV:

```cmd
set "STUDY_RAW_CSV=C:\path\to\original.csv"
```

Then use the existing Python environment:

```cmd
.venv\Scripts\python.exe -m hajj_study.audit_data --csv "%STUDY_RAW_CSV%"
.venv\Scripts\python.exe -m hajj_study.split_data --csv "%STUDY_RAW_CSV%" --out artifacts/row_splits.csv
.venv\Scripts\python.exe -m hajj_study.train_model --csv "%STUDY_RAW_CSV%" --splits artifacts/row_splits.csv --out artifacts/model
.venv\Scripts\python.exe -m hajj_study.cases --csv "%STUDY_RAW_CSV%" --splits artifacts/row_splits.csv --predictions artifacts/model/predictions.jsonl --out artifacts/g3_cases
```

These commands prepare data and model artifacts for a new experiment. They are not needed to inspect the already recorded M6 results.

The Windows data-preparation commands are provided for a new experiment and have not been independently tested in the reviewed frozen environment. Only the offline test command was verified locally on Windows.

Do not run these commands in the original frozen experiment directory.

### Live Experiment Commands

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

On Windows CMD, use `.venv\Scripts\python.exe` instead of `python` and `%STUDY_RAW_CSV%` instead of `$STUDY_RAW_CSV`.

**Important:** These commands describe the experimental workflow. Do not execute them against the frozen study directory. Live commands may create files, overwrite outputs, or make new Gemini API requests.

Frozen experiments must not be changed or overwritten. Raw data, gold labels, model artifacts, and traces remain private.

## Results

The following results are from the original recorded experiment.

| Metric | Single-Agent | Multi-Agent |
|---|---:|---:|
| Correct decisions | 37/90 | 31/90 |
| Coverage | 76.67% ± 3.33 pp | 82.22% ± 3.85 pp |
| Selective error | 46.44% ± 3.30 pp | 58.12% ± 0.37 pp |
| API attempts/case | 1.411 ± 0.038 | 5.156 ± 0.506 |
| Recorded execution time | 1.870 ± 0.280 s | 12.853 ± 0.263 s |

Percentages and per-case measures use mean ± sample SD across three seeds; correct decisions are totals across 30 Test cases × 3 seeds = 90 primary evaluations per condition. The additional 27 clean-reference runs per condition are separate from this primary denominator. The timing excludes deliberate pacing delays.

**Measurement definitions:** Coverage = issued decisions / all primary Test evaluations; selective error = incorrect issued decisions / issued decisions; correct decisions = correct issued outcomes / all 90 primary evaluations. `API attempts/case` includes provider retries and is distinct from logical agent-role calls per case. The private M6 scoring artifact records both measures; logical-call values are not reproduced here without independently checking that artifact.

**Timing qualification:** Recorded execution time measures the observed workflow rather than pure model service latency. The Multi-Agent path persists per-call checkpoints within the measured interval, so orchestration and file-I/O overhead are not matched exactly across conditions. Provider service time and full recorded execution time are separate metrics in the private scoring artifact.

### Recorded Token Cost per Task

Token counts below are from the original saved private M6 score artifact, `artifacts/g7/m6_scores.json`. They cover **90 primary Test evaluations per condition** (30 cases × three seeds), **excluding** the separately recorded clean-reference evaluation overhead. Only aggregate figures are shown; no private traces or gold labels are published.

| Token-cost metric | Single-Agent | Multi-Agent |
|---|---:|---:|
| Recorded total tokens, primary Test runs | 80,700 | 212,036 |
| Recorded tokens per task (total / 90) | 896.67 | ≥ 2,355.96 |
| API attempts missing token usage | 0 | 42 |
| Complete provider token usage? | Yes, in saved records | No |

**Interpretation:** The Multi-Agent value is a **lower bound on recorded token consumption**, not a complete per-task cost: 42 API attempts lack usable token-usage metadata. Thus the observed token totals cannot establish the exact cost ratio or monetary cost of the two conditions. The Single-Agent figure is complete with respect to the recorded provider usage fields. Do not treat the Multi-Agent lower bound as an exact total or infer prices without an applicable provider rate card. Clean-reference calls are evaluation overhead and are not included in this per-task table.

A complete monetary cost was unavailable because 42 Multi-Agent attempts had incomplete token usage metadata.

### Results Interpretation

The Single-Agent system achieved more correct decisions on the 90 test evaluations: 37 compared with 31 for the Multi-Agent system.

The Multi-Agent system achieved higher coverage, but also higher selective error, more API attempts, and longer recorded execution time.

Higher coverage does not necessarily mean better decision quality. The results do not demonstrate that the Multi-Agent architecture outperformed the Single-Agent baseline overall.

### Additional Operational Metrics

The following values were verified against the original private `artifacts/g7/m6_scores.json` supplied separately for audit. Only aggregate metrics are published here; raw private cases, labels, traces, and score artifacts are not committed to the repository. Percentage values are mean ± sample SD across seeds 11, 42, and 77, in percentage points (pp); per-case call values are mean ± sample SD.

| Metric | Single-Agent | Multi-Agent |
|---|---:|---:|
| Unsupported escalation | 46.44% ± 3.30 pp | 58.12% ± 0.37 pp |
| Under-escalation | 0.00% ± 0.00 pp | 0.00% ± 0.00 pp |
| Disrupted-case detection | N/A | 0.00% ± 0.00 pp |
| Clean-decision recovery | N/A | 0.00% ± 0.00 pp |
| Trajectory safe completion | 64.44% ± 1.92 pp | 51.11% ± 3.85 pp |
| Logical agent calls/case | 1.411 ± 0.038 | 4.844 ± 0.204 |

**Metric definitions and interpretation:**

- **Unsupported escalation:** issued risk levels above the programmatic proxy reference level / all issued decisions.
- **Under-escalation:** issued risk levels below the programmatic proxy reference level / all issued decisions.
- **Disrupted-case detection:** among pairs where the initial corrupted-run risk differs from the first clean-reference risk (`clean_first`), the proportion whose initial report correctly flags an altered field **and its expected corruption type**.
- **Clean-decision recovery:** among those same raw-decision-changed pairs, the proportion whose final **issued** risk equals the first clean-reference risk (`clean_first`); abstention is not counted as recovery.
- **Trajectory safe completion:** correct issued decisions plus abstentions / all primary Test evaluations; failed runs are not safe abstentions.
- **Logical agent calls/case:** measured logical role calls, distinct from provider API attempts (which include retries).

**N/A is not 0%.** For Single-Agent, the two corruption-conditional metrics have zero qualifying raw-decision-changed pairs in each seed, so their denominators are zero and the score artifact records `null` (no valid seeds). For Multi-Agent, those denominators are nonzero and both metrics are 0%. These observations should not be interpreted as proof that the Single-Agent system detects or recovers from corruption better.

The numerical aggregates reproduce the saved scoring artifact; independent verification of the original programmatic proxy labels still requires the missing original G3 `cases.jsonl`, `gold_labels.jsonl`, and `case_manifest.json` files. Do not substitute reconstructed or synthetic labels.

The three seeds are repeated evaluations of the same cases, not three independent datasets. The reported variability is descriptive and does not establish statistical significance.

## M4–M6 Evidence

- **M4 (C4.1–C4.3):** 10/10 live Gemini smoke calls, 40/40 Validation evaluations, a working Single-Agent baseline, and successful clean-clone offline tests.
- **M5 (C5.1–C5.4):** 30 golden test cases; 90 primary and 27 clean-reference Single-Agent runs (117/117 completed); automated scoring, three-seed variance, and trajectory-safe completion (64.44%).
- **M6 (C6.1–C6.4):** Three specialists and one coordinator, matched cases and seeds, 117 recorded treatment run keys (116 completed; one `INVALID_REPORT` failure), variance, cost/latency measures, and trace-based failure analysis. The failed run remains in the Test denominator and is neither a successful decision nor a safe abstention.

All incorrect issued decisions were over-escalations relative to reconstructed reference labels. Verification did not correct the HIGH risk prediction in case HJ-T028.

### Failure Analysis by Condition

The following descriptive counts and examples come from the original saved M6 paired scoring evidence. They compare issued risk levels with **programmatic proxy reference labels**, not independently validated real-world Hajj risk outcomes. The 90 primary evaluations in each condition are the same 30 Test cases repeated over seeds 11, 42, and 77.

| Outcome (primary Test evaluations) | Single-Agent | Multi-Agent |
|---|---:|---:|
| Correct issued decisions | 37 | 31 |
| Incorrect issued decisions | 32 | 43 |
| Abstentions | 21 | 15 |
| Failed runs | 0 | 1 |
| Total evaluations | 90 | 90 |

All incorrect **issued** decisions in the recorded scores were **over-escalations** relative to the proxy labels; this is an observed error pattern, not proof of the internal reason for any particular decision. Abstentions and the one failed run are distinct outcomes, not incorrect issued decisions.

Examples from the saved paired comparisons:

| Test case / seed | Proxy reference | Single-Agent | Multi-Agent | Observed contrast |
|---|---|---|---|---|
| `HJ-T028` / 11, 42, 77 | MEDIUM | MEDIUM (correct) | HIGH (incorrect) | Repeated Multi-Agent over-escalation; verification did not correct its HIGH output |
| `HJ-T004` / 11, 42, 77 | Below HIGH | ABSTAIN | HIGH (incorrect) | Multi-Agent issued an over-escalated decision where Single-Agent abstained |
| `HJ-T001` / 42 | — | ABSTAIN | FAILED | Coordinator structured-report validation failure; not a safe abstention |
| `HJ-T005` / 77 | LOW | LOW (correct) | MEDIUM (incorrect) | Multi-Agent over-escalation on a clean case |
| `HJ-T001` / 77 | MEDIUM | HIGH (incorrect) | ABSTAIN | Single-Agent over-escalation while Multi-Agent abstained |

**Limits of causal interpretation:** The traces support comparing final outcomes, observed escalation patterns, verification behavior, and the recorded coordinator failure category. They do **not** establish a specific internal cause for every erroneous risk judgment. In the `INVALID_REPORT` case, the rejected coordinator response and exact validation-rule violation were not retained. The original G3 gold-label files remain necessary for independent verification of the proxy references; no missing cases or labels were reconstructed.

### Recorded Multi-Agent Failure

One Multi-Agent test run, `HJ-T001` with seed 42, ended with `INVALID_REPORT`.

The coordinator's response failed structured-report validation. The recorded attempts include repeated invalid responses and one network error.

The checkpoint preserves failure categories, attempt metadata, and response hashes, but not the rejected coordinator response or the detailed validation error.

Therefore, the exact violated validation rule cannot be determined from the retained evidence.

This failure remains part of the original experiment and is not counted as a successful decision.

## Reproducibility and Private Artifacts

The repository contains the implementation, automated tests, experimental configuration logic, and a summary of the recorded results.

Private experiment artifacts are excluded from Git using `.gitignore`. These include the `artifacts/` directory, frozen case files, gold labels, model artifacts, checkpoints, and detailed execution traces.

The original local M6 artifacts include:

- `artifacts/g7/run_manifest.json`
- `artifacts/g7/multi_runs.jsonl`
- `artifacts/g7/m6_scores.json`
- `artifacts/g7/checkpoints/`
- `artifacts/g7/validation_smoke.json`

These paths describe locally retained evidence. Their presence is not guaranteed in a fresh GitHub clone.

### Inspecting Existing Results

When the original private artifacts are available locally, the saved M6 score summary can be inspected without calling Gemini.

On Windows CMD:

```cmd
.venv\Scripts\python.exe -m json.tool artifacts\g7\m6_scores.json
```

On Linux / macOS:

```bash
python -m json.tool artifacts/g7/m6_scores.json
```

These commands read the existing JSON file and display its contents. They do not rerun the experiment or recalculate the scores.

### Recalculating Scores vs. Rerunning Gemini

**Inspecting saved results** means reading the original score file. It does not require API calls.

**Recalculating scores** means applying the scoring implementation to the original frozen execution traces and reference labels. This requires the original private artifacts and their matching frozen manifests. It must not use recreated, approximate, or synthetic replacements for missing originals.

**Rerunning the experiment** means making new Gemini API requests. Hosted-model outputs may differ across executions, even when the same seed is supplied. A new run is not identical to the original frozen evidence.

The public repository alone is insufficient to independently reproduce the exact original scoring results without the required private artifacts.

Do not upload private traces, raw model responses, gold labels, API keys, or frozen checkpoints to the public repository.

## Historical Contract Status

`CONTRACT.md` records the original G0–G4 implementation contract and explicitly predates the G5 freeze and later M4–M6 execution. Its historical pending-status wording should not be interpreted as the current execution status. The original frozen G5 manifest and private execution evidence, rather than a retrospective edit to that historical contract, are the sources for the recorded experimental configuration.

## Limitations

- The study uses simulated Hajj crowd-management data and reconstructed proxy reference labels, not verified real-world safety outcomes.
- Both architectures were evaluated using matched cases and seeds, but their observed coverage differed.
- The common-issued comparison is a sensitivity analysis, not a fixed-coverage experiment.
- Repeated seeds on the same cases do not represent independent datasets.
- The Multi-Agent condition had one recorded failed test run.
- Incomplete token usage metadata prevents a complete monetary cost comparison.
- Recorded execution-time measurements include architecture-specific orchestration and checkpoint-persistence overhead; they are not a pure provider-latency benchmark.
- No statistical significance claim is made from the three-seed descriptive results.

The results do not establish real-world safety performance.

The original G3 golden cases and gold-label files were not available in the reviewed local repository or evidence archive. Therefore, independent verification of the original reference labels remains pending access to the original private files. No replacement or synthetic files were used during this review.