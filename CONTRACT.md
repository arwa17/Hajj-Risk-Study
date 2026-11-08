# CPCS 683 — Controlled Hajj Risk Study Contract

**Status:** G0–G4 implementation contract. G5 live model settings and coverage threshold are **NOT frozen**. This document records implementation decisions, not proof that M4/M5/M6 have been completed.

## Source hierarchy
1. **Official:** `Project_Description_and_Milestones`, `Project grading rubric` — C4.1–C6.4.
2. **Approved research:** M2 Proposal pp. 2–7, M3 Related Work, Evaluation Plan — question, role decomposition, fixed comparison, 30 cases and seeds.
3. **Reference paper:** Samrgandi (2026), p. 426 for seven weights, thresholds, 17 predictors and p. 428 for reported model performance. Original raw-value risk encoding was not published; do **not** claim exact replication.
4. **Implementation choices:** This contract and `hajj_study/` modules, with passing acceptance tests. An implementation choice is not an official rubric requirement.

## Research comparison
- Only manipulated design factor: **agent-role and message organization**.
- Baseline: Gemini Generalist (one agent). Treatment: Crowd, Health, Environment specialists and one Gemini Coordinator (expert-team fan-out/fan-in).
- Identical source rows, shared ML predictions, gate and verification budget, frozen Gemini configuration, seeds `11,42,77`, metrics and test cases.
- Never use the prior hand-authored 36-case M5/M6 results as results for the proposal-aligned study.

## Ground truth and data protection
- Source: original synthetic Kaggle CSV, 10,000×30. SHA-256 `7c289edf5b3390b479c48ac5a493db566f6c46d4d587ae101b03d885bf948e8a`.
- Proxy from **clean** records: density `.25`, temperature `.20`, fatigue `.15`, stress `.15`, queue `.10`, emergency `.10`, distance `.05`; LOW `<1.8`, MEDIUM `1.8–2.4`, HIGH `>2.4`.
- Chosen reconstruction: categorical Low/Medium/High→1/2/3, numeric temperature & queue Min–Max to `[1,3]`, distance inverse scale `[3,1]`, emergency Yes→3/No→0. Expected class counts LOW=3337, MEDIUM=5755, HIGH=908; **not identical** to published counts.
- Source indices and dirty/clean records: 7,000 train / 1,000 validation / 2,000 test, stratified seed 42. ML trains on the 17 eligible predictors (explicit allowlist), never on seven label fields; preprocessing and SMOTENC fit only on train.
- Strictly private: original CSV, reconstructed source-row labels, clean references, corruption metadata, trained model, raw traces and API credentials pending dataset-license verification. Do not commit private gold or data to a public repo.

## Immutable case-suite decisions (G3)
- Test: 30 cases = 18 typical clean, 3 missing, 3 noisy, 3 conflicting, 3 borderline clean; 10 each class; 9 clean counterpart runs per seed/condition.
- Validation: 40 cases = 24 typical clean + 4 each missing/noisy/conflicting/borderline. Separate records and selection seeds.
- Three test corruptions and four validation corruptions span two domains. Deliberate stress-conditioned sampling is **not** an estimate of population incident prevalence.
- Conflicting readings alternate correct position in balanced, deterministic order; numeric types are identical. Agent input does not expose which is correct.
- Case IDs shuffled to avoid revealing category by sequence. All suite files are SHA-256 recorded privately.

## Information firewall (G4)
| Component | Allowed information | Not allowed |
|---|---|---|
| Generalist | 7 observed fields (clean/corrupted), identical frozen shared-ML prediction | proxy formula/weights/thresholds, reference label, hidden clean values, 17 raw ML features, metadata |
| Crowd specialist | density, queue, distance | health/environment evidence, ML-global prediction, labels |
| Health specialist | fatigue, stress, emergency | crowd/environment evidence, ML-global prediction, labels |
| Environment specialist | temperature | all other raw evidence, ML-global prediction, labels |
| Coordinator | 3 validated specialist reports, same global ML prediction as Generalist | raw 17 ML inputs, hidden reference/clean information |
| Shared gate | final report, its reported flags, integer `k`, verification state | scorer metadata, reference labels and risk-score formula |
| Scorer | private clean records, labels and actual traces | cannot alter inference or overwrite risk decisions |

- Initial corrupted observations have **no trusted corruption-quality flags**. Agents must infer flags from what they see. Missing is `null`, conflict is two unlabeled-as-clean readings, suspected noise is a silently modified value.
- Generalist and Coordinator final JSON: `case_id`, `proposed_risk_level`, `evidence_flags[{field,type,reason}]`, `supporting_evidence`. Specialists additionally report domain and `domain_risk_signal` and may flag only their own fields.
- Flag vocabulary: `missing`, `conflicting`, `suspected_noisy`. Severity priority: conflicting=3, missing=2, noisy=1. Tie priority: crowd > health > environment. False flags are **allowed and measured**; they are not auto-corrected from hidden metadata.
- Qualitative field ranges and LOW/MEDIUM/HIGH semantics are shared fairly. No proxy weights, thresholds or formula in prompts.

## Shared deterministic gate (G4)
1. Validate final JSON and `case_id`. A malformed provider response after the **same bounded retry policy** is FAILED, never ABSTAIN.
2. `sufficiency = 7 − number_of_distinct_flagged_fields`; this is only an operational completeness measure, not probability of correctness.
3. If `sufficiency >= k`, issue the **agent's** proposed risk level without re-scoring it.
4. Otherwise, choose the domain containing the strongest unresolved flag; tie crowd → health → environment; verify **one** complete domain maximum.
5. Replace that domain's observed values with clean values only; preserve observations in the other domains. Single: rerun Generalist. Multi: rerun selected specialist and Coordinator, preserve other reports.
6. Flags in unverified domains are inherited even if a rerun omits them. If post-verification sufficiency still `< k`, ABSTAIN; if malformed/failed, FAILED.
7. The gate never recomputes or substitutes the reference label. The verification oracle must not release any other domain or the gold label.

## G5–G8 pending conditions
- Candidate model ID per plan: `gemini-3.5-flash-lite`. This is **not a verified/frozen runtime setting** until a ten-call live smoke confirms request acceptance, schema, `finishReason=STOP`, thinking-token usage and model version. No API key has been supplied or tested here.
- Do not hardcode successful `temperature`, `seed`, `thinkingBudget` or `maxOutputTokens` support until G5 verifies them for the account. The mock provider can test recording and error paths, not provider compatibility.
- Calibrate common integer threshold `k∈[0,7]` using single-agent validation only; freeze it and the complete manifest before any held-out test calls. Target 80% coverage is a target, **not an observed result or guarantee of equal coverage**.
- M5: 30 cases × seeds 11,42,77 = 90 baseline core evaluations, plus 27 clean-reference evaluations. M6: same grid for treatment. Failure analysis must use actual traces; report coverage for both systems, shared-issued subset, variance, latency and attempted API calls.
- All mock testing and locally generated references are readiness artifacts, not M4 live-demo evidence or M5/M6 findings.
