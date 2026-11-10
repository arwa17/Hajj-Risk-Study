# M6.2 Coordinator Prompt Comparison

## Objective
Compare the Original and Improved Coordinator system prompts while keeping the remaining experimental conditions identical.

## Experimental Setup
- Case: HJ-V001
- Seed: 42
- Model: gemini-3.5-flash-lite
- Temperature: 0.0
- Thinking budget: None (omitted from API request)
- Maximum output tokens: 4096
- Configuration source: artifacts/g5/manifest_frozen.json
- Inputs: three previously validated specialist reports from the G7 validation-only checkpoint.

## Controlled Comparison
- Original prompt: original_prompts.py, verified against Git commit dceb4cb.
- Improved prompt: hajj_study/prompts.py.
- Both groups use the same case, specialist reports, response schema, provider, generation settings, and validation logic.
- The intended difference is the Coordinator system instruction only.
- The pilot tests the Coordinator initial call, not the complete end-to-end multi-agent workflow.

## Local Verification
- Pilot data loading: PASS.
- Prompt comparison: PASS; only systemInstruction differs.
- Original and Improved mocked responses: PASS.
- Retry, failure, and missing-usage tests: PASS.
- Regression tests: 5 passed and 8 subtests passed.
- Mock output: artifacts/m62_coordinator_pilot/mock/pilot_results.json.

## Evaluation Measures
- First-attempt validity and final validity after retries.
- Number of attempts and failure category.
- Reported token usage, including completeness of usage reporting.
- Latency when available.
- Semantic quality requires separate evaluation criteria and evidence.

## Current Limitations
- All M6.2 pilot results so far are MOCK, not Gemini Live results.
- Simulated token counts are not actual API costs.
- Passing mock tests does not establish that the Improved prompt outperforms the Original prompt.
- A single-case pilot cannot establish statistical improvement.
- Hosted-model seeds do not guarantee identical outputs across repeated executions.

## Safety and Reproducibility
- Run local checks: .venv\Scripts\python.exe -m experiments.m62_coordinator_prompt.run_pilot
- No Gemini Live calls are performed by the current runner.
- Live execution requires explicit approval, a configured GEMINI_API_KEY, and cost review.
- Preserve frozen G5, M6, and G7 artifacts without modification.
