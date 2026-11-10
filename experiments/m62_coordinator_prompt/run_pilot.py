import json
import os
import runpy
from pathlib import Path

from hajj_study.corrupt import DOMAINS
from hajj_study.prompts import coordinator_messages
from hajj_study.provider import CallFailure, GeminiProvider
from hajj_study.schemas import (
    final_response_schema,
    validate_case,
    validate_final_report,
    validate_specialist_report,
)
from tests.study.test_provider_mock import (
    MockTests,
    gemini_response,
)


ROOT = Path(__file__).resolve().parents[2]

CASES = ROOT / "artifacts/g3_cases/validation/cases.jsonl"
CHECKPOINTS = ROOT / "artifacts/g7/validation_only/checkpoints"

OUTPUT_DIR = ROOT / "artifacts/m62_coordinator_pilot/mock"

ORIGINAL_PROMPT_PATH = (
    Path(__file__).resolve().parent / "original_prompts.py"
)

TOKEN_KEYS = (
    "prompt_tokens",
    "candidates_tokens",
    "thoughts_tokens",
    "total_tokens",
)


def load_pilot_data():
    files = sorted(CHECKPOINTS.glob("*.json"))

    if len(files) != 1:
        raise RuntimeError(
            "Expected exactly one validation checkpoint"
        )

    checkpoint = json.loads(
        files[0].read_text(encoding="utf-8")
    )

    calls = checkpoint["calls"]

    reports = {
        domain: calls[i]["report"]
        for i, domain in enumerate(DOMAINS)
    }

    cid = reports["crowd"]["case_id"]

    cases = [
        json.loads(line)
        for line in CASES.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    case = validate_case(
        next(c for c in cases if c["case_id"] == cid)
    )

    reports = {
        domain: validate_specialist_report(
            reports[domain], cid, domain
        )
        for domain in DOMAINS
    }

    mock_report = validate_final_report(
        calls[3]["report"], cid
    )

    return case, reports, mock_report


def load_original_messages():
    if not ORIGINAL_PROMPT_PATH.is_file():
        raise FileNotFoundError(
            "Original prompt file is missing: "
            + str(ORIGINAL_PROMPT_PATH)
        )

    namespace = runpy.run_path(
        str(ORIGINAL_PROMPT_PATH)
    )

    return namespace["coordinator_messages"]


def prompt_groups():
    return (
        ("Original", load_original_messages()),
        ("Improved", coordinator_messages),
    )


def load_frozen_generation():
    manifest = json.loads((ROOT / "artifacts/g5/manifest_frozen.json").read_text(encoding="utf-8"))
    config = manifest["generation"]
    assert config["model_id"] == manifest["model_id"]
    assert 42 in config["seed_policy"]
    return config


def compare_prompts(case, reports):
    original_messages = load_original_messages()

    cid = case["case_id"]
    ml = case["ml_prediction"]

    original = original_messages(cid, reports, ml)
    improved = coordinator_messages(cid, reports, ml)

    config = load_frozen_generation()

    # This provider is used only to construct requests.
    # No network call is made.
    provider = GeminiProvider(
        model_id=config["model_id"],
        api_key="testkey",
        temperature=config["temperature"],
        seed=42,
        thinking_budget=config["thinking_budget"],
        max_output_tokens=config["max_output_tokens"],
    )

    schema = final_response_schema()

    request_a = provider.request_data(
        *original, schema
    )

    request_b = provider.request_data(
        *improved, schema
    )

    assert set(request_a) == set(request_b)

    changed = {
        key
        for key in request_a
        if request_a[key] != request_b[key]
    }

    assert changed == {"systemInstruction"}
    assert request_a["contents"] == request_b["contents"]

    assert (
        request_a["generationConfig"]
        == request_b["generationConfig"]
    )

    print("Prompt comparison: PASS")
    print("Changed fields:", ", ".join(sorted(changed)))
    print("User inputs identical: True")
    print("Generation settings identical: True")


def build_result_record(
    *,
    case_id,
    seed,
    group,
    attempts,
    final_valid,
    model_version=None,
    elapsed_ms=None,
    failure_category=None,
):
    """
    Build a consistent result record.

    Missing token usage is not treated as zero.
    """

    usage = {}

    for key in TOKEN_KEYS:
        values = [
            (attempt.get("usage") or {}).get(key)
            for attempt in attempts
        ]

        available = [
            value
            for value in values
            if value is not None
        ]

        usage[key] = (
            sum(available) if available else None
        )

    token_usage_complete = bool(attempts) and all(
        (attempt.get("usage") or {}).get(key)
        is not None
        for attempt in attempts
        for key in TOKEN_KEYS
    )

    return {
        "case_id": case_id,
        "seed": seed,
        "group": group,
        "mode": "MOCK",
        "first_attempt_valid": bool(
            attempts
            and attempts[0]["status"] == "STOP"
        ),
        "final_valid": bool(final_valid),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "usage": usage,
        "token_usage_complete": token_usage_complete,
        "model_version": model_version,
        "elapsed_ms": elapsed_ms,
        "failure_category": failure_category,
    }


def execute_mock_call(
    *,
    case_id,
    seed,
    group,
    messages,
    responses,
):
    system, user = messages

    provider = MockTests().provider(responses)

    try:
        result = provider.generate_validated(
            system_prompt=system,
            user_prompt=user,
            response_schema=final_response_schema(),
            validator=lambda response: validate_final_report(
                response, case_id
            ),
        )

        return build_result_record(
            case_id=case_id,
            seed=seed,
            group=group,
            attempts=result.attempts,
            final_valid=True,
            model_version=result.model_version,
            elapsed_ms=result.total_elapsed_ms,
        )

    except CallFailure as exc:
        return build_result_record(
            case_id=case_id,
            seed=seed,
            group=group,
            attempts=exc.attempts,
            final_valid=False,
            failure_category=exc.category,
        )


def run_mock_comparison(case, reports, mock_report):
    cid = case["case_id"]

    records = []

    for group, messages_fn in prompt_groups():
        messages = messages_fn(
            cid,
            reports,
            case["ml_prediction"],
        )

        record = execute_mock_call(
            case_id=cid,
            seed=42,
            group=group,
            messages=messages,
            responses=[
                gemini_response(report=mock_report)
            ],
        )

        assert record["final_valid"]
        assert record["first_attempt_valid"]
        assert record["attempt_count"] == 1
        assert record["usage"]["total_tokens"] == 203
        assert record["token_usage_complete"]

        records.append(record)

        print(
            group,
            "| Mock: PASS",
            "| Attempts:",
            record["attempt_count"],
            "| Tokens (simulated):",
            record["usage"]["total_tokens"],
        )

    return records


def test_retry_recording():
    invalid = gemini_response()

    invalid["candidates"][0]["content"]["parts"][1][
        "text"
    ] = "not-json"

    record = execute_mock_call(
        case_id="HJ-V003",
        seed=42,
        group="RetryTest",
        messages=("mock-system", "mock-user"),
        responses=[
            invalid,
            gemini_response(),
        ],
    )

    assert record["first_attempt_valid"] is False
    assert record["final_valid"] is True
    assert record["attempt_count"] == 2

    assert [
        attempt["status"]
        for attempt in record["attempts"]
    ] == [
        "INVALID_REPORT",
        "STOP",
    ]

    assert record["usage"]["total_tokens"] == 406
    assert record["token_usage_complete"] is True

    print("Runner Retry Test: PASS")
    print(
        "First-attempt valid:",
        record["first_attempt_valid"],
    )
    print("Final valid:", record["final_valid"])
    print("Attempts:", record["attempt_count"])
    print(
        "Total simulated tokens:",
        record["usage"]["total_tokens"],
    )


def test_failed_call_recording():
    first_invalid = gemini_response()
    second_invalid = gemini_response()

    first_invalid["candidates"][0]["content"][
        "parts"
    ][1]["text"] = "not-json"

    second_invalid["candidates"][0]["content"][
        "parts"
    ][1]["text"] = "still-not-json"

    record = execute_mock_call(
        case_id="HJ-V003",
        seed=42,
        group="FailureTest",
        messages=("mock-system", "mock-user"),
        responses=[
            first_invalid,
            second_invalid,
        ],
    )

    assert record["first_attempt_valid"] is False
    assert record["final_valid"] is False
    assert record["attempt_count"] == 2

    assert record["failure_category"] == (
        "INVALID_REPORT"
    )

    assert record["usage"]["total_tokens"] == 406
    assert record["token_usage_complete"] is True

    print("Runner Failure Test: PASS")
    print("Final valid:", record["final_valid"])
    print(
        "Failure category:",
        record["failure_category"],
    )
    print(
        "Total simulated tokens:",
        record["usage"]["total_tokens"],
    )


def test_missing_usage_recording():
    attempts = [
        {
            "attempt": 1,
            "status": "NETWORK",
            "usage": {},
        },
        {
            "attempt": 2,
            "status": "STOP",
            "usage": {
                "prompt_tokens": 37,
                "candidates_tokens": 45,
                "thoughts_tokens": 121,
                "total_tokens": 203,
            },
        },
    ]

    record = build_result_record(
        case_id="HJ-V003",
        seed=42,
        group="MissingUsageTest",
        attempts=attempts,
        final_valid=True,
    )

    assert record["attempt_count"] == 2
    assert record["first_attempt_valid"] is False
    assert record["final_valid"] is True
    assert record["token_usage_complete"] is False

    # This is the reported subtotal, not necessarily
    # the complete cost of both attempts.
    assert record["usage"]["total_tokens"] == 203

    print("Missing Usage Test: PASS")
    print(
        "Reported tokens:",
        record["usage"]["total_tokens"],
    )
    print(
        "Usage complete:",
        record["token_usage_complete"],
    )


def save_mock_results(records):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = OUTPUT_DIR / "pilot_results.json"

    output_file.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    saved = json.loads(
        output_file.read_text(encoding="utf-8")
    )

    assert len(saved) == len(records)

    print("Mock results saved:", output_file)
    print("Records:", len(saved))


def main():
    case, reports, mock_report = load_pilot_data()

    print("Pilot data: PASS")
    print("Case ID:", case["case_id"])
    print(
        "Specialist domains:",
        ", ".join(reports),
    )

    compare_prompts(case, reports)

    records = run_mock_comparison(
        case,
        reports,
        mock_report,
    )

    test_retry_recording()
    test_failed_call_recording()
    test_missing_usage_recording()

    save_mock_results(records)


if __name__ == "__main__":
    main()
