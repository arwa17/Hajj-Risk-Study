import unittest
from hajj_study.prompts import coordinator_messages
from tests.study.test_schema_leakage import CID, CASE, specialist

class ValidationImprovementTests(unittest.TestCase):
    def test_coordinator_prompt_states_validator_limits(self):
        reports = {d: specialist(d) for d in ("crowd", "health", "environment")}
        system, _ = coordinator_messages(CID, reports, CASE["ml_prediction"])
        for requirement in ("0-7 evidence flags", "1-400 characters", "1-7 supporting evidence", "1-450 characters", "field and type", "case_id must match"):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, system)

    def test_invalid_json_has_safe_diagnostic(self):
        from tests.study.test_provider_mock import MockTests, gemini_response
        import json
        malformed = gemini_response()
        malformed["candidates"][0]["content"]["parts"][1]["text"] = "PRIVATE_REJECTED_RESPONSE"
        provider = MockTests().provider([malformed, gemini_response()])
        result = provider.generate_validated(system_prompt="sys", user_prompt="user", response_schema={}, validator=lambda d: d)
        self.assertEqual(result.attempts[0]["validation_error_code"], "INVALID_JSON")
        self.assertNotIn("PRIVATE_REJECTED_RESPONSE", json.dumps(result.attempts))
        self.assertEqual([a["status"] for a in result.attempts], ["INVALID_REPORT", "STOP"])

    def test_contract_error_code_is_recorded_safely(self):
        from tests.study.test_provider_mock import MockTests, gemini_response
        from hajj_study.schemas import ContractError
        import json
        provider = MockTests().provider([gemini_response(), gemini_response()])
        def reject(report):
            raise ContractError("PRIVATE_VALIDATION_DETAILS", code="INVALID_RISK_LEVEL")
        with self.assertRaises(Exception) as caught:
            provider.generate_validated(system_prompt="sys", user_prompt="user", response_schema={}, validator=reject)
        attempts = caught.exception.attempts
        self.assertEqual(len(attempts), 2)
        self.assertEqual([a["validation_error_code"] for a in attempts], ["INVALID_RISK_LEVEL"] * 2)
        self.assertNotIn("PRIVATE_VALIDATION_DETAILS", json.dumps(attempts))

    def test_final_report_validation_error_codes(self):
        from hajj_study.schemas import ContractError, validate_final_report
        base = {"case_id": "HJ-V003", "proposed_risk_level": "MEDIUM", "evidence_flags": [], "supporting_evidence": ["Observed evidence"]}
        for field, value, expected in [("proposed_risk_level", "UNKNOWN", "INVALID_RISK_LEVEL"), ("supporting_evidence", [], "INVALID_EVIDENCE_COUNT")]:
            with self.subTest(field=field):
                report = dict(base)
                report[field] = value
                with self.assertRaises(ContractError) as caught:
                    validate_final_report(report, "HJ-V003")
                self.assertEqual(caught.exception.code, expected)

    def test_unknown_validation_code_is_not_logged(self):
        from hajj_study.provider import _validation_error_code
        from hajj_study.schemas import ContractError
        secret = "PRIVATE_UNTRUSTED_DIAGNOSTIC"
        self.assertEqual(
            _validation_error_code(ContractError("Rejected", code=secret)),
            "VALIDATION_FAILED",
        )
        self.assertEqual(
            _validation_error_code(ValueError(secret)),
            "VALIDATION_FAILED",
        )
