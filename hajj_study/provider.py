"""G4 Gemini generateContent adapter with mockable transport and auditable token usage.

This module does not access the network during tests. Run the G5 live smoke
before declaring any model ID, temperature, seed or thinking setting frozen.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from urllib import error, request
from urllib.parse import quote

from hajj_study.schemas import ContractError


_ALLOWED_VALIDATION_CODES = frozenset({
    'VALIDATION_FAILED',
    'FINAL_REPORT_SCHEMA_MISMATCH',
    'INVALID_RISK_LEVEL',
    'INVALID_FLAGS_COUNT',
    'INVALID_FLAG_STRUCTURE',
    'INVALID_FLAG_VALUE',
    'INVALID_FLAG_REASON',
    'DUPLICATE_FLAG',
    'INVALID_EVIDENCE_COUNT',
    'INVALID_EVIDENCE_TEXT',
    'FORBIDDEN_AGENT_TERM',
})


def _validation_error_code(exc):
    if isinstance(exc, json.JSONDecodeError):
        return "INVALID_JSON"
    if isinstance(exc, ContractError) and exc.code in _ALLOWED_VALIDATION_CODES:
        return exc.code
    return "VALIDATION_FAILED"


class ProviderError(RuntimeError):
    def __init__(self, reason: str, *, category='PROVIDER_ERROR', retryable=False,
                 usage=None, finish_reason=None, status_code=None):
        super().__init__(reason)
        self.category = category
        self.retryable = retryable
        self.usage = usage or {}
        self.finish_reason = finish_reason
        self.status_code = status_code


class CallFailure(ProviderError):
    def __init__(self, original: ProviderError, attempts: list):
        super().__init__(str(original), category=original.category, retryable=original.retryable,
                         usage=original.usage, finish_reason=original.finish_reason,
                         status_code=original.status_code)
        self.attempts = attempts


@dataclass(frozen=True)
class ProviderResult:
    text: str
    finish_reason: str
    model_version: str | None
    usage: dict
    elapsed_ms: float
    response_sha256: str


@dataclass(frozen=True)
class ValidatedCall:
    report: dict
    attempts: list[dict]
    model_version: str | None
    output_text: str
    total_elapsed_ms: float


class GeminiProvider:
    """One shared provider and retry policy for single/treatment conditions."""
    def __init__(self, *, model_id: str, api_key: str | None = None,
                 temperature: float | None = None, seed: int | None = None,
                 thinking_budget: int | None = None, max_output_tokens: int = 4096,
                 timeout_s: float = 120.0, base_url='https://generativelanguage.googleapis.com/v1beta',
                 transport=None, sleeper=None):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError('Explicit model_id required before smoke')
        if temperature is not None and (isinstance(temperature, bool) or
            not isinstance(temperature, (float, int)) or not 0 <= temperature <= 2):
            raise ValueError('Temperature must be omitted or in [0,2]')
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647):
            raise ValueError('seed out of API range')
        if thinking_budget is not None and (not isinstance(thinking_budget, int) or isinstance(thinking_budget, bool)):
            raise ValueError('thinking_budget must be a supported integer; verify it during smoke')
        if not isinstance(max_output_tokens, int) or max_output_tokens < 256:
            raise ValueError('max_output_tokens too small for structured reports')
        if timeout_s <= 0:
            raise ValueError('timeout_s must be positive')
        self.model_id = model_id
        self.api_key = api_key if api_key is not None else os.getenv('GEMINI_API_KEY', '')
        self.temperature = float(temperature) if temperature is not None else None
        self.seed = seed
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.base_url = base_url.rstrip('/')
        self.transport = transport or request.urlopen
        self.sleeper = sleeper or time.sleep

    def request_data(self, system_prompt: str, user_prompt: str, response_schema: dict) -> dict:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError('Missing system prompt')
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError('Missing user prompt')
        config = {
            'responseMimeType': 'application/json',
            'responseSchema': response_schema,
            'maxOutputTokens': self.max_output_tokens,
        }
        if self.temperature is not None:
            config['temperature'] = self.temperature
        if self.seed is not None:
            config['seed'] = self.seed
        if self.thinking_budget is not None:
            config['thinkingConfig'] = {'thinkingBudget': self.thinking_budget}
        return {
            'systemInstruction': {'parts': [{'text': system_prompt}]},
            'contents': [{'role': 'user', 'parts': [{'text': user_prompt}]}],
            'generationConfig': config,
        }

    def generate_once(self, system_prompt: str, user_prompt: str, response_schema: dict) -> ProviderResult:
        if not self.api_key:
            raise ProviderError('GEMINI_API_KEY is not configured', category='NO_API_KEY')
        payload = self.request_data(system_prompt, user_prompt, response_schema)
        payload_bytes = json.dumps(payload, separators=(',', ':'), sort_keys=True).encode('utf-8')
        endpoint = self.base_url + '/models/' + quote(self.model_id, safe='-_.') + ':generateContent'
        req = request.Request(endpoint, data=payload_bytes, method='POST',
                              headers={'Content-Type': 'application/json', 'x-goog-api-key': self.api_key})
        started = time.perf_counter()
        try:
            with self.transport(req, timeout=self.timeout_s) as response:
                obj = json.loads(response.read().decode('utf-8'))
        except error.HTTPError as exc:
            # Do not emit response bodies or credentials into saved traces.
            raise ProviderError(f'HTTP {exc.code}', category='HTTP',
                                retryable=(exc.code == 429 or 500 <= exc.code <= 599),
                                status_code=exc.code) from exc
        except (error.URLError, OSError, TimeoutError) as exc:
            raise ProviderError(type(exc).__name__, category='NETWORK', retryable=True) from exc
        except (ValueError, UnicodeDecodeError) as exc:
            raise ProviderError('Malformed Gemini transport JSON', category='INVALID_PROVIDER_JSON') from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        usage = obj.get('usageMetadata') if isinstance(obj, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        token_use = {
            'prompt_tokens': usage.get('promptTokenCount'),
            'candidates_tokens': usage.get('candidatesTokenCount'),
            'thoughts_tokens': usage.get('thoughtsTokenCount'),
            'total_tokens': usage.get('totalTokenCount'),
        }
        try:
            candidate = obj['candidates'][0]
            reason = candidate.get('finishReason')
            parts = candidate.get('content', {}).get('parts', [])
            if not isinstance(parts, list):
                raise (TypeError)('parts is not a list')
            visible = ''.join(part.get('text', '') for part in parts
                              if isinstance(part, dict) and not part.get('thought', False))
            model_version = obj.get('modelVersion')
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError('No valid Gemini candidate', category='MISSING_CANDIDATE', usage=token_use) from exc
        if reason != 'STOP':
            raise ProviderError(f'Gemini candidate finished with {reason!r}',
                                category='MAX_TOKENS' if reason == 'MAX_TOKENS' else 'NON_STOP',
                                usage=token_use, finish_reason=reason)
        if not visible.strip():
            raise ProviderError('Empty visible Gemini response', category='EMPTY_TEXT',
                                usage=token_use, finish_reason=reason)
        return ProviderResult(text=visible, finish_reason=reason, model_version=model_version,
                              usage=token_use, elapsed_ms=elapsed_ms,
                              response_sha256=hashlib.sha256(visible.encode('utf-8')).hexdigest())

    def generate_validated(self, *, system_prompt: str, user_prompt: str,
                           response_schema: dict, validator) -> ValidatedCall:
        """Bounded attempt policy: one retry for schema JSON and 429/5xx/network.

        A retry is an API attempt, not another verification operation. Failures
        must be surfaced as FAILED by the outer experiment runner.
        """
        attempts = []
        call_started = time.perf_counter()
        for n in (1, 2):
            try:
                result = self.generate_once(system_prompt, user_prompt, response_schema)
                try:
                    report = validator(json.loads(result.text))
                except (ValueError, TypeError, KeyError) as exc:
                    attempts.append({'attempt': n, 'status': 'INVALID_REPORT',
                                     'finish_reason': result.finish_reason,
                                     'model_version': result.model_version,
                                     'usage': result.usage, 'elapsed_ms': result.elapsed_ms,
                                     'response_sha256': result.response_sha256,
                                     'validation_error_code': _validation_error_code(exc)})
                    failure = ProviderError('Malformed/invalid structured report',
                                            category='INVALID_REPORT', retryable=True,
                                            usage=result.usage, finish_reason=result.finish_reason)
                    if n == 2:
                        raise CallFailure(failure, attempts) from exc
                    continue
                attempts.append({'attempt': n, 'status': 'STOP',
                                 'finish_reason': result.finish_reason,
                                 'model_version': result.model_version, 'usage': result.usage,
                                 'elapsed_ms': result.elapsed_ms, 'response_sha256': result.response_sha256})
                return ValidatedCall(report=report, attempts=attempts,
                                     model_version=result.model_version, output_text=result.text,
                                     total_elapsed_ms=(time.perf_counter()-call_started)*1000)
            except CallFailure:
                raise
            except ProviderError as exc:
                attempts.append({'attempt': n, 'status': exc.category,
                                 'finish_reason': exc.finish_reason, 'usage': exc.usage,
                                 'http_status': exc.status_code, 'elapsed_ms': None})
                if not exc.retryable or n == 2:
                    raise CallFailure(exc, attempts) from exc
                self.sleeper(1.0)  # recorded as elapsed total, excluded from per-call service timing
        raise AssertionError('unreachable')
