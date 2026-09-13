"""Opt-in paid integration tests for Claude Agent SDK.

Run with:
    RUN_LIVE_NORMALIZER_TESTS=1 pytest tests/test_normalizer_live.py -q
"""

from __future__ import annotations

import os

import pytest

from contracts import assert_valid
from normalizer.normalize import normalize_async


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_NORMALIZER_TESTS") != "1",
    reason="paid Claude integration tests are opt-in",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_id",
    [
        "request_01",  # deterministic review/sign
        "request_03",  # message plus image
    ],
)
async def test_live_normalization_produces_signed_valid_state(request_id):
    decision_input = await normalize_async(
        request_id,
        use_cache=False,
        keep_workspace=False,
    )
    assert decision_input.financial_state.attestation.model_provider == "anthropic"
    assert "approved_unsigned_state_sha256=" in (
        decision_input.financial_state.attestation.notes or ""
    )
    assert_valid(decision_input)
