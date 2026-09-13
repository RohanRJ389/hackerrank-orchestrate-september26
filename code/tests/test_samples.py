"""Regression scoreboard against the 25 solved samples.

These tests are a ratchet, not a pass/fail accuracy gate. The state builder
behind them is a stand-in for the real normalizer and ignores messages and
images, so exact agreement is not expected yet. What they catch is a change
that quietly makes the engine worse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.scoreboard import (
    SCORED_FIELDS,
    exact_matches,
    field_totals,
    sample_ids,
    score_all,
)

BASELINE_PATH = Path(__file__).parent / "baselines" / "sample_scoreboard.json"


@pytest.fixture(scope="module")
def scores() -> dict:
    return score_all()


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text())


def test_every_sample_produces_a_decision(scores):
    assert sorted(scores) == sample_ids()
    assert len(scores) == 25


@pytest.mark.parametrize("field", SCORED_FIELDS)
def test_no_field_regresses(scores, baseline, field):
    actual = field_totals(scores)[field]
    expected = baseline["field_totals"][field]
    assert actual >= expected, (
        f"{field} matched {actual}/25, down from the {expected}/25 baseline. "
        f"If this drop is intended, refresh tests/baselines/sample_scoreboard.json."
    )


def test_exact_matches_do_not_regress(scores, baseline):
    actual = exact_matches(scores)
    assert actual >= baseline["exact_matches"]


def test_no_individual_sample_regresses(scores, baseline):
    worsened = [
        (request_id, field)
        for request_id, result in scores.items()
        for field, matched in result.items()
        if baseline["per_request"][request_id][field] and not matched
    ]
    assert not worsened, f"previously matching fields now fail: {worsened}"
