"""Scores engine output against the solved samples, field by field."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from contracts import DecisionInput
from engine import decide

from .state_builder import Dataset, StateBuilder

# Free text is not comparable field-for-field; it is checked by its own tests.
SCORED_FIELDS = (
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
)

AMOUNT_TOLERANCE = Decimal("0.05")


def _matches(field: str, actual: str, expected: str) -> bool:
    if field == "amount_safe_to_pay":
        try:
            return abs(Decimal(actual) - Decimal(expected)) <= AMOUNT_TOLERANCE
        except InvalidOperation:
            return actual == expected
    return actual.strip() == expected.strip()


def sample_ids() -> list[str]:
    return sorted(Dataset.load().samples)


def score_sample(request_id: str) -> dict[str, bool]:
    data = Dataset.load()
    decision_input: DecisionInput = StateBuilder(request_id, data).build()
    actual = decide(decision_input).to_row()
    expected = data.samples[request_id]
    return {
        field: _matches(field, actual[field], expected[field])
        for field in SCORED_FIELDS
    }


def score_all() -> dict[str, dict[str, bool]]:
    return {request_id: score_sample(request_id) for request_id in sample_ids()}


def field_totals(scores: dict[str, dict[str, bool]]) -> dict[str, int]:
    return {
        field: sum(1 for result in scores.values() if result[field])
        for field in SCORED_FIELDS
    }


def exact_matches(scores: dict[str, dict[str, bool]]) -> int:
    return sum(1 for result in scores.values() if all(result.values()))
