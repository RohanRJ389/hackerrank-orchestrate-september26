"""Edge cases and output-contract invariants for the decision engine."""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from contracts import DecisionInput, PaymentMethod
from engine import Timeline, decide, eligible_installment_options
from engine.forecast import format_amount
from engine.output import CSV_COLUMNS, format_safe_amount
from engine.plans import change_sets
from engine.ranking import best_plan, completes_in_time
from harness.scoreboard import sample_ids
from harness.state_builder import StateBuilder

FIXTURES = Path(__file__).parent / "fixtures" / "contracts" / "valid"
GOLDEN = Path(__file__).parent / "fixtures" / "decisions"

STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def load(name: str) -> DecisionInput:
    return DecisionInput.model_validate(json.loads((FIXTURES / f"{name}.json").read_text()))


ALL_FIXTURES = sorted(path.stem for path in FIXTURES.glob("*.json"))


# -- formatting -------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [("25256", "25256"), ("620.4", "620.40"), ("23.5", "23.50"),
     ("15952906.666", "15952906.67"), ("0", "0")],
)
def test_payment_amounts_pad_to_two_decimals(value, expected):
    assert format_amount(Decimal(value)) == expected


@pytest.mark.parametrize(
    "value,expected",
    [("17229139.2", "17229139.2"), ("603.30", "603.3"), ("25256", "25256"),
     ("0", "0"), ("0.00", "0")],
)
def test_capacity_amounts_are_unpadded(value, expected):
    assert format_safe_amount(Decimal(value)) == expected


# -- output contract --------------------------------------------------------

@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_row_satisfies_the_output_contract(name):
    decision_input = load(name)
    row = decide(decision_input).to_row()
    request = decision_input.request

    assert tuple(row) == CSV_COLUMNS
    assert row["affordability_status"] in STATUSES
    assert row["recommended_payment_method"] in METHODS
    assert Decimal(0) <= Decimal(row["amount_safe_to_pay"]) <= request.requested_amount


@pytest.mark.parametrize("request_id", sample_ids())
def test_every_sample_row_satisfies_the_output_contract(request_id):
    decision_input = StateBuilder(request_id).build()
    row = decide(decision_input).to_row()

    assert row["affordability_status"] in STATUSES
    assert row["recommended_payment_method"] in METHODS
    assert Decimal(0) <= Decimal(row["amount_safe_to_pay"]) <= \
        decision_input.request.requested_amount
    assert row["decision_explanation"].strip()
    assert len(row["spending_changes_needed"].split("|")) <= 3 or \
        row["spending_changes_needed"] == "none"


@pytest.mark.parametrize("request_id", sample_ids())
def test_status_and_method_agree(request_id):
    row = decide(StateBuilder(request_id).build()).to_row()
    status, method = row["affordability_status"], row["recommended_payment_method"]

    if status == "affordable_now":
        assert method == "full_payment"
        assert row["earliest_date_for_full_payment"] == \
            row["payment_plan"].split(":")[0]
    if status == "not_affordable":
        assert method == "not_recommended"
        assert row["payment_plan"] == "none"
    if method == "wait":
        assert status == "affordable_later"
    if method == "not_recommended":
        assert row["spending_changes_needed"] == "none"


@pytest.mark.parametrize("request_id", sample_ids())
def test_payment_plan_is_chronological_and_complete(request_id):
    decision_input = StateBuilder(request_id).build()
    row = decide(decision_input).to_row()
    if row["payment_plan"] == "none":
        return

    entries = [entry.split(":") for entry in row["payment_plan"].split("|")]
    days = [date.fromisoformat(day) for day, _ in entries]
    assert days == sorted(days)

    total = sum(Decimal(amount) for _, amount in entries)
    if row["recommended_payment_method"] in {"full_payment", "partial_payment", "wait"}:
        assert total == decision_input.request.requested_amount
    else:
        # Installments legitimately total more, because of the financing fee.
        assert total >= decision_input.request.requested_amount


@pytest.mark.parametrize("request_id", sample_ids())
def test_partial_payment_follows_the_two_payment_rule(request_id):
    decision_input = StateBuilder(request_id).build()
    row = decide(decision_input).to_row()
    if row["recommended_payment_method"] != "partial_payment":
        return

    entries = [entry.split(":") for entry in row["payment_plan"].split("|")]
    assert len(entries) == 2
    assert entries[0][0] == decision_input.request.request_date.isoformat()
    assert Decimal(entries[0][1]) == Decimal(row["amount_safe_to_pay"])
    assert entries[1][0] == row["earliest_date_for_full_payment"]
    assert date.fromisoformat(entries[1][0]) <= decision_input.request.desired_completion_date
    assert decision_input.request.allows_partial_payment


# -- preference and deadline rules -----------------------------------------

@pytest.mark.parametrize("request_id", sample_ids())
def test_recommendation_respects_accepted_methods(request_id):
    decision_input = StateBuilder(request_id).build()
    row = decide(decision_input).to_row()
    accepted = {m.value for m in decision_input.financial_state.preferences.accepted_payment_methods}
    method = row["recommended_payment_method"]
    if method in {"wait", "not_recommended"}:
        return
    assert method in accepted


@pytest.mark.parametrize("request_id", sample_ids())
def test_installments_respect_the_month_cap(request_id):
    decision_input = StateBuilder(request_id).build()
    cap = decision_input.financial_state.preferences.max_installment_months
    for option in eligible_installment_options(decision_input):
        assert cap is not None and option.number_of_payments <= cap


@pytest.mark.parametrize("request_id", sample_ids())
def test_recommended_plan_finishes_by_the_deadline(request_id):
    decision_input = StateBuilder(request_id).build()
    row = decide(decision_input).to_row()
    if row["payment_plan"] == "none":
        return
    last = date.fromisoformat(row["payment_plan"].split("|")[-1].split(":")[0])
    assert last <= decision_input.request.desired_completion_date


@pytest.mark.parametrize("request_id", sample_ids())
def test_spending_changes_target_permitted_series(request_id):
    decision_input = StateBuilder(request_id).build()
    row = decide(decision_input).to_row()
    if row["spending_changes_needed"] == "none":
        return

    allowed = {s.target_event_id: s for s in decision_input.financial_state.adjustable_series}
    protected = decision_input.financial_state.preferences.protected_categories
    for action in row["spending_changes_needed"].split("|"):
        parts = action.split(":")
        series = allowed[parts[1]]
        assert series.category not in protected
        if parts[0] == "reduce_to":
            assert Decimal(parts[2]) == series.minimum_allowed_amount


# -- safety invariant -------------------------------------------------------

@pytest.mark.parametrize("request_id", sample_ids())
def test_recommended_plan_never_breaches_the_minimum(request_id):
    decision_input = StateBuilder(request_id).build()
    decision = decide(decision_input)
    if not decision.payments:
        return
    timeline = Timeline(decision_input.financial_state, decision.spending_changes)
    assert timeline.survives_payments(decision.payments)


@pytest.mark.parametrize("request_id", sample_ids())
def test_suffix_minimum_never_decreases_over_time(request_id):
    state = StateBuilder(request_id).build().financial_state
    timeline = Timeline(state)
    day = state.as_of_date
    previous = timeline.suffix_min(day)
    while day <= state.forecast_end_date:
        current = timeline.suffix_min(day)
        assert current >= previous
        previous = current
        day += timedelta(days=1)


@pytest.mark.parametrize("request_id", sample_ids())
def test_earliest_date_is_the_first_affordable_day(request_id):
    decision_input = StateBuilder(request_id).build()
    state = decision_input.financial_state
    amount = decision_input.request.requested_amount
    timeline = Timeline(state)
    earliest = decide(decision_input).earliest_date_for_full_payment

    if earliest is None:
        assert not timeline.survives_payments([(state.forecast_end_date, amount)])
        return

    assert state.as_of_date <= earliest <= state.forecast_end_date
    assert timeline.survives_payments([(earliest, amount)])
    if earliest > state.as_of_date:
        assert not timeline.survives_payments([(earliest - timedelta(days=1), amount)])


# -- determinism ------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_decisions_are_deterministic(name):
    decision_input = load(name)
    assert decide(decision_input).to_row() == decide(decision_input).to_row()


def test_at_most_three_spending_changes_are_ever_offered():
    for name in ALL_FIXTURES:
        for changes in change_sets(load(name)):
            assert 1 <= len(changes) <= 3
            targets = [change.target_event_id for change in changes]
            assert len(set(targets)) == len(targets)


def test_ranking_rejects_plans_that_miss_the_deadline():
    decision_input = load("installment_plan")
    timeline = Timeline(decision_input.financial_state)
    from engine.plans import candidate_plans
    plans = candidate_plans(decision_input, timeline, Decimal(0), None)
    yesterday = decision_input.request.request_date - timedelta(days=1)
    assert best_plan(plans, yesterday) is None
    assert not any(completes_in_time(plan, yesterday) for plan in plans)


# -- partial payment --------------------------------------------------------
# No solved sample reaches a partial-payment recommendation through the test
# state builder, so the rule is exercised against a fixture adjusted to meet
# all four of its preconditions.

def _partial_ready() -> DecisionInput:
    """`wait_for_confirmed_salary`, but with partial payment made available."""
    decision_input = load("wait_for_confirmed_salary")
    state = decision_input.financial_state
    preferences = state.preferences.model_copy(
        update={"accepted_payment_methods": frozenset(
            state.preferences.accepted_payment_methods | {PaymentMethod.PARTIAL_PAYMENT}
        )}
    )
    request = decision_input.request.model_copy(
        update={
            "allows_partial_payment": True,
            "desired_completion_date": state.forecast_end_date,
        }
    )
    return decision_input.model_copy(
        update={"financial_state": state.model_copy(update={"preferences": preferences}),
                "request": request}
    )


def test_partial_payment_splits_across_exactly_two_dates():
    decision_input = _partial_ready()
    decision = decide(decision_input)
    row = decision.to_row()

    assert row["recommended_payment_method"] == "partial_payment"
    assert row["affordability_status"] == "affordable_with_plan"

    first, second = decision.payments
    assert first[0] == decision_input.request.request_date
    assert second[0] == decision.earliest_date_for_full_payment
    assert first[1] + second[1] == decision_input.request.requested_amount
    assert Timeline(decision_input.financial_state).survives_payments(decision.payments)


def test_partial_payment_is_withheld_when_the_request_forbids_it():
    decision_input = _partial_ready()
    blocked = decision_input.model_copy(
        update={"request": decision_input.request.model_copy(
            update={"allows_partial_payment": False})}
    )
    assert decide(blocked).to_row()["recommended_payment_method"] != "partial_payment"


def test_partial_payment_is_withheld_when_the_remainder_misses_the_deadline():
    decision_input = _partial_ready()
    tight = decision_input.model_copy(
        update={"request": decision_input.request.model_copy(
            update={"desired_completion_date": decision_input.request.request_date})}
    )
    assert decide(tight).to_row()["recommended_payment_method"] != "partial_payment"


# -- golden outputs ---------------------------------------------------------

@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_golden_outputs_are_unchanged(name):
    expected = json.loads((GOLDEN / f"{name}.json").read_text())
    assert decide(load(name)).to_row() == expected
