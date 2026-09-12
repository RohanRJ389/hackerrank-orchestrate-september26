"""Tests for the normalizer/decision-engine boundary contract."""

from __future__ import annotations

import copy
import csv
import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts import (
    SCHEMA_VERSION,
    ContractViolationError,
    DecisionInput,
    Severity,
    assert_valid,
    validate_decision_input,
)
from contracts.generate_schema import SCHEMA_PATH, build_schema

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"
DATASET = Path(__file__).resolve().parents[2] / "dataset"

VALID = sorted((FIXTURES / "valid").glob("*.json"))
STRUCTURAL = sorted((FIXTURES / "invalid_structural").glob("*.json"))
SEMANTIC = sorted((FIXTURES / "invalid_semantic").glob("*.json"))

EXPECTED_SEMANTIC_ERRORS = {
    "pending_credit_included": {"excluded_event_included"},
    "protected_category_adjustable": {
        "protected_flow_adjustable",
        "protected_category_adjustable",
        "reduce_not_permitted",
    },
    "recurrence_without_history": {"recurrence_without_history"},
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture
def doc() -> dict:
    """A structurally and semantically clean document, ready to be broken."""
    return load(FIXTURES / "valid" / "immediate_full_payment.json")


def errors_for(document: dict) -> set[str]:
    decision_input = DecisionInput.model_validate(document)
    return {
        violation.code
        for violation in validate_decision_input(decision_input)
        if violation.severity is Severity.ERROR
    }


# --- fixtures ------------------------------------------------------------


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.stem)
def test_valid_fixtures_pass_every_check(path: Path) -> None:
    decision_input = DecisionInput.model_validate(load(path))
    warnings = assert_valid(decision_input)
    assert all(w.severity is Severity.WARNING for w in warnings)


@pytest.mark.parametrize("path", STRUCTURAL, ids=lambda p: p.stem)
def test_structural_fixtures_fail_to_construct(path: Path) -> None:
    with pytest.raises(ValidationError):
        DecisionInput.model_validate(load(path))


@pytest.mark.parametrize("path", SEMANTIC, ids=lambda p: p.stem)
def test_semantic_fixtures_construct_but_do_not_validate(path: Path) -> None:
    decision_input = DecisionInput.model_validate(load(path))
    with pytest.raises(ContractViolationError):
        assert_valid(decision_input)
    assert errors_for(load(path)) == EXPECTED_SEMANTIC_ERRORS[path.stem]


# --- serialization -------------------------------------------------------


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.stem)
def test_round_trip_is_stable(path: Path) -> None:
    original = DecisionInput.model_validate(load(path))
    reloaded = DecisionInput.model_validate(json.loads(original.model_dump_json()))
    assert reloaded == original


def test_money_survives_as_exact_decimal(doc: dict) -> None:
    decision_input = DecisionInput.model_validate(doc)
    fuel = next(
        flow for flow in decision_input.financial_state.cash_flows
        if flow.occurrence_id == "occ_pending_fuel"
    )
    assert fuel.amount == Decimal("567.60")

    payload = json.loads(decision_input.model_dump_json())
    amounts = [flow["amount"] for flow in payload["financial_state"]["cash_flows"]]
    assert all(isinstance(amount, str) for amount in amounts)
    assert "567.60" in amounts


def test_float_amounts_do_not_silently_lose_precision(doc: dict) -> None:
    doc["financial_state"]["opening_balance"] = "58481.10"
    decision_input = DecisionInput.model_validate(doc)
    assert decision_input.financial_state.opening_balance == Decimal("58481.10")


def test_unknown_fields_are_rejected(doc: dict) -> None:
    doc["financial_state"]["surprise"] = 1
    with pytest.raises(ValidationError):
        DecisionInput.model_validate(doc)


def test_checked_in_schema_matches_the_models() -> None:
    assert json.loads(SCHEMA_PATH.read_text()) == build_schema(), (
        "run `cd code && python3 -m contracts.generate_schema`"
    )


def test_contract_version_is_pinned(doc: dict) -> None:
    assert DecisionInput.model_validate(doc).schema_version == SCHEMA_VERSION
    doc["schema_version"] = "2.0.0"
    with pytest.raises(ValidationError):
        DecisionInput.model_validate(doc)


# --- structural invariants -----------------------------------------------


def test_forecast_window_must_be_ninety_days(doc: dict) -> None:
    doc["financial_state"]["forecast_end_date"] = "2024-05-01"
    with pytest.raises(ValidationError, match="90 days"):
        DecisionInput.model_validate(doc)


def test_cash_flow_must_fall_inside_the_window(doc: dict) -> None:
    doc["financial_state"]["cash_flows"][0]["date"] = "2024-07-01"
    with pytest.raises(ValidationError, match="outside the forecast window"):
        DecisionInput.model_validate(doc)


def test_occurrence_ids_must_be_unique(doc: dict) -> None:
    flows = doc["financial_state"]["cash_flows"]
    flows.append(copy.deepcopy(flows[0]))
    with pytest.raises(ValidationError, match="duplicate occurrence_id"):
        DecisionInput.model_validate(doc)


def test_expense_origin_cannot_produce_income(doc: dict) -> None:
    flow = doc["financial_state"]["cash_flows"][0]
    flow["direction"] = "inflow"
    with pytest.raises(ValidationError, match="is invalid for inflow"):
        DecisionInput.model_validate(doc)


def test_conversion_fields_travel_together(doc: dict) -> None:
    doc["financial_state"]["cash_flows"][0]["original_currency"] = "USD"
    with pytest.raises(ValidationError, match="must be set together"):
        DecisionInput.model_validate(doc)


def test_converted_amount_must_cite_its_rate() -> None:
    doc = load(FIXTURES / "valid" / "foreign_currency_income.json")
    flow = doc["financial_state"]["cash_flows"][0]
    flow["sources"] = [s for s in flow["sources"] if s["kind"] != "exchange_rate"]
    with pytest.raises(ValidationError, match="exchange-rate row"):
        DecisionInput.model_validate(doc)


def test_series_must_reference_known_occurrences(doc: dict) -> None:
    doc["financial_state"]["adjustable_series"][0]["occurrence_ids"] = ["occ_missing"]
    with pytest.raises(ValidationError, match="unknown occurrences"):
        DecisionInput.model_validate(doc)


def test_reducible_series_needs_a_floor() -> None:
    doc = load(FIXTURES / "valid" / "wait_for_confirmed_salary.json")
    doc["financial_state"]["adjustable_series"][0]["minimum_allowed_amount"] = None
    with pytest.raises(ValidationError, match="minimum allowed amount"):
        DecisionInput.model_validate(doc)


def test_reduction_floor_must_be_below_the_current_amount() -> None:
    doc = load(FIXTURES / "valid" / "wait_for_confirmed_salary.json")
    doc["financial_state"]["adjustable_series"][0]["minimum_allowed_amount"] = "117800"
    with pytest.raises(ValidationError, match="below the current amount"):
        DecisionInput.model_validate(doc)


def test_installments_require_a_declared_month_cap(doc: dict) -> None:
    preferences = doc["financial_state"]["preferences"]
    preferences["accepted_payment_methods"] = ["installments"]
    preferences["max_installment_months"] = None
    with pytest.raises(ValidationError, match="max_installment_months is required"):
        DecisionInput.model_validate(doc)


def test_month_cap_is_meaningless_without_installments(doc: dict) -> None:
    doc["financial_state"]["preferences"]["max_installment_months"] = 6
    with pytest.raises(ValidationError, match="must be empty"):
        DecisionInput.model_validate(doc)


def test_unresolved_conflicts_must_choose_the_safer_reading() -> None:
    doc = load(FIXTURES / "valid" / "installment_plan.json")
    doc["financial_state"]["evidence_resolutions"][0]["is_conservative_fallback"] = True
    with pytest.raises(ValidationError, match="safer interpretation"):
        DecisionInput.model_validate(doc)


def test_schedule_must_add_up_to_the_total(doc: dict) -> None:
    option = doc["request"]["payment_options"][1]
    option["schedule"][0]["amount"] = "1"
    with pytest.raises(ValidationError, match="add up to total_payable_amount"):
        DecisionInput.model_validate(doc)


def test_schedule_must_follow_the_stated_frequency(doc: dict) -> None:
    option = doc["request"]["payment_options"][1]
    option["schedule"][1]["date"] = "2024-04-11"
    with pytest.raises(ValidationError, match="payment_frequency_days"):
        DecisionInput.model_validate(doc)


def test_full_payment_option_is_a_single_payment(doc: dict) -> None:
    option = doc["request"]["payment_options"][0]
    option["number_of_payments"] = 2
    with pytest.raises(ValidationError, match="exactly one payment"):
        DecisionInput.model_validate(doc)


def test_payment_option_ids_are_unique(doc: dict) -> None:
    options = doc["request"]["payment_options"]
    options[1]["payment_option_id"] = options[0]["payment_option_id"]
    with pytest.raises(ValidationError, match="unique within a request"):
        DecisionInput.model_validate(doc)


def test_deadline_cannot_precede_the_request(doc: dict) -> None:
    doc["request"]["desired_completion_date"] = "2024-03-01"
    with pytest.raises(ValidationError, match="cannot precede request_date"):
        DecisionInput.model_validate(doc)


def test_state_and_request_must_share_a_date_and_currency(doc: dict) -> None:
    doc["request"]["request_date"] = "2024-03-04"
    with pytest.raises(ValidationError, match="as of the request date"):
        DecisionInput.model_validate(doc)


# --- semantic invariants -------------------------------------------------


def test_income_cannot_be_treated_as_flexible_spending(doc: dict) -> None:
    for flow in doc["financial_state"]["cash_flows"]:
        if flow["direction"] == "inflow":
            flow["adjustable_series_id"] = "series_delivery"
            break
    assert "income_marked_adjustable" in errors_for(doc)


def test_protected_categories_must_be_flagged(doc: dict) -> None:
    for flow in doc["financial_state"]["cash_flows"]:
        if flow["category"] == "rent":
            flow["is_protected"] = False
    assert "protection_not_marked" in errors_for(doc)


def test_one_off_expenses_cannot_be_adjusted(doc: dict) -> None:
    for flow in doc["financial_state"]["cash_flows"]:
        if flow["occurrence_id"] == "occ_delivery_2024_03":
            flow["origin"] = "committed_one_time_debit"
    assert "adjustment_targets_one_off" in errors_for(doc)


def test_two_series_cannot_target_the_same_event(doc: dict) -> None:
    series = doc["financial_state"]["adjustable_series"]
    duplicate = copy.deepcopy(series[0])
    duplicate["series_id"] = "series_delivery_again"
    series.append(duplicate)
    for flow in doc["financial_state"]["cash_flows"]:
        if flow["adjustable_series_id"] == "series_delivery":
            flow["adjustable_series_id"] = "series_delivery"
    assert "duplicate_adjustment_target" in errors_for(doc)


def test_choosing_between_records_requires_citing_both() -> None:
    doc = load(FIXTURES / "valid" / "installment_plan.json")
    resolution = doc["financial_state"]["evidence_resolutions"][0]
    resolution["sources"] = resolution["sources"][:1]
    assert "conflict_without_both_records" in errors_for(doc)


def test_fee_and_total_must_agree_with_the_requested_amount(doc: dict) -> None:
    doc["request"]["payment_options"][0]["financing_fee"] = "100"
    assert "fee_total_mismatch" in errors_for(doc)


def test_option_cannot_start_before_the_request(doc: dict) -> None:
    option = doc["request"]["payment_options"][0]
    option["first_payment_date"] = "2024-03-01"
    option["schedule"][0]["date"] = "2024-03-01"
    assert "option_starts_before_request" in errors_for(doc)


def test_opening_balance_below_minimum_is_only_a_warning(doc: dict) -> None:
    doc["financial_state"]["opening_balance"] = "1000"
    decision_input = DecisionInput.model_validate(doc)
    warnings = assert_valid(decision_input)
    assert "opening_below_minimum" in {w.code for w in warnings}


# --- fidelity to the dataset ---------------------------------------------


def _dataset_rows(filename: str, key: str) -> dict[str, dict]:
    with open(DATASET / filename, newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.stem)
def test_fixture_requests_match_the_dataset(path: Path) -> None:
    decision_input = DecisionInput.model_validate(load(path))
    request = decision_input.request
    row = _dataset_rows("sample_requests.csv", "request_id")[request.request_id]

    assert request.user_id == row["user_id"]
    assert request.request_date.isoformat() == row["request_date"]
    assert request.requested_amount == Decimal(row["requested_amount"])
    assert request.desired_completion_date.isoformat() == row["desired_completion_date"]
    assert request.allows_partial_payment == (row["allows_partial_payment"] == "true")

    profile = _dataset_rows("financial_profiles.csv", "user_id")[request.user_id]
    state = decision_input.financial_state
    assert state.home_currency.value == profile["home_currency"]
    assert state.opening_balance == Decimal(profile["current_available_balance"])
    assert state.minimum_balance_to_keep == Decimal(profile["minimum_balance_to_keep"])


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.stem)
def test_fixture_payment_options_match_the_dataset(path: Path) -> None:
    decision_input = DecisionInput.model_validate(load(path))
    offered = _dataset_rows("request_payment_options.csv", "payment_option_id")

    for option in decision_input.request.payment_options:
        row = offered[option.payment_option_id]
        assert row["request_id"] == decision_input.request.request_id
        assert option.method.value == row["payment_method"]
        assert option.payment_amount == Decimal(row["payment_amount"])
        assert option.number_of_payments == int(row["number_of_payments"])
        assert option.total_payable_amount == Decimal(row["total_payable_amount"])
