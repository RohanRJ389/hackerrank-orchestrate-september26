"""Semantic checks that span models and the user's stated policy.

Structural rules live on the models themselves. These checks answer a different
question: is the signed financial state a legitimate reading of the dataset and
of what the user actually permits?
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from .models import (
    AdjustmentAction,
    CashFlowOrigin,
    Certainty,
    DecisionInput,
    Direction,
    EvidenceKind,
    FinancialState,
    OfferedPaymentMethod,
    PaymentMethod,
    RECURRING_ORIGINS,
    ResolutionBasis,
)

AMOUNT_TOLERANCE = Decimal("0.05")


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ContractViolation:
    code: str
    message: str
    severity: Severity
    location: str

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.location}: {self.message} ({self.code})"


class ContractViolationError(ValueError):
    def __init__(self, violations: list[ContractViolation]) -> None:
        self.violations = violations
        super().__init__("\n".join(str(violation) for violation in violations))


def validate_financial_state(state: FinancialState) -> list[ContractViolation]:
    violations: list[ContractViolation] = []
    preferences = state.preferences
    flows_by_id = {flow.occurrence_id: flow for flow in state.cash_flows}

    if state.opening_balance < state.minimum_balance_to_keep:
        violations.append(
            ContractViolation(
                code="opening_below_minimum",
                message="opening balance already sits below the minimum the user keeps",
                severity=Severity.WARNING,
                location=f"financial_state[{state.user_id}]",
            )
        )

    excluded_event_ids = {
        excluded.source.ref_id
        for excluded in state.excluded_evidence
        if excluded.source.kind is EvidenceKind.FINANCIAL_EVENT
    }

    for flow in state.cash_flows:
        location = f"cash_flows[{flow.occurrence_id}]"

        expected_protection = flow.category in preferences.protected_categories
        if expected_protection and not flow.is_protected:
            violations.append(
                ContractViolation(
                    code="protection_not_marked",
                    message=f"category '{flow.category}' is protected by the user profile",
                    severity=Severity.ERROR,
                    location=location,
                )
            )
        if flow.is_protected and flow.adjustable_series_id is not None:
            violations.append(
                ContractViolation(
                    code="protected_flow_adjustable",
                    message="a protected expense cannot belong to an adjustable series",
                    severity=Severity.ERROR,
                    location=location,
                )
            )
        if flow.direction is Direction.INFLOW and flow.adjustable_series_id is not None:
            violations.append(
                ContractViolation(
                    code="income_marked_adjustable",
                    message="income cannot be stopped or reduced as flexible spending",
                    severity=Severity.ERROR,
                    location=location,
                )
            )

        if flow.origin in RECURRING_ORIGINS and not any(
            source.kind is EvidenceKind.FINANCIAL_EVENT for source in flow.sources
        ):
            violations.append(
                ContractViolation(
                    code="recurrence_without_history",
                    message="a recurring projection must cite the history that supports it",
                    severity=Severity.ERROR,
                    location=location,
                )
            )

        if flow.origin is CashFlowOrigin.CONFIRMED_INCOME and flow.certainty is not Certainty.CONFIRMED:
            violations.append(
                ContractViolation(
                    code="confirmed_income_not_confirmed",
                    message="confirmed income must be marked as confirmed",
                    severity=Severity.ERROR,
                    location=location,
                )
            )

        if flow.origin is CashFlowOrigin.RESERVED_PENDING_DEBIT and flow.certainty is not Certainty.CONFIRMED:
            violations.append(
                ContractViolation(
                    code="pending_debit_not_confirmed",
                    message="a reserved pending debit must be treated as confirmed",
                    severity=Severity.ERROR,
                    location=location,
                )
            )

        if flow.source_event_id is not None and flow.source_event_id in excluded_event_ids:
            violations.append(
                ContractViolation(
                    code="excluded_event_included",
                    message=f"event {flow.source_event_id} is both excluded and projected",
                    severity=Severity.ERROR,
                    location=location,
                )
            )

    seen_targets: set[str] = set()
    for series in state.adjustable_series:
        location = f"adjustable_series[{series.series_id}]"

        if series.target_event_id in seen_targets:
            violations.append(
                ContractViolation(
                    code="duplicate_adjustment_target",
                    message=f"event {series.target_event_id} is targeted by more than one series",
                    severity=Severity.ERROR,
                    location=location,
                )
            )
        seen_targets.add(series.target_event_id)

        if series.category in preferences.protected_categories:
            violations.append(
                ContractViolation(
                    code="protected_category_adjustable",
                    message=f"category '{series.category}' is protected and cannot be changed",
                    severity=Severity.ERROR,
                    location=location,
                )
            )

        may_stop = series.category in preferences.stoppable_categories
        may_reduce = series.category in preferences.reducible_categories
        if series.action is AdjustmentAction.STOP and not may_stop:
            violations.append(
                ContractViolation(
                    code="stop_not_permitted",
                    message=f"the user will not stop spending in '{series.category}'",
                    severity=Severity.ERROR,
                    location=location,
                )
            )
        if series.action is AdjustmentAction.REDUCE and not may_reduce:
            violations.append(
                ContractViolation(
                    code="reduce_not_permitted",
                    message=f"the user will not reduce spending in '{series.category}'",
                    severity=Severity.ERROR,
                    location=location,
                )
            )
        if series.action is AdjustmentAction.STOP_OR_REDUCE and not (may_stop and may_reduce):
            violations.append(
                ContractViolation(
                    code="stop_or_reduce_not_permitted",
                    message=f"the user permits only one kind of change in '{series.category}'",
                    severity=Severity.ERROR,
                    location=location,
                )
            )

        for occurrence_id in series.occurrence_ids:
            flow = flows_by_id[occurrence_id]
            if flow.direction is not Direction.OUTFLOW:
                violations.append(
                    ContractViolation(
                        code="adjustment_targets_inflow",
                        message=f"occurrence {occurrence_id} is not an expense",
                        severity=Severity.ERROR,
                        location=location,
                    )
                )
            if flow.origin not in RECURRING_ORIGINS:
                violations.append(
                    ContractViolation(
                        code="adjustment_targets_one_off",
                        message=f"occurrence {occurrence_id} is not a recurring expense",
                        severity=Severity.ERROR,
                        location=location,
                    )
                )
            if flow.category != series.category:
                violations.append(
                    ContractViolation(
                        code="adjustment_category_mismatch",
                        message=f"occurrence {occurrence_id} is not in '{series.category}'",
                        severity=Severity.ERROR,
                        location=location,
                    )
                )
            if flow.adjustable_series_id != series.series_id:
                violations.append(
                    ContractViolation(
                        code="adjustment_link_not_mutual",
                        message=f"occurrence {occurrence_id} does not link back to this series",
                        severity=Severity.ERROR,
                        location=location,
                    )
                )
            if abs(flow.amount - series.current_amount) > AMOUNT_TOLERANCE:
                violations.append(
                    ContractViolation(
                        code="adjustment_amount_mismatch",
                        message=(
                            f"occurrence {occurrence_id} is {flow.amount}, but the series "
                            f"reports {series.current_amount}"
                        ),
                        severity=Severity.WARNING,
                        location=location,
                    )
                )

    for resolution in state.evidence_resolutions:
        location = f"evidence_resolutions[{resolution.resolution_id}]"
        needs_two_records = resolution.basis in (
            ResolutionBasis.NEWER_RECORD_SAME_SOURCE,
            ResolutionBasis.SETTLED_OVER_ESTIMATE,
        )
        if needs_two_records and len(resolution.sources) < 2:
            violations.append(
                ContractViolation(
                    code="conflict_without_both_records",
                    message="choosing between records requires citing both of them",
                    severity=Severity.ERROR,
                    location=location,
                )
            )

    return violations


def validate_decision_input(decision_input: DecisionInput) -> list[ContractViolation]:
    violations = validate_financial_state(decision_input.financial_state)
    request = decision_input.request
    preferences = decision_input.financial_state.preferences

    for option in request.payment_options:
        location = f"payment_options[{option.payment_option_id}]"

        expected_total = request.requested_amount + option.financing_fee
        if abs(expected_total - option.total_payable_amount) > AMOUNT_TOLERANCE:
            violations.append(
                ContractViolation(
                    code="fee_total_mismatch",
                    message=(
                        f"requested {request.requested_amount} plus fee {option.financing_fee} "
                        f"is not {option.total_payable_amount}"
                    ),
                    severity=Severity.ERROR,
                    location=location,
                )
            )

        if option.method is OfferedPaymentMethod.FULL_PAYMENT:
            if option.financing_fee != 0:
                violations.append(
                    ContractViolation(
                        code="full_payment_has_fee",
                        message="paying in full should not carry a financing fee",
                        severity=Severity.WARNING,
                        location=location,
                    )
                )
            if option.first_payment_date != request.request_date:
                violations.append(
                    ContractViolation(
                        code="full_payment_not_on_request_date",
                        message="a full payment option should be payable on the request date",
                        severity=Severity.WARNING,
                        location=location,
                    )
                )

        if option.first_payment_date < request.request_date:
            violations.append(
                ContractViolation(
                    code="option_starts_before_request",
                    message="a payment option cannot begin before the request date",
                    severity=Severity.ERROR,
                    location=location,
                )
            )

    if not request.allows_partial_payment and PaymentMethod.PARTIAL_PAYMENT in (
        preferences.accepted_payment_methods
    ):
        violations.append(
            ContractViolation(
                code="partial_payment_unavailable",
                message="the user accepts partial payment but this request does not allow it",
                severity=Severity.WARNING,
                location=f"request[{request.request_id}]",
            )
        )

    return violations


def assert_valid(decision_input: DecisionInput) -> list[ContractViolation]:
    """Raise on error-level violations and return any remaining warnings."""

    violations = validate_decision_input(decision_input)
    errors = [v for v in violations if v.severity is Severity.ERROR]
    if errors:
        raise ContractViolationError(errors)
    return violations
