"""Candidate plans the engine may recommend, and the ones it may not."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from itertools import combinations, product
from typing import Optional

from contracts import (
    AdjustableSeries,
    AdjustmentAction,
    DecisionInput,
    OfferedPaymentMethod,
    PaymentMethod,
    PaymentOption,
)

from .forecast import SpendingChange, Timeline

MAX_SPENDING_CHANGES = 3


@dataclass(frozen=True)
class Plan:
    """A concrete, simulated way to satisfy the request."""

    method: PaymentMethod | str  # PaymentMethod, or "wait"
    payments: tuple[tuple[date, Decimal], ...]
    changes: tuple[SpendingChange, ...] = ()
    payment_option_id: Optional[str] = None

    @property
    def total_paid(self) -> Decimal:
        return sum((amount for _, amount in self.payments), Decimal(0))

    @property
    def start_date(self) -> date:
        return self.payments[0][0]

    @property
    def completion_date(self) -> date:
        return self.payments[-1][0]

    @property
    def payment_count(self) -> int:
        return len(self.payments)


def _actions_for(series: AdjustableSeries) -> list[SpendingChange]:
    """Every distinct action this series permits.

    Stopping and reducing the same event are mutually exclusive, so a series
    contributes at most one action to any candidate set.
    """
    actions: list[SpendingChange] = []
    if series.action in (AdjustmentAction.STOP, AdjustmentAction.STOP_OR_REDUCE):
        actions.append(
            SpendingChange(series.series_id, series.target_event_id, AdjustmentAction.STOP, None)
        )
    if series.action in (AdjustmentAction.REDUCE, AdjustmentAction.STOP_OR_REDUCE):
        actions.append(
            SpendingChange(
                series.series_id,
                series.target_event_id,
                AdjustmentAction.REDUCE,
                series.minimum_allowed_amount,
            )
        )
    return actions


def change_sets(decision_input: DecisionInput) -> list[tuple[SpendingChange, ...]]:
    """Allowed spending-change combinations, fewest changes first.

    Reductions always go to the floor: that is what the solved samples do, and
    a smaller cut can only make a blocked plan less likely to fit.
    """
    series_list = list(decision_input.financial_state.adjustable_series)
    results: list[tuple[SpendingChange, ...]] = []
    for size in range(1, min(MAX_SPENDING_CHANGES, len(series_list)) + 1):
        for chosen in combinations(series_list, size):
            for actions in product(*(_actions_for(s) for s in chosen)):
                results.append(tuple(actions))
    return results


def eligible_installment_options(decision_input: DecisionInput) -> list[PaymentOption]:
    """Supplied installment offers the user's preferences actually allow."""
    preferences = decision_input.financial_state.preferences
    if PaymentMethod.INSTALLMENTS not in preferences.accepted_payment_methods:
        return []
    cap = preferences.max_installment_months
    return [
        option
        for option in decision_input.request.payment_options
        if option.method is OfferedPaymentMethod.INSTALLMENTS
        and (cap is None or option.number_of_payments <= cap)
    ]


def _full_payment_plans(
    decision_input: DecisionInput,
    baseline: Timeline,
    earliest_full: Optional[date],
) -> list[Plan]:
    request = decision_input.request
    preferences = decision_input.financial_state.preferences
    if PaymentMethod.FULL_PAYMENT not in preferences.accepted_payment_methods:
        return []

    amount = request.requested_amount
    today = request.request_date
    plans: list[Plan] = []

    if baseline.survives_payments([(today, amount)]):
        plans.append(Plan(PaymentMethod.FULL_PAYMENT, ((today, amount),)))
    else:
        for changes in change_sets(decision_input):
            adjusted = Timeline(decision_input.financial_state, changes)
            if adjusted.survives_payments([(today, amount)]):
                plans.append(Plan(PaymentMethod.FULL_PAYMENT, ((today, amount),), changes))

    # Waiting is the same payment, later. It only makes sense if paying today
    # was not already safe.
    if earliest_full is not None and earliest_full > today:
        plans.append(Plan("wait", ((earliest_full, amount),)))

    return plans


def _partial_payment_plan(
    decision_input: DecisionInput,
    safe_today: Decimal,
    earliest_full: Optional[date],
) -> list[Plan]:
    request = decision_input.request
    preferences = decision_input.financial_state.preferences

    if not request.allows_partial_payment:
        return []
    if PaymentMethod.PARTIAL_PAYMENT not in preferences.accepted_payment_methods:
        return []
    if not (Decimal(0) < safe_today < request.requested_amount):
        return []
    if earliest_full is None or earliest_full > request.desired_completion_date:
        return []

    remainder = request.requested_amount - safe_today
    return [
        Plan(
            PaymentMethod.PARTIAL_PAYMENT,
            ((request.request_date, safe_today), (earliest_full, remainder)),
        )
    ]


def _installment_plans(decision_input: DecisionInput, baseline: Timeline) -> list[Plan]:
    plans: list[Plan] = []
    for option in eligible_installment_options(decision_input):
        schedule = tuple((entry.date, entry.amount) for entry in option.schedule)
        if baseline.survives_payments(schedule):
            plans.append(
                Plan(PaymentMethod.INSTALLMENTS, schedule, (), option.payment_option_id)
            )
            continue
        for changes in change_sets(decision_input):
            adjusted = Timeline(decision_input.financial_state, changes)
            if adjusted.survives_payments(schedule):
                plans.append(
                    Plan(
                        PaymentMethod.INSTALLMENTS,
                        schedule,
                        changes,
                        option.payment_option_id,
                    )
                )
                break
    return plans


def candidate_plans(
    decision_input: DecisionInput,
    baseline: Timeline,
    safe_today: Decimal,
    earliest_full: Optional[date],
) -> list[Plan]:
    """Every safe plan the user's preferences permit, before ranking."""
    return [
        *_full_payment_plans(decision_input, baseline, earliest_full),
        *_partial_payment_plan(decision_input, safe_today, earliest_full),
        *_installment_plans(decision_input, baseline),
    ]
