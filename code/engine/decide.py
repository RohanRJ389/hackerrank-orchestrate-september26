"""Turn a normalized financial state and request into one recommendation."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from contracts import DecisionInput, PaymentMethod

from .forecast import SpendingChange, Timeline, format_amount
from .output import DecisionOutput
from .plans import Plan, candidate_plans
from .ranking import best_plan

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _spoken_date(day: date) -> str:
    return f"{day.day} {_MONTHS[day.month - 1]} {day.year}"


def _money(currency: str, value: Decimal) -> str:
    formatted = format_amount(value)
    if "." in formatted:
        whole, fraction = formatted.split(".")
        return f"{currency} {int(whole):,}.{fraction}"
    return f"{currency} {int(formatted):,}"


def _status_for(plan: Plan) -> str:
    if plan.method == "wait":
        return "affordable_later"
    if plan.method is PaymentMethod.FULL_PAYMENT and not plan.changes:
        return "affordable_now"
    return "affordable_with_plan"


def _method_for(plan: Plan) -> str:
    # PaymentMethod subclasses str, so check the enum before the str fallback.
    return plan.method.value if isinstance(plan.method, PaymentMethod) else plan.method


def _sentence_case(text: str) -> str:
    """Upper-case the first letter only, leaving currency codes intact."""
    return text[0].upper() + text[1:] if text else text


def _describe_changes(changes: tuple[SpendingChange, ...], currency: str) -> str:
    parts = []
    for change in changes:
        if change.new_amount is None:
            parts.append(f"stop {change.target_event_id}")
        else:
            parts.append(f"reduce {change.target_event_id} to {_money(currency, change.new_amount)}")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _explain(
    decision_input: DecisionInput,
    plan: Optional[Plan],
    safe_today: Decimal,
    earliest_full: Optional[date],
) -> str:
    request = decision_input.request
    state = decision_input.financial_state
    currency = state.home_currency.value
    minimum = _money(currency, state.minimum_balance_to_keep)
    amount = _money(currency, request.requested_amount)

    if plan is None:
        if earliest_full is None:
            return (
                f"Do not proceed with the {amount} request. Although "
                f"{_money(currency, safe_today)} is available today, the full amount "
                f"cannot be completed safely within 90 days."
            )
        return (
            f"Do not make this payment by {_spoken_date(request.desired_completion_date)}. "
            f"None of the available options keeps the {minimum} minimum protected."
        )

    prefix = ""
    if plan.changes:
        prefix = f"{_sentence_case(_describe_changes(plan.changes, currency))}, then "

    if plan.method == "wait":
        return (
            f"Pay {amount} in full on {_spoken_date(plan.start_date)}. "
            f"Paying earlier would take the balance below the {minimum} minimum."
        )

    if plan.method is PaymentMethod.FULL_PAYMENT:
        if plan.start_date == request.request_date:
            body = f"pay {amount} today"
        else:
            body = f"pay {amount} on {_spoken_date(plan.start_date)}"
        sentence = (prefix + body) if prefix else _sentence_case(body)
        return f"{sentence}. This leaves at least {minimum} available."

    if plan.method is PaymentMethod.PARTIAL_PAYMENT:
        first, second = plan.payments
        body = (
            f"pay {_money(currency, first[1])} today and the remaining "
            f"{_money(currency, second[1])} on {_spoken_date(second[0])}"
        )
        sentence = (prefix + body) if prefix else _sentence_case(body)
        return (
            f"{sentence}. This completes the full request and keeps the "
            f"{minimum} minimum protected."
        )

    instalment = _money(currency, plan.payments[0][1])
    body = (
        f"use {plan.payment_count} installments of {instalment}, starting "
        f"{_spoken_date(plan.start_date)}"
    )
    sentence = (prefix + body) if prefix else _sentence_case(body)
    return f"{sentence}. This leaves at least {minimum} available."


def decide(decision_input: DecisionInput) -> DecisionOutput:
    """Produce the recommendation for one request.

    `amount_safe_to_pay` and `earliest_date_for_full_payment` describe raw
    financial capacity: both ignore the user's method preferences and any
    optional spending changes, which is why a request can be capable of full
    payment today yet still be recommended as installments.
    """
    request = decision_input.request
    baseline = Timeline(decision_input.financial_state)

    safe_today = baseline.safe_to_pay_on(request.request_date, request.requested_amount)
    earliest_full = baseline.first_day_affording(request.requested_amount)

    plans = candidate_plans(decision_input, baseline, safe_today, earliest_full)
    winner = best_plan(plans, request.desired_completion_date)

    if winner is None:
        return DecisionOutput(
            request_id=request.request_id,
            amount_safe_to_pay=safe_today,
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payments=(),
            earliest_date_for_full_payment=earliest_full,
            spending_changes=(),
            decision_explanation=_explain(decision_input, None, safe_today, earliest_full),
        )

    return DecisionOutput(
        request_id=request.request_id,
        amount_safe_to_pay=safe_today,
        affordability_status=_status_for(winner),
        recommended_payment_method=_method_for(winner),
        payments=winner.payments,
        earliest_date_for_full_payment=earliest_full,
        spending_changes=winner.changes,
        decision_explanation=_explain(decision_input, winner, safe_today, earliest_full),
    )
