"""Ordering over safe plans, straight from the specification."""

from __future__ import annotations

from datetime import date

from .plans import Plan

_FAR_FUTURE = date.max


def completes_in_time(plan: Plan, deadline: date) -> bool:
    return plan.completion_date <= deadline


def rank_key(plan: Plan, deadline: date) -> tuple:
    """Sort key implementing the spec's preference order.

    1. Complete the full request by `desired_completion_date`
    2. Require no spending changes
    3. Minimise the total amount paid
    4. Start payment earlier
    5. Use fewer payments
    6. Lowest `payment_option_id`

    Fewest spending changes is inserted before the final tie-break: the spec
    treats "no changes" as a preference, so between two plans that both need
    changes, disturbing the user less is the better plan.
    """
    return (
        not completes_in_time(plan, deadline),
        bool(plan.changes),
        plan.total_paid,
        plan.start_date,
        plan.payment_count,
        len(plan.changes),
        plan.payment_option_id or "",
    )


def best_plan(plans: list[Plan], deadline: date) -> Plan | None:
    """The winning plan, or None when nothing completes the request in time.

    Missing the deadline is disqualifying rather than merely unattractive: a
    plan that never finishes the purchase is not a recommendation.
    """
    in_time = [plan for plan in plans if completes_in_time(plan, deadline)]
    if not in_time:
        return None
    return min(in_time, key=lambda plan: rank_key(plan, deadline))
