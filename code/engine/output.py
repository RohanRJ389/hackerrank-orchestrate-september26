"""The output.csv row produced for one request."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Optional

from .forecast import SpendingChange, format_amount

CSV_COLUMNS = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)


def format_safe_amount(value: Decimal) -> str:
    """Capacity is reported as a plain number, without padding to two decimals.

    The samples write `17229139.2` and `603.3`, unlike the payment plan which
    pads to `620.40`.
    """
    normalized = value.quantize(Decimal("0.01")).normalize()
    if normalized == normalized.to_integral_value():
        return str(int(normalized))
    return format(normalized, "f")


@dataclass(frozen=True)
class DecisionOutput:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payments: tuple[tuple[date, Decimal], ...]
    earliest_date_for_full_payment: Optional[date]
    spending_changes: tuple[SpendingChange, ...]
    decision_explanation: str

    @property
    def payment_plan(self) -> str:
        if not self.payments:
            return "none"
        return "|".join(
            f"{day.isoformat()}:{format_amount(amount)}" for day, amount in self.payments
        )

    @property
    def spending_changes_needed(self) -> str:
        if not self.spending_changes:
            return "none"
        return "|".join(change.render() for change in self.spending_changes)

    def to_row(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "amount_safe_to_pay": format_safe_amount(self.amount_safe_to_pay),
            "affordability_status": self.affordability_status,
            "recommended_payment_method": self.recommended_payment_method,
            "payment_plan": self.payment_plan,
            "earliest_date_for_full_payment": (
                self.earliest_date_for_full_payment.isoformat()
                if self.earliest_date_for_full_payment
                else ""
            ),
            "spending_changes_needed": self.spending_changes_needed,
            "decision_explanation": self.decision_explanation,
        }
