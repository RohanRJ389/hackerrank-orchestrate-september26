"""Balance projection over the 90-day safety window.

The engine never infers recurrence: every movement it simulates is already a
dated occurrence in the financial state. This module only adds arithmetic.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Iterable, Optional

from contracts import (
    AdjustmentAction,
    CashFlowOccurrence,
    Direction,
    FinancialState,
)


@dataclass(frozen=True)
class SpendingChange:
    """A stop or reduce action the user has agreed to consider."""

    series_id: str
    target_event_id: str
    action: AdjustmentAction  # STOP_OR_REDUCE is resolved before this point
    new_amount: Optional[Decimal]  # None when stopping

    def render(self) -> str:
        if self.action is AdjustmentAction.STOP:
            return f"stop:{self.target_event_id}"
        return f"reduce_to:{self.target_event_id}:{format_amount(self.new_amount)}"


def format_amount(value: Decimal) -> str:
    """Whole numbers lose their decimals; everything else keeps exactly two.

    Mirrors the solved samples: `25256`, `620.40`, `23.50`.
    """
    quantized = value.quantize(Decimal("0.01"))
    if quantized == quantized.to_integral_value():
        return str(int(quantized))
    return f"{quantized:.2f}"


def signed(flow: CashFlowOccurrence) -> Decimal:
    return flow.amount if flow.direction is Direction.INFLOW else -flow.amount


class Timeline:
    """Projected balance across the forecast window.

    The balance is only inspected on days where something happens: between two
    movements it is flat, so a trough can only occur immediately after a
    movement (or at the opening balance).
    """

    def __init__(self, state: FinancialState, changes: Iterable[SpendingChange] = ()) -> None:
        self.state = state
        self.start = state.as_of_date
        self.end = state.forecast_end_date
        self.minimum = state.minimum_balance_to_keep

        stopped: set[str] = set()
        reduced: dict[str, Decimal] = {}
        for change in changes:
            series = next(s for s in state.adjustable_series if s.series_id == change.series_id)
            if change.action is AdjustmentAction.STOP:
                stopped.update(series.occurrence_ids)
            else:
                for occurrence_id in series.occurrence_ids:
                    reduced[occurrence_id] = change.new_amount

        movements: dict[date, Decimal] = {}
        for flow in state.cash_flows:
            if flow.occurrence_id in stopped:
                continue
            amount = signed(flow)
            if flow.occurrence_id in reduced:
                amount = -reduced[flow.occurrence_id]
            movements[flow.date] = movements.get(flow.date, Decimal(0)) + amount

        self._movements = movements
        self._dates: list[date] = sorted(movements)
        self._balances: list[Decimal] = []
        running = state.opening_balance
        for day in self._dates:
            running += movements[day]
            self._balances.append(running)
        self.closing_balance = running

        # suffix_min[i] is the lowest balance from _dates[i] onwards.
        self._suffix_min: list[Decimal] = [Decimal(0)] * len(self._balances)
        lowest = running
        for i in range(len(self._balances) - 1, -1, -1):
            lowest = min(lowest, self._balances[i])
            self._suffix_min[i] = lowest

    def balance_on(self, day: date) -> Decimal:
        """Balance at the end of `day`."""
        index = bisect_left(self._dates, day)
        if index < len(self._dates) and self._dates[index] == day:
            return self._balances[index]
        return self._balances[index - 1] if index else self.state.opening_balance

    def suffix_min(self, day: date) -> Decimal:
        """Lowest balance from the end of `day` onwards.

        A payment made on `day` lands after that day's movements, so money
        arriving on `day` — a salary credit, typically — is available to it.
        Reading the pre-movement balance instead would push every
        payday-funded recommendation to the day after payday.

        Troughs strictly before `day` are excluded on purpose: the payment
        cannot deepen a dip it comes after. `is_safe` covers those separately.
        """
        index = bisect_left(self._dates, day)
        if index >= len(self._dates):
            return self.closing_balance
        if self._dates[index] == day:
            return self._suffix_min[index]
        # Nothing happens on `day`; the balance carried in is the day's balance.
        carried = self._balances[index - 1] if index else self.state.opening_balance
        return min(carried, self._suffix_min[index])

    def lowest_balance(self) -> Decimal:
        return self.suffix_min(self.start)

    def headroom_on(self, day: date) -> Decimal:
        """How much could be taken out on `day` without breaching the minimum."""
        return self.suffix_min(day) - self.minimum

    def is_safe(self) -> bool:
        return self.lowest_balance() >= self.minimum

    def safe_to_pay_on(self, day: date, cap: Decimal) -> Decimal:
        """Largest payment on `day` that keeps the window safe, capped and floored."""
        return max(Decimal(0), min(cap, self.headroom_on(day)))

    def first_day_affording(self, amount: Decimal) -> Optional[date]:
        """Earliest day a single payment of `amount` stays safe.

        When the baseline already breaks the minimum on its own, no payment
        date can rescue it: paying before the breach makes it worse, and paying
        after it cannot undo it. So there is no such day.
        """
        if not self.is_safe():
            return None
        needed = self.minimum + amount
        if self.suffix_min(self.start) >= needed:
            return self.start
        for day in self._dates:
            if day > self.end:
                break
            if self.suffix_min(day) >= needed:
                return day
        return None

    def survives_payments(self, payments: Iterable[tuple[date, Decimal]]) -> bool:
        """Whether the window stays above the minimum once payments are applied.

        Payments falling after the forecast window are not simulated: there is
        no evidence to check them against.
        """
        combined = dict(self._movements)
        for day, amount in payments:
            if day > self.end:
                continue
            combined[day] = combined.get(day, Decimal(0)) - amount

        running = self.state.opening_balance
        if running < self.minimum:
            return False
        for day in sorted(combined):
            running += combined[day]
            if running < self.minimum:
                return False
        return True
