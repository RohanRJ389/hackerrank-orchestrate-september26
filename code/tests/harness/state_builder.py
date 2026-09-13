"""Test-only deterministic `DecisionInput` builder.

This exists so the decision engine can be scored against the 25 solved samples
before the real normalizer lands. It reads only structured CSV data and applies
simple, explainable heuristics.

It deliberately ignores `messages.csv` and `images.csv`. Those need judgement,
which is the normalizer's job, so any sample whose answer depends on a payroll
amendment or an image-derived amount is expected to diverge here. The scoreboard
labels that divergence rather than hiding it.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import median
from typing import Iterable, Optional

from contracts import DecisionInput

DATASET = Path(__file__).resolve().parents[3] / "dataset"
HORIZON_DAYS = 90

# Categories whose spending continues whether or not the user plans for it.
VARIABLE_ESSENTIAL = {"groceries", "transport", "dining", "healthcare", "shopping",
                      "entertainment"}
CASH_IGNORED_STATUSES = {"cancelled", "failed", "unrealized"}
NON_CASH_EVENT_TYPES = {"investment_valuation"}


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Optional[Decimal]
    currency: str
    event_date: date
    settlement_date: Optional[date]
    status: str
    linked_event_id: str
    flexibility: str
    minimum_allowed_amount: Optional[Decimal]

    @property
    def is_credit(self) -> bool:
        return self.direction == "credit"


def _decimal(value: str) -> Optional[Decimal]:
    return Decimal(value) if value else None


def _date(value: str) -> Optional[date]:
    return date.fromisoformat(value) if value else None


class Dataset:
    """Lazily parsed dataset, shared across all sample builds."""

    _instance: Optional["Dataset"] = None

    def __init__(self) -> None:
        self.profiles = {r["user_id"]: r for r in self._read("financial_profiles.csv")}
        self.samples = {r["request_id"]: r for r in self._read("sample_requests.csv")}
        self.requests = {r["request_id"]: r for r in self._read("requests.csv")}
        self.requests.update(self.samples)

        self.events_by_user: dict[str, list[Event]] = defaultdict(list)
        for row in self._read("financial_events.csv"):
            event = Event(
                event_id=row["event_id"],
                user_id=row["user_id"],
                event_type=row["event_type"],
                description=row["description"],
                category=row["category"],
                direction=row["direction"],
                amount=_decimal(row["amount"]),
                currency=row["currency"],
                event_date=_date(row["event_date"]),
                settlement_date=_date(row["settlement_date"]),
                status=row["status"],
                linked_event_id=row["linked_event_id"],
                flexibility=row["flexibility"],
                minimum_allowed_amount=_decimal(row["minimum_allowed_amount"]),
            )
            self.events_by_user[event.user_id].append(event)

        self.options_by_request: dict[str, list[dict]] = defaultdict(list)
        for row in self._read("request_payment_options.csv"):
            self.options_by_request[row["request_id"]].append(row)

        self.rates: dict[tuple[str, str], list[tuple[date, Decimal]]] = defaultdict(list)
        for row in self._read("exchange_rates.csv"):
            key = (row["from_currency"], row["to_currency"])
            self.rates[key].append((date.fromisoformat(row["rate_date"]), Decimal(row["rate"])))
        for series in self.rates.values():
            series.sort()

    @staticmethod
    def _read(name: str) -> list[dict]:
        with open(DATASET / name, newline="") as handle:
            return list(csv.DictReader(handle))

    @classmethod
    def load(cls) -> "Dataset":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def convert(self, amount: Decimal, source: str, target: str, when: date) -> Decimal:
        """Convert using the rate dated nearest on or before `when`."""
        if source == target:
            return amount
        direct = self.rates.get((source, target))
        if direct:
            return (amount * self._rate_on(direct, when)).quantize(Decimal("0.01"))
        inverse = self.rates.get((target, source))
        if inverse:
            return (amount / self._rate_on(inverse, when)).quantize(Decimal("0.01"))
        raise KeyError(f"no rate for {source}->{target}")

    @staticmethod
    def _rate_on(series: list[tuple[date, Decimal]], when: date) -> Decimal:
        best = series[0][1]
        for rate_date, rate in series:
            if rate_date <= when:
                best = rate
            else:
                break
        return best


def _split(value: str) -> list[str]:
    return [item for item in value.split("|") if item]


def _monthly_groups(events: Iterable[Event]) -> dict[tuple[str, str], list[Event]]:
    groups: dict[tuple[str, str], list[Event]] = defaultdict(list)
    for event in events:
        groups[(event.description, event.category)].append(event)
    return groups


def _advance_month(start: date, months: int) -> date:
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    day = min(start.day, [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
                          else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return date(year, month, day)


class StateBuilder:
    """Builds one `DecisionInput` for a request using structured data only."""

    def __init__(self, request_id: str, dataset: Optional[Dataset] = None) -> None:
        self.data = dataset or Dataset.load()
        self.request_row = self.data.requests[request_id]
        self.user_id = self.request_row["user_id"]
        self.profile = self.data.profiles[self.user_id]
        self.currency = self.profile["home_currency"]
        self.as_of = date.fromisoformat(self.request_row["request_date"])
        self.end = self.as_of + timedelta(days=HORIZON_DAYS)
        self.protected = set(_split(self.profile["expense_categories_to_protect"]))
        self.reducible = set(_split(self.profile["expense_categories_user_is_willing_to_reduce"]))
        self.stoppable = set(_split(self.profile["expense_categories_user_is_willing_to_stop"]))
        self.events = self.data.events_by_user[self.user_id]

        self._flows: list[dict] = []
        self._series: list[dict] = []
        self._excluded: list[dict] = []
        self._counter = 0

    # -- helpers ---------------------------------------------------------

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:03d}"

    def _home_amount(self, event: Event, when: date) -> Decimal:
        return self.data.convert(event.amount, event.currency, self.currency, when)

    def _add_flow(self, day: date, direction: str, amount: Decimal, category: str,
                  origin: str, certainty: str, description: str, event: Optional[Event] = None,
                  series_id: Optional[str] = None) -> str:
        occurrence_id = self._next_id("occ")
        sources = [{"kind": "financial_event", "ref_id": event.event_id, "detail": None}] if event \
            else [{"kind": "profile", "ref_id": self.user_id, "detail": "forecast from history"}]
        self._flows.append({
            "occurrence_id": occurrence_id,
            "date": day.isoformat(),
            "direction": direction,
            "amount": str(amount),
            "category": category,
            "origin": origin,
            "certainty": certainty,
            "is_protected": category in self.protected,
            "description": description[:200],
            "adjustable_series_id": series_id,
            "source_event_id": event.event_id if event else None,
            "original_currency": None,
            "original_amount": None,
            "exchange_rate_date": None,
            "sources": sources,
        })
        return occurrence_id

    def _exclude(self, event: Event, reason: str, explanation: str) -> None:
        self._excluded.append({
            "source": {"kind": "financial_event", "ref_id": event.event_id, "detail": None},
            "reason": reason,
            "explanation": explanation,
        })

    # -- construction ----------------------------------------------------

    def _add_dated_commitments(self) -> None:
        """Pending debits and scheduled obligations already on the books."""
        for event in self.events:
            when = event.settlement_date or event.event_date
            if event.status in CASH_IGNORED_STATUSES or event.event_type in NON_CASH_EVENT_TYPES:
                if when >= self.as_of:
                    self._exclude(event, {
                        "cancelled": "cancelled",
                        "failed": "failed_no_retry",
                        "unrealized": "unrealized_valuation",
                    }.get(event.status, "not_relevant"), f"status={event.status}")
                continue
            if not (self.as_of <= when <= self.end):
                continue
            if event.amount is None:
                # Resolving this needs the linked image, which is the normalizer's job.
                continue
            if event.status not in {"pending", "scheduled"}:
                continue

            amount = self._home_amount(event, when)
            if event.is_credit:
                if event.status == "pending":
                    self._exclude(event, "pending_credit", "credit has not settled yet")
                    continue
                self._add_flow(when, "inflow", amount, event.category,
                               "confirmed_income", "confirmed", event.description, event)
            else:
                origin = "reserved_pending_debit" if event.status == "pending" else "scheduled_debit"
                self._add_flow(when, "outflow", amount, event.category,
                               origin, "confirmed", event.description, event)

    def _recurring_history(self) -> dict[tuple[str, str], list[Event]]:
        """Settled events before the request date, grouped into candidate series."""
        history = [
            event for event in self.events
            if event.status == "settled"
            and event.amount is not None
            and (event.settlement_date or event.event_date) < self.as_of
        ]
        return _monthly_groups(history)

    def _add_variable_budget(self) -> None:
        """Forecast everyday spending as a per-category run rate.

        Grouping this spending by description would invent half a dozen
        parallel grocery "subscriptions" and double-count the category several
        times over. A daily rate per category is both simpler and closer to how
        the money actually leaves the account.
        """
        window_start = self.as_of - timedelta(days=90)
        by_category: dict[str, list[Event]] = defaultdict(list)
        for event in self.events:
            when = event.settlement_date or event.event_date
            if (event.status == "settled" and not event.is_credit and event.amount is not None
                    and event.category in VARIABLE_ESSENTIAL
                    and window_start <= when < self.as_of):
                by_category[event.category].append(event)

        for category, group in by_category.items():
            observed = min((e.settlement_date or e.event_date) for e in group)
            days = max((self.as_of - observed).days, 1)
            total = sum(
                (self.data.convert(e.amount, e.currency, self.currency,
                                   e.settlement_date or e.event_date) for e in group),
                Decimal(0),
            )
            weekly = (total / days * 7).quantize(Decimal("0.01"))
            if weekly <= 0:
                continue
            day = self.as_of + timedelta(days=7)
            while day <= self.end:
                self._add_flow(day, "outflow", weekly, category,
                               "variable_essential_forecast", "forecast",
                               f"Forecast weekly {category} spending")
                day += timedelta(days=7)

    def _add_salary(self) -> None:
        """Project salary monthly from the most recent payroll record.

        Salary is the one series where a single record is enough: an employed
        user is paid again next month. Waiting for three observations would
        make anyone who recently changed jobs look like they have no income.
        """
        salaries = [
            event for event in self.events
            if event.category == "salary" and event.is_credit and event.amount is not None
            and event.status in {"settled", "scheduled"}
        ]
        if not salaries:
            return
        salaries.sort(key=lambda e: e.settlement_date or e.event_date)
        latest = salaries[-1]
        anchor = latest.settlement_date or latest.event_date
        # A prorated first payslip understates the real monthly figure.
        amount = max(e.amount for e in salaries if e.currency == latest.currency)

        step = 0
        while True:
            step += 1
            day = _advance_month(anchor, step)
            if day > self.end:
                break
            if day < self.as_of:
                continue
            self._add_flow(day, "inflow",
                           self.data.convert(amount, latest.currency, self.currency, day),
                           "salary", "recurring_income_forecast", "forecast",
                           "Forecast monthly salary", latest)

    def _add_recurring(self) -> None:
        for (description, category), group in self._recurring_history().items():
            if len(group) < 3 or category in VARIABLE_ESSENTIAL or category == "salary":
                continue
            group.sort(key=lambda e: e.settlement_date or e.event_date)
            dates = [e.settlement_date or e.event_date for e in group]
            gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
            typical_gap = int(median(gaps))
            if typical_gap <= 0:
                continue

            last = group[-1]
            is_monthly = 26 <= typical_gap <= 32
            # A commitment repeats at its most recent amount.
            amount = last.amount
            origin = ("recurring_income_forecast" if last.is_credit
                      else "recurring_expense_forecast")
            certainty = "forecast"

            series_id = None
            if not last.is_credit and last.flexibility != "fixed":
                series_id = self._register_series(last, amount)

            occurrence_ids: list[str] = []
            cursor = dates[-1]
            step = 0
            while True:
                step += 1
                cursor = (_advance_month(dates[-1], step) if is_monthly
                          else dates[-1] + timedelta(days=typical_gap * step))
                if cursor > self.end:
                    break
                if cursor < self.as_of:
                    continue
                converted = self.data.convert(amount, last.currency, self.currency, cursor)
                occurrence_ids.append(self._add_flow(
                    cursor,
                    "inflow" if last.is_credit else "outflow",
                    converted,
                    category,
                    origin,
                    certainty,
                    description,
                    last,
                    series_id,
                ))

            if series_id is not None:
                if occurrence_ids:
                    self._attach_occurrences(series_id, occurrence_ids)
                else:
                    self._series = [s for s in self._series if s["series_id"] != series_id]

    def _register_series(self, event: Event, amount: Decimal) -> Optional[str]:
        """Register a flexible expense the user has agreed may change."""
        category = event.category
        if category in self.protected:
            return None
        may_stop = category in self.stoppable
        may_reduce = category in self.reducible and event.minimum_allowed_amount is not None
        if event.flexibility == "stoppable":
            may_reduce = False
        if event.flexibility == "reducible":
            may_stop = False

        if may_stop and may_reduce:
            action = "stop_or_reduce"
        elif may_stop:
            action = "stop"
        elif may_reduce:
            action = "reduce"
        else:
            return None

        floor = event.minimum_allowed_amount if action != "stop" else None
        if floor is not None and floor >= amount:
            return None

        series_id = self._next_id("series")
        self._series.append({
            "series_id": series_id,
            "target_event_id": event.event_id,
            "category": category,
            "action": action,
            "current_amount": str(amount),
            "minimum_allowed_amount": str(floor) if floor is not None else None,
            "occurrence_ids": [],
            "sources": [{"kind": "financial_event", "ref_id": event.event_id, "detail": None}],
        })
        return series_id

    def _attach_occurrences(self, series_id: str, occurrence_ids: list[str]) -> None:
        for series in self._series:
            if series["series_id"] == series_id:
                series["occurrence_ids"] = occurrence_ids
                return

    def _payment_options(self) -> list[dict]:
        built = []
        for row in sorted(self.data.options_by_request[self.request_row["request_id"]],
                          key=lambda r: r["payment_option_id"]):
            count = int(row["number_of_payments"])
            freq = int(row["payment_frequency_days"]) if row["payment_frequency_days"] else None
            start = date.fromisoformat(row["first_payment_date"])
            built.append({
                "payment_option_id": row["payment_option_id"],
                "method": row["payment_method"],
                "payment_amount": row["payment_amount"],
                "number_of_payments": count,
                "first_payment_date": row["first_payment_date"],
                "payment_frequency_days": freq,
                "financing_fee": row["financing_fee"],
                "total_payable_amount": row["total_payable_amount"],
                "schedule": [
                    {"date": (start + timedelta(days=(freq or 0) * i)).isoformat(),
                     "amount": row["payment_amount"]}
                    for i in range(count)
                ],
            })
        return built

    def build(self) -> DecisionInput:
        self._add_dated_commitments()
        self._add_recurring()
        self._add_salary()
        self._add_variable_budget()

        max_months = self.profile["max_installment_months"]
        document = {
            "schema_version": "1.0.0",
            "financial_state": {
                "schema_version": "1.0.0",
                "user_id": self.user_id,
                "home_currency": self.currency,
                "as_of_date": self.as_of.isoformat(),
                "forecast_end_date": self.end.isoformat(),
                "opening_balance": self.profile["current_available_balance"],
                "minimum_balance_to_keep": self.profile["minimum_balance_to_keep"],
                "preferences": {
                    "financial_priorities": _split(self.profile["financial_priorities"]),
                    "protected_categories": sorted(self.protected),
                    "reducible_categories": sorted(self.reducible),
                    "stoppable_categories": sorted(self.stoppable),
                    "accepted_payment_methods": _split(
                        self.profile["payment_methods_user_will_consider"]),
                    "max_installment_months": int(max_months) if max_months else None,
                },
                "cash_flows": self._flows,
                "adjustable_series": self._series,
                "evidence_resolutions": [],
                "excluded_evidence": self._excluded,
                "assumptions": [
                    "Structured events only; messages and images are not interpreted.",
                ],
                "attestation": {
                    "model_provider": "none",
                    "model_name": "deterministic-test-harness",
                    "generated_at": "2026-09-13T00:00:00+00:00",
                    "attested": True,
                    "notes": "Test harness state, not a model-signed normalization.",
                },
            },
            "request": {
                "schema_version": "1.0.0",
                "request_id": self.request_row["request_id"],
                "user_id": self.user_id,
                "request_date": self.request_row["request_date"],
                "request_type": self.request_row["request_type"],
                "requested_amount": self.request_row["requested_amount"],
                "currency": self.currency,
                "desired_completion_date": self.request_row["desired_completion_date"],
                "allows_partial_payment": self.request_row["allows_partial_payment"] == "true",
                "request_text": self.request_row["request_text"],
                "payment_options": self._payment_options(),
            },
        }
        return DecisionInput.model_validate(document)


def build_state(request_id: str) -> DecisionInput:
    return StateBuilder(request_id).build()
