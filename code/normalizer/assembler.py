"""Deterministically assemble, validate, and hash a normalized state."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from contracts import DecisionInput, FinancialState, assert_valid
from pydantic import ValidationError

from .dataset import Dataset, Event, RequestBundle
from .models import (
    CandidateDecision,
    DraftResult,
    NormalizationDirectives,
    NormalizationPacket,
)


def advance_month(start: date, months: int = 1) -> date:
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    lengths = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return date(year, month, min(start.day, lengths[month - 1]))


def _split(value: str) -> list[str]:
    return [item for item in value.split("|") if item]


def _source(ref_id: str, detail: str | None = None) -> dict[str, Any]:
    if ref_id.startswith("event_"):
        kind = "financial_event"
    elif ref_id.startswith("message_"):
        kind = "message"
    elif ref_id.startswith("image_"):
        kind = "image"
    elif ref_id.startswith("request_"):
        kind = "request"
    elif ref_id.startswith("user_"):
        kind = "profile"
    else:
        kind = "exchange_rate"
    return {"kind": kind, "ref_id": ref_id, "detail": detail}


class StateAssembler:
    def __init__(
        self,
        dataset: Dataset,
        packet: NormalizationPacket,
        directives: NormalizationDirectives | None = None,
    ) -> None:
        self.dataset = dataset
        self.packet = packet
        self.bundle = dataset.bundle(packet.request_id)
        self.directives = directives or NormalizationDirectives(request_id=packet.request_id)
        if self.directives.request_id != packet.request_id:
            raise ValueError("directives target a different request")
        candidate_ids = {item.candidate_id for item in packet.recurrence_candidates}
        unknown_candidates = {
            item.candidate_id for item in self.directives.candidate_decisions
        } - candidate_ids
        if unknown_candidates:
            raise ValueError(f"unknown recurrence candidates: {sorted(unknown_candidates)}")
        event_ids = {event.event_id for event in self.bundle.events}
        unknown_events = {
            item.event_id
            for item in self.directives.event_decisions
            if item.action != "add"
        } - event_ids
        if unknown_events:
            raise ValueError(f"unknown financial events: {sorted(unknown_events)}")
        variable_categories = {item.category for item in packet.variable_budgets}
        unknown_categories = {
            item.category for item in self.directives.variable_budget_decisions
        } - variable_categories
        if unknown_categories:
            raise ValueError(f"unknown variable categories: {sorted(unknown_categories)}")
        self.as_of = date.fromisoformat(self.bundle.request["request_date"])
        self.end = self.as_of + timedelta(days=90)
        self.home = self.bundle.profile["home_currency"]
        self.protected = set(_split(self.bundle.profile["expense_categories_to_protect"]))
        self.reducible = set(
            _split(self.bundle.profile["expense_categories_user_is_willing_to_reduce"])
        )
        self.stoppable = set(
            _split(self.bundle.profile["expense_categories_user_is_willing_to_stop"])
        )
        self.flows: list[dict[str, Any]] = []
        self.series: list[dict[str, Any]] = []
        self.excluded: list[dict[str, Any]] = []
        self.resolutions: list[dict[str, Any]] = []
        self.counter = 0

    def _id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:04d}"

    def _converted(
        self, amount: Decimal, currency: str, when: date
    ) -> tuple[Decimal, dict[str, Any]]:
        converted, rate = self.dataset.convert(amount, currency, self.home, when)
        fields: dict[str, Any] = {
            "original_currency": None,
            "original_amount": None,
            "exchange_rate_date": None,
        }
        if rate is not None:
            fields = {
                "original_currency": currency,
                "original_amount": format(amount, "f"),
                "exchange_rate_date": rate.rate_date.isoformat(),
                "rate_source": _source(rate.ref_id, f"rate={format(rate.rate, 'f')}"),
            }
        return converted, fields

    def _add_flow(
        self,
        *,
        when: date,
        direction: str,
        amount: Decimal,
        currency: str,
        category: str,
        origin: str,
        certainty: str,
        description: str,
        source_refs: tuple[str, ...],
        source_event_id: str | None = None,
        series_id: str | None = None,
    ) -> str:
        converted, conversion = self._converted(amount, currency, when)
        sources = [_source(ref) for ref in source_refs]
        rate_source = conversion.pop("rate_source", None)
        if rate_source:
            sources.append(rate_source)
        occurrence_id = self._id("occ")
        self.flows.append(
            {
                "occurrence_id": occurrence_id,
                "date": when.isoformat(),
                "direction": "inflow" if direction == "credit" else "outflow",
                "amount": format(converted, "f"),
                "category": category,
                "origin": origin,
                "certainty": certainty,
                "is_protected": category in self.protected,
                "description": description[:200],
                "adjustable_series_id": series_id,
                "source_event_id": source_event_id,
                **conversion,
                "sources": sources or [_source(self.packet.user_id)],
            }
        )
        return occurrence_id

    def _exclude(self, event: Event, reason: str, explanation: str) -> None:
        self.excluded.append(
            {
                "source": _source(event.event_id),
                "reason": reason,
                "explanation": explanation[:500],
            }
        )

    def _event_decision(self, event_id: str):
        return next(
            (item for item in self.directives.event_decisions if item.event_id == event_id),
            None,
        )

    def _add_future_events(self) -> None:
        for event in self.bundle.events:
            decision = self._event_decision(event.event_id)
            if decision and decision.action == "exclude":
                self._exclude(
                    event,
                    decision.exclusion_reason or "superseded_by_amendment",
                    decision.rationale or "Excluded by model review.",
                )
                continue

            if event.status == "pending" and event.direction == "credit":
                self._exclude(event, "pending_credit", "Credit has not settled.")
                continue
            if event.status == "unrealized":
                self._exclude(event, "unrealized_valuation", "No cash proceeds exist.")
                continue
            if event.status == "cancelled":
                self._exclude(event, "cancelled", "Cancelled record is not projected.")
                continue
            if event.status == "failed":
                self._exclude(event, "failed_no_retry", "Failed attempt itself is not cash.")
                continue
            if event.status not in {"pending", "scheduled"}:
                continue

            when = decision.date if decision and decision.date else event.cash_date
            if not self.as_of <= when <= self.end:
                self._exclude(event, "outside_forecast_window", "Falls outside the 90-day window.")
                continue
            amount = decision.amount if decision and decision.amount else event.amount
            if amount is None:
                self._exclude(event, "not_relevant", "Amount is unresolved without image evidence.")
                continue
            direction = decision.direction if decision and decision.direction else event.direction
            category = decision.category if decision and decision.category else event.category
            if direction == "credit":
                origin = "confirmed_income"
            else:
                origin = (
                    "reserved_pending_debit"
                    if event.status == "pending"
                    else "scheduled_debit"
                )
            refs = (event.event_id,) + (
                tuple(decision.source_refs) if decision else ()
            )
            occurrence_id = self._add_flow(
                when=when,
                direction=direction,
                amount=amount,
                currency=event.currency,
                category=category,
                origin=decision.origin if decision and decision.origin else origin,
                certainty="confirmed",
                description=event.description,
                source_refs=refs,
                source_event_id=event.event_id,
            )
            if decision and decision.action == "amend":
                self.resolutions.append(
                    {
                        "resolution_id": self._id("resolution"),
                        "summary": decision.rationale or f"Amended {event.event_id}.",
                        "basis": "explicit_amendment",
                        "sources": [_source(ref) for ref in refs],
                        "affected_occurrence_ids": [occurrence_id],
                        "is_conservative_fallback": False,
                    }
                )

        for decision in self.directives.event_decisions:
            if decision.action != "add":
                continue
            if not all((decision.amount, decision.date, decision.direction, decision.category)):
                raise ValueError(f"added event {decision.event_id} is incomplete")
            self._add_flow(
                when=decision.date,
                direction=decision.direction,
                amount=decision.amount,
                currency=self.home,
                category=decision.category,
                origin=decision.origin
                or ("confirmed_income" if decision.direction == "credit" else "committed_one_time_debit"),
                certainty="confirmed",
                description=decision.rationale or "Model-confirmed financial fact",
                source_refs=decision.source_refs or (decision.event_id,),
            )

    def _candidate_decision(self, candidate_id: str) -> CandidateDecision:
        return next(
            (
                item
                for item in self.directives.candidate_decisions
                if item.candidate_id == candidate_id
            ),
            CandidateDecision(candidate_id=candidate_id),
        )

    def _add_recurrences(self) -> None:
        for candidate in self.packet.recurrence_candidates:
            decision = self._candidate_decision(candidate.candidate_id)
            source_refs = candidate.source_event_ids + decision.source_refs
            if decision.action == "reject":
                if decision.source_refs:
                    self.resolutions.append(
                        {
                            "resolution_id": self._id("resolution"),
                            "summary": decision.rationale
                            or f"Rejected recurrence {candidate.candidate_id}.",
                            "basis": "explicit_amendment",
                            "sources": [_source(ref) for ref in source_refs],
                            "affected_occurrence_ids": [],
                            "is_conservative_fallback": False,
                        }
                    )
                continue
            amount = decision.amount or candidate.amount
            anchor = decision.anchor_date or candidate.anchor_date
            series_id: str | None = None
            occurrence_ids: list[str] = []
            if candidate.adjustable:
                series_id = self._id("series")

            step = 0
            while True:
                step += 1
                when = (
                    advance_month(anchor, step)
                    if candidate.monthly
                    else anchor + timedelta(days=(candidate.interval_days or 7) * step)
                )
                if when > self.end or (decision.end_date and when > decision.end_date):
                    break
                if when < self.as_of:
                    continue
                occurrence_ids.append(
                    self._add_flow(
                        when=when,
                        direction=candidate.direction,
                        amount=amount,
                        currency=candidate.currency,
                        category=candidate.category,
                        origin=(
                            "recurring_income_forecast"
                            if candidate.direction == "credit"
                            else "recurring_expense_forecast"
                        ),
                        certainty="forecast",
                        description=candidate.description,
                        source_refs=source_refs,
                        source_event_id=candidate.source_event_ids[-1],
                        series_id=series_id,
                    )
                )
            if series_id and occurrence_ids:
                may_stop = candidate.category in self.stoppable and candidate.flexibility in {
                    "stoppable",
                    "reducible_or_stoppable",
                }
                may_reduce = (
                    candidate.category in self.reducible
                    and candidate.minimum_allowed_amount is not None
                    and candidate.flexibility in {"reducible", "reducible_or_stoppable"}
                )
                action = "stop_or_reduce" if may_stop and may_reduce else "stop" if may_stop else "reduce"
                current, _ = self.dataset.convert(
                    amount, candidate.currency, self.home, occurrence_date := date.fromisoformat(
                        next(flow["date"] for flow in self.flows if flow["occurrence_id"] == occurrence_ids[0])
                    )
                )
                floor = None
                if may_reduce:
                    floor, _ = self.dataset.convert(
                        candidate.minimum_allowed_amount,
                        candidate.currency,
                        self.home,
                        occurrence_date,
                    )
                self.series.append(
                    {
                        "series_id": series_id,
                        "target_event_id": candidate.source_event_ids[-1],
                        "category": candidate.category,
                        "action": action,
                        "current_amount": format(current, "f"),
                        "minimum_allowed_amount": format(floor, "f") if floor is not None else None,
                        "occurrence_ids": occurrence_ids,
                        "sources": [_source(ref) for ref in source_refs],
                    }
                )
            if decision.action == "amend" and decision.source_refs:
                self.resolutions.append(
                    {
                        "resolution_id": self._id("resolution"),
                        "summary": decision.rationale
                        or f"Amended recurrence {candidate.candidate_id}.",
                        "basis": "explicit_amendment",
                        "sources": [_source(ref) for ref in source_refs],
                        "affected_occurrence_ids": occurrence_ids,
                        "is_conservative_fallback": False,
                    }
                )

    def _add_variable_budgets(self) -> None:
        overrides = {
            item.category: item for item in self.directives.variable_budget_decisions
        }
        for budget in self.packet.variable_budgets:
            override = overrides.get(budget.category)
            weekly = override.weekly_amount if override else budget.weekly_amount
            if weekly <= 0:
                continue
            refs = budget.source_event_ids + (override.source_refs if override else ())
            when = self.as_of + timedelta(days=7)
            while when <= self.end:
                self._add_flow(
                    when=when,
                    direction="debit",
                    amount=weekly,
                    currency=self.home,
                    category=budget.category,
                    origin="variable_essential_forecast",
                    certainty="forecast",
                    description=f"Forecast weekly {budget.category} spending",
                    source_refs=refs,
                )
                when += timedelta(days=7)

    def _attestation(self, model: str) -> dict[str, Any]:
        return {
            "model_provider": "anthropic" if model != "deterministic" else "deterministic",
            "model_name": model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "attested": True,
            "notes": "assembled_from_model_directives",
        }

    def _enriched_request(self) -> dict[str, Any]:
        request = self.bundle.request
        options = []
        for row in self.bundle.payment_options:
            start = date.fromisoformat(row["first_payment_date"])
            count = int(row["number_of_payments"])
            frequency = int(row["payment_frequency_days"]) if row["payment_frequency_days"] else None
            options.append(
                {
                    "payment_option_id": row["payment_option_id"],
                    "method": row["payment_method"],
                    "payment_amount": row["payment_amount"],
                    "number_of_payments": count,
                    "first_payment_date": start.isoformat(),
                    "payment_frequency_days": frequency,
                    "financing_fee": row["financing_fee"],
                    "total_payable_amount": row["total_payable_amount"],
                    "schedule": [
                        {
                            "date": (
                                start + timedelta(days=(frequency or 0) * index)
                            ).isoformat(),
                            "amount": row["payment_amount"],
                        }
                        for index in range(count)
                    ],
                }
            )
        return {
            "schema_version": "1.0.0",
            "request_id": request["request_id"],
            "user_id": request["user_id"],
            "request_date": request["request_date"],
            "request_type": request["request_type"],
            "requested_amount": request["requested_amount"],
            "currency": self.home,
            "desired_completion_date": request["desired_completion_date"],
            "allows_partial_payment": request["allows_partial_payment"] == "true",
            "request_text": request["request_text"],
            "payment_options": options,
        }

    def document(self, model: str = "pending") -> dict[str, Any]:
        self._add_future_events()
        self._add_recurrences()
        self._add_variable_budgets()
        for ref in self.directives.untrusted_instruction_sources:
            self.excluded.append(
                {
                    "source": _source(ref),
                    "reason": "untrusted_instruction",
                    "explanation": "Embedded instruction was ignored; financial facts were reviewed separately.",
                }
            )
        profile = self.bundle.profile
        max_months = profile["max_installment_months"]
        return {
            "schema_version": "1.0.0",
            "financial_state": {
                "schema_version": "1.0.0",
                "user_id": self.bundle.user_id,
                "home_currency": self.home,
                "as_of_date": self.as_of.isoformat(),
                "forecast_end_date": self.end.isoformat(),
                "opening_balance": profile["current_available_balance"],
                "minimum_balance_to_keep": profile["minimum_balance_to_keep"],
                "preferences": {
                    "financial_priorities": _split(profile["financial_priorities"]),
                    "protected_categories": _split(profile["expense_categories_to_protect"]),
                    "reducible_categories": _split(
                        profile["expense_categories_user_is_willing_to_reduce"]
                    ),
                    "stoppable_categories": _split(
                        profile["expense_categories_user_is_willing_to_stop"]
                    ),
                    "accepted_payment_methods": _split(
                        profile["payment_methods_user_will_consider"]
                    ),
                    "max_installment_months": int(max_months) if max_months else None,
                },
                "cash_flows": sorted(self.flows, key=lambda item: (item["date"], item["occurrence_id"])),
                "adjustable_series": self.series,
                "evidence_resolutions": self.resolutions,
                "excluded_evidence": self.excluded,
                "assumptions": list(self.directives.assumptions)
                or ["Deterministic baseline reviewed against supplied evidence."],
                "attestation": self._attestation(model),
            },
            "request": self._enriched_request(),
        }


def canonical_unsigned_state(state: FinancialState | dict[str, Any]) -> bytes:
    data = (
        state.model_dump(mode="json", exclude={"attestation"})
        if isinstance(state, FinancialState)
        else {key: value for key, value in state.items() if key != "attestation"}
    )
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def state_hash(state: FinancialState | dict[str, Any]) -> str:
    return hashlib.sha256(canonical_unsigned_state(state)).hexdigest()


def build_decision_input(
    dataset: Dataset,
    packet: NormalizationPacket,
    directives: NormalizationDirectives | None = None,
    *,
    model: str = "pending",
) -> tuple[DecisionInput, list[str]]:
    document = StateAssembler(dataset, packet, directives).document(model=model)
    decision_input = DecisionInput.model_validate(document)
    warnings = [str(item) for item in assert_valid(decision_input)]
    return decision_input, warnings


def try_build(
    dataset: Dataset,
    packet: NormalizationPacket,
    directives: NormalizationDirectives | None = None,
) -> DraftResult:
    try:
        decision_input, warnings = build_decision_input(dataset, packet, directives)
        return DraftResult(
            request_id=packet.request_id,
            valid=True,
            state_hash=state_hash(decision_input.financial_state),
            warnings=tuple(warnings),
        )
    except (ValidationError, ValueError, KeyError) as error:
        return DraftResult(
            request_id=packet.request_id,
            valid=False,
            errors=(str(error),),
        )
