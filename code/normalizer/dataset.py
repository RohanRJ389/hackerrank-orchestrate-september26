"""Typed, request-scoped access to the participant-facing dataset."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "dataset"


def _date(value: str) -> date | None:
    return date.fromisoformat(value) if value else None


def _decimal(value: str) -> Decimal | None:
    return Decimal(value) if value else None


@dataclass(frozen=True)
class Event:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Decimal | None
    currency: str
    event_date: date
    settlement_date: date | None
    status: str
    linked_event_id: str
    flexibility: str
    minimum_allowed_amount: Decimal | None

    @property
    def cash_date(self) -> date:
        return self.settlement_date or self.event_date

    @property
    def is_credit(self) -> bool:
        return self.direction == "credit"

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("event_date", "settlement_date"):
            value[key] = value[key].isoformat() if value[key] else None
        for key in ("amount", "minimum_allowed_amount"):
            value[key] = format(value[key], "f") if value[key] is not None else None
        return value


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: str
    related_event_id: str
    sent_at: str
    source_type: str
    message_text: str

    def serializable(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str
    path: Path

    def serializable(self) -> dict[str, str]:
        return {
            "image_id": self.image_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "related_event_id": self.related_event_id,
            "filename": self.path.name,
        }


@dataclass(frozen=True)
class Rate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: Decimal

    @property
    def ref_id(self) -> str:
        return f"{self.rate_date.isoformat()}:{self.from_currency}:{self.to_currency}"


@dataclass(frozen=True)
class RequestBundle:
    profile: dict[str, str]
    request: dict[str, str]
    payment_options: tuple[dict[str, str], ...]
    events: tuple[Event, ...]
    messages: tuple[Message, ...]
    images: tuple[ImageRef, ...]

    @property
    def request_id(self) -> str:
        return self.request["request_id"]

    @property
    def user_id(self) -> str:
        return self.request["user_id"]

    @property
    def has_unstructured_evidence(self) -> bool:
        return bool(self.messages or self.images)


class Dataset:
    """Loads CSVs once and returns isolated bundles for one request."""

    def __init__(self, root: Path = DEFAULT_DATASET) -> None:
        self.root = root.resolve()
        self.profiles = {r["user_id"]: r for r in self._read("financial_profiles.csv")}
        self.requests = {r["request_id"]: r for r in self._read("requests.csv")}
        self.requests.update({r["request_id"]: r for r in self._read("sample_requests.csv")})

        self.events_by_user: dict[str, list[Event]] = defaultdict(list)
        self.events_by_id: dict[str, Event] = {}
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
                event_date=_date(row["event_date"]),  # type: ignore[arg-type]
                settlement_date=_date(row["settlement_date"]),
                status=row["status"],
                linked_event_id=row["linked_event_id"],
                flexibility=row["flexibility"],
                minimum_allowed_amount=_decimal(row["minimum_allowed_amount"]),
            )
            self.events_by_user[event.user_id].append(event)
            self.events_by_id[event.event_id] = event

        self.options_by_request: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self._read("request_payment_options.csv"):
            self.options_by_request[row["request_id"]].append(row)

        self.messages_by_user: dict[str, list[Message]] = defaultdict(list)
        for row in self._read("messages.csv"):
            self.messages_by_user[row["user_id"]].append(Message(**row))

        self.images_by_user: dict[str, list[ImageRef]] = defaultdict(list)
        for row in self._read("images.csv"):
            image = ImageRef(
                **row,
                path=(self.root / "media" / "images" / f"{row['image_id']}.png").resolve(),
            )
            self.images_by_user[row["user_id"]].append(image)

        self.rates: dict[tuple[str, str], list[Rate]] = defaultdict(list)
        for row in self._read("exchange_rates.csv"):
            rate = Rate(
                rate_date=date.fromisoformat(row["rate_date"]),
                from_currency=row["from_currency"],
                to_currency=row["to_currency"],
                rate=Decimal(row["rate"]),
            )
            self.rates[(rate.from_currency, rate.to_currency)].append(rate)
        for values in self.rates.values():
            values.sort(key=lambda item: item.rate_date)

    def _read(self, name: str) -> list[dict[str, str]]:
        with (self.root / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def bundle(self, request_id: str) -> RequestBundle:
        request = self.requests[request_id]
        user_id = request["user_id"]
        return RequestBundle(
            profile=self.profiles[user_id],
            request=request,
            payment_options=tuple(
                sorted(
                    self.options_by_request[request_id],
                    key=lambda row: row["payment_option_id"],
                )
            ),
            events=tuple(self.events_by_user[user_id]),
            messages=tuple(self.messages_by_user[user_id]),
            images=tuple(self.images_by_user[user_id]),
        )

    def rate_on(self, source: str, target: str, when: date) -> Rate | None:
        if source == target:
            return None
        direct = self.rates.get((source, target))
        if direct:
            eligible = [row for row in direct if row.rate_date <= when]
            return eligible[-1] if eligible else direct[0]
        inverse = self.rates.get((target, source))
        if inverse:
            eligible = [row for row in inverse if row.rate_date <= when]
            selected = eligible[-1] if eligible else inverse[0]
            return Rate(
                selected.rate_date,
                source,
                target,
                Decimal(1) / selected.rate,
            )
        raise KeyError(f"no exchange rate for {source}->{target} on {when}")

    def convert(
        self, amount: Decimal, source: str, target: str, when: date
    ) -> tuple[Decimal, Rate | None]:
        rate = self.rate_on(source, target, when)
        if rate is None:
            return amount, None
        return (amount * rate.rate).quantize(Decimal("0.01")), rate
