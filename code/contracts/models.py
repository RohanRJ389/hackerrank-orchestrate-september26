"""Executable boundary contract between the normalizer and the decision engine.

The normalizer owns interpretation of raw, partially unstructured evidence. The
decision engine consumes only :class:`DecisionInput` and performs no inference
about recurrence, currency conversion, or evidence conflicts.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1.0.0"
FORECAST_HORIZON_DAYS = 90

# Monetary values are exchanged as strings so that no consumer can silently
# reintroduce binary floating point error.
Money = Annotated[
    Decimal,
    Field(ge=0, allow_inf_nan=False),
    PlainSerializer(lambda value: format(value, "f"), return_type=str),
]
PositiveMoney = Annotated[
    Decimal,
    Field(gt=0, allow_inf_nan=False),
    PlainSerializer(lambda value: format(value, "f"), return_type=str),
]

UserId = Annotated[str, Field(pattern=r"^user_\d+$")]
RequestId = Annotated[str, Field(pattern=r"^request_\d+$")]
EventId = Annotated[str, Field(pattern=r"^event_\d+$")]
Identifier = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^\S+$")]


class Currency(str, Enum):
    INR = "INR"
    ZAR = "ZAR"
    IDR = "IDR"
    USD = "USD"
    EUR = "EUR"


class RequestType(str, Enum):
    PURCHASE = "purchase"
    TRAVEL = "travel"
    EDUCATION = "education"
    FAMILY_TRANSFER = "family_transfer"
    DEBT_REPAYMENT = "debt_repayment"
    INVESTMENT = "investment"
    HOUSING = "housing"
    EMERGENCY_EXPENSE = "emergency_expense"
    OTHER = "other"


class PaymentMethod(str, Enum):
    """Payment methods a user may accept and a seller may offer."""

    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"


class OfferedPaymentMethod(str, Enum):
    """Methods that appear in `request_payment_options.csv`."""

    FULL_PAYMENT = "full_payment"
    INSTALLMENTS = "installments"


class Direction(str, Enum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"


class CashFlowOrigin(str, Enum):
    """Why a dated cash movement belongs on the forward timeline.

    The enum deliberately has no member for pending credits, unrealized
    valuations, cancelled records, or failed debits with no retry: those can
    never become projected cash and must be reported as excluded evidence.
    """

    CONFIRMED_INCOME = "confirmed_income"
    RECURRING_INCOME_FORECAST = "recurring_income_forecast"
    RESERVED_PENDING_DEBIT = "reserved_pending_debit"
    SCHEDULED_DEBIT = "scheduled_debit"
    RECURRING_EXPENSE_FORECAST = "recurring_expense_forecast"
    VARIABLE_ESSENTIAL_FORECAST = "variable_essential_forecast"
    COMMITTED_ONE_TIME_DEBIT = "committed_one_time_debit"


INFLOW_ORIGINS = frozenset(
    {CashFlowOrigin.CONFIRMED_INCOME, CashFlowOrigin.RECURRING_INCOME_FORECAST}
)
OUTFLOW_ORIGINS = frozenset(CashFlowOrigin) - INFLOW_ORIGINS
RECURRING_ORIGINS = frozenset(
    {
        CashFlowOrigin.RECURRING_INCOME_FORECAST,
        CashFlowOrigin.RECURRING_EXPENSE_FORECAST,
        CashFlowOrigin.VARIABLE_ESSENTIAL_FORECAST,
    }
)


class Certainty(str, Enum):
    """How firm the dated amount is.

    `CONFIRMED` covers settled, scheduled, and explicitly confirmed records.
    `FORECAST` covers amounts the normalizer projected from history.
    """

    CONFIRMED = "confirmed"
    FORECAST = "forecast"


class AdjustmentAction(str, Enum):
    STOP = "stop"
    REDUCE = "reduce"
    STOP_OR_REDUCE = "stop_or_reduce"


class EvidenceKind(str, Enum):
    FINANCIAL_EVENT = "financial_event"
    MESSAGE = "message"
    IMAGE = "image"
    PROFILE = "profile"
    REQUEST = "request"
    PAYMENT_OPTION = "payment_option"
    EXCHANGE_RATE = "exchange_rate"


class ExclusionReason(str, Enum):
    PENDING_CREDIT = "pending_credit"
    UNREALIZED_VALUATION = "unrealized_valuation"
    CANCELLED = "cancelled"
    FAILED_NO_RETRY = "failed_no_retry"
    DUPLICATE_RECORD = "duplicate_record"
    INTERNAL_TRANSFER = "internal_transfer"
    SUPERSEDED_BY_AMENDMENT = "superseded_by_amendment"
    UNCONFIRMED_INCOME = "unconfirmed_income"
    OUTSIDE_FORECAST_WINDOW = "outside_forecast_window"
    NOT_RELEVANT = "not_relevant"
    UNTRUSTED_INSTRUCTION = "untrusted_instruction"


class ResolutionBasis(str, Enum):
    """Conflict-resolution precedence, highest priority first."""

    EXPLICIT_AMENDMENT = "explicit_amendment"
    NEWER_RECORD_SAME_SOURCE = "newer_record_same_source"
    SETTLED_OVER_ESTIMATE = "settled_over_estimate"
    SAFER_INTERPRETATION = "safer_interpretation"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class SourceRef(ContractModel):
    """Pointer to the dataset row or artifact that supports a derived fact."""

    kind: EvidenceKind
    ref_id: Identifier
    detail: Optional[str] = Field(default=None, max_length=500)


class ProfilePreferences(ContractModel):
    financial_priorities: tuple[str, ...] = ()
    protected_categories: frozenset[str] = frozenset()
    reducible_categories: frozenset[str] = frozenset()
    stoppable_categories: frozenset[str] = frozenset()
    accepted_payment_methods: frozenset[PaymentMethod]
    max_installment_months: Optional[int] = Field(default=None, ge=1, le=60)

    @model_validator(mode="after")
    def _installments_need_a_limit(self) -> "ProfilePreferences":
        considers_installments = PaymentMethod.INSTALLMENTS in self.accepted_payment_methods
        if considers_installments and self.max_installment_months is None:
            raise ValueError(
                "max_installment_months is required when the user considers installments"
            )
        if not considers_installments and self.max_installment_months is not None:
            raise ValueError(
                "max_installment_months must be empty when the user rejects installments"
            )
        return self


class CashFlowOccurrence(ContractModel):
    """One dated cash movement on the forward timeline, in home currency."""

    occurrence_id: Identifier
    date: date
    direction: Direction
    amount: PositiveMoney
    category: Annotated[str, Field(min_length=1, max_length=64)]
    origin: CashFlowOrigin
    certainty: Certainty
    is_protected: bool
    description: Annotated[str, Field(min_length=1, max_length=200)]
    adjustable_series_id: Optional[Identifier] = None
    source_event_id: Optional[EventId] = None
    original_currency: Optional[Currency] = None
    original_amount: Optional[PositiveMoney] = None
    exchange_rate_date: Optional[date] = None
    sources: tuple[SourceRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_direction_and_conversion(self) -> "CashFlowOccurrence":
        allowed = INFLOW_ORIGINS if self.direction is Direction.INFLOW else OUTFLOW_ORIGINS
        if self.origin not in allowed:
            raise ValueError(f"origin {self.origin.value} is invalid for {self.direction.value}")

        conversion_fields = (self.original_currency, self.original_amount, self.exchange_rate_date)
        if any(field is not None for field in conversion_fields) and not all(
            field is not None for field in conversion_fields
        ):
            raise ValueError(
                "original_currency, original_amount, and exchange_rate_date must be set together"
            )
        if self.original_currency is not None and not any(
            source.kind is EvidenceKind.EXCHANGE_RATE for source in self.sources
        ):
            raise ValueError("a converted amount must cite the exchange-rate row it used")
        if self.origin is CashFlowOrigin.VARIABLE_ESSENTIAL_FORECAST and (
            self.certainty is not Certainty.FORECAST
        ):
            raise ValueError("variable essential spending must be marked as a forecast")
        return self


class AdjustableSeries(ContractModel):
    """A flexible recurring expense the engine may stop or reduce.

    `target_event_id` is what a `spending_changes_needed` action must cite, so
    it has to be a real event the user can act on.
    """

    series_id: Identifier
    target_event_id: EventId
    category: Annotated[str, Field(min_length=1, max_length=64)]
    action: AdjustmentAction
    current_amount: PositiveMoney
    minimum_allowed_amount: Optional[Money] = None
    occurrence_ids: tuple[Identifier, ...] = Field(min_length=1)
    sources: tuple[SourceRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_reduction_floor(self) -> "AdjustableSeries":
        reducible = self.action in (AdjustmentAction.REDUCE, AdjustmentAction.STOP_OR_REDUCE)
        if reducible and self.minimum_allowed_amount is None:
            raise ValueError("a reducible series must state its minimum allowed amount")
        if self.minimum_allowed_amount is not None:
            if not reducible:
                raise ValueError("minimum_allowed_amount only applies to a reducible series")
            if self.minimum_allowed_amount >= self.current_amount:
                raise ValueError("minimum_allowed_amount must be below the current amount")
        if len(set(self.occurrence_ids)) != len(self.occurrence_ids):
            raise ValueError("occurrence_ids must be unique")
        return self


class EvidenceResolution(ContractModel):
    """An auditable record of how the normalizer settled ambiguous evidence."""

    resolution_id: Identifier
    summary: Annotated[str, Field(min_length=1, max_length=500)]
    basis: ResolutionBasis
    sources: tuple[SourceRef, ...] = Field(min_length=1)
    affected_occurrence_ids: tuple[Identifier, ...] = ()
    is_conservative_fallback: bool = False

    @model_validator(mode="after")
    def _fallbacks_must_be_safer(self) -> "EvidenceResolution":
        if self.is_conservative_fallback and self.basis is not ResolutionBasis.SAFER_INTERPRETATION:
            raise ValueError(
                "an unresolved conflict must be recorded as the safer interpretation"
            )
        return self


class ExcludedEvidence(ContractModel):
    source: SourceRef
    reason: ExclusionReason
    explanation: Annotated[str, Field(min_length=1, max_length=500)]


class ModelAttestation(ContractModel):
    """The normalizer's sign-off on the financial state it produced."""

    model_provider: Annotated[str, Field(min_length=1, max_length=64)]
    model_name: Annotated[str, Field(min_length=1, max_length=128)]
    generated_at: datetime
    attested: Literal[True]
    notes: Optional[str] = Field(default=None, max_length=1000)


class FinancialState(ContractModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    user_id: UserId
    home_currency: Currency
    as_of_date: date
    forecast_end_date: date
    opening_balance: Money
    minimum_balance_to_keep: Money
    preferences: ProfilePreferences
    cash_flows: tuple[CashFlowOccurrence, ...] = ()
    adjustable_series: tuple[AdjustableSeries, ...] = ()
    evidence_resolutions: tuple[EvidenceResolution, ...] = ()
    excluded_evidence: tuple[ExcludedEvidence, ...] = ()
    assumptions: tuple[str, ...] = ()
    attestation: ModelAttestation

    @model_validator(mode="after")
    def _check_timeline(self) -> "FinancialState":
        if (self.forecast_end_date - self.as_of_date).days != FORECAST_HORIZON_DAYS:
            raise ValueError(
                f"forecast_end_date must be exactly {FORECAST_HORIZON_DAYS} days after as_of_date"
            )

        occurrence_ids: set[str] = set()
        for flow in self.cash_flows:
            if flow.occurrence_id in occurrence_ids:
                raise ValueError(f"duplicate occurrence_id {flow.occurrence_id}")
            occurrence_ids.add(flow.occurrence_id)
            if not self.as_of_date <= flow.date <= self.forecast_end_date:
                raise ValueError(
                    f"occurrence {flow.occurrence_id} falls outside the forecast window"
                )

        series_ids: set[str] = set()
        for series in self.adjustable_series:
            if series.series_id in series_ids:
                raise ValueError(f"duplicate series_id {series.series_id}")
            series_ids.add(series.series_id)
            missing = set(series.occurrence_ids) - occurrence_ids
            if missing:
                raise ValueError(
                    f"series {series.series_id} references unknown occurrences: {sorted(missing)}"
                )

        for flow in self.cash_flows:
            if flow.adjustable_series_id is not None and flow.adjustable_series_id not in series_ids:
                raise ValueError(
                    f"occurrence {flow.occurrence_id} references unknown series "
                    f"{flow.adjustable_series_id}"
                )

        for resolution in self.evidence_resolutions:
            missing = set(resolution.affected_occurrence_ids) - occurrence_ids
            if missing:
                raise ValueError(
                    f"resolution {resolution.resolution_id} references unknown occurrences: "
                    f"{sorted(missing)}"
                )
        return self


class PaymentScheduleEntry(ContractModel):
    date: date
    amount: PositiveMoney


class PaymentOption(ContractModel):
    """A seller offer, expanded into the exact dates the buyer would pay."""

    payment_option_id: Identifier
    method: OfferedPaymentMethod
    payment_amount: PositiveMoney
    number_of_payments: Annotated[int, Field(ge=1, le=60)]
    first_payment_date: date
    payment_frequency_days: Optional[Annotated[int, Field(ge=1, le=366)]] = None
    financing_fee: Money
    total_payable_amount: PositiveMoney
    schedule: tuple[PaymentScheduleEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_schedule(self) -> "PaymentOption":
        if self.method is OfferedPaymentMethod.FULL_PAYMENT:
            if self.number_of_payments != 1:
                raise ValueError("a full payment option must have exactly one payment")
            if self.payment_frequency_days is not None:
                raise ValueError("a full payment option must not have a payment frequency")
        elif self.number_of_payments < 2 or self.payment_frequency_days is None:
            raise ValueError("an installment option needs multiple payments and a frequency")

        if len(self.schedule) != self.number_of_payments:
            raise ValueError("schedule length must equal number_of_payments")
        if self.schedule[0].date != self.first_payment_date:
            raise ValueError("the schedule must start on first_payment_date")

        previous = self.schedule[0]
        for entry in self.schedule[1:]:
            gap = (entry.date - previous.date).days
            if gap != self.payment_frequency_days:
                raise ValueError("schedule dates must follow payment_frequency_days")
            previous = entry

        scheduled_total = sum((entry.amount for entry in self.schedule), Decimal(0))
        if abs(scheduled_total - self.total_payable_amount) > Decimal("0.05"):
            raise ValueError("schedule amounts must add up to total_payable_amount")
        return self


class EnrichedRequest(ContractModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    request_id: RequestId
    user_id: UserId
    request_date: date
    request_type: RequestType
    requested_amount: PositiveMoney
    currency: Currency
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: Annotated[str, Field(min_length=1, max_length=2000)]
    payment_options: tuple[PaymentOption, ...] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def _check_options(self) -> "EnrichedRequest":
        if self.desired_completion_date < self.request_date:
            raise ValueError("desired_completion_date cannot precede request_date")
        option_ids = [option.payment_option_id for option in self.payment_options]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("payment_option_id values must be unique within a request")
        return self


class DecisionInput(ContractModel):
    """The complete, self-contained input to the deterministic decision engine."""

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    financial_state: FinancialState
    request: EnrichedRequest

    @model_validator(mode="after")
    def _check_alignment(self) -> "DecisionInput":
        state, request = self.financial_state, self.request
        if state.user_id != request.user_id:
            raise ValueError("financial state and request describe different users")
        if state.as_of_date != request.request_date:
            raise ValueError("financial state must be evaluated as of the request date")
        if state.home_currency is not request.currency:
            raise ValueError("request amounts must use the user's home currency")
        return self

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(f"unsupported contract version {value}")
        return value
