"""Internal contracts for deterministic assembly and model review."""

from __future__ import annotations

from datetime import date as Date
from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InternalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Route(str, Enum):
    DETERMINISTIC_REVIEW = "deterministic_review"
    EVIDENCE_REVIEW = "evidence_review"


class CandidateKind(str, Enum):
    SALARY = "salary"
    FIXED_RECURRENCE = "fixed_recurrence"
    VARIABLE_ESSENTIAL = "variable_essential"


class RecurrenceCandidate(InternalModel):
    candidate_id: str
    kind: CandidateKind
    direction: Literal["credit", "debit"]
    category: str
    description: str
    amount: Decimal
    currency: str
    anchor_date: Date
    interval_days: int | None = Field(default=None, ge=1)
    monthly: bool
    source_event_ids: tuple[str, ...]
    confidence: Decimal = Field(ge=0, le=1)
    adjustable: bool = False
    flexibility: str = "fixed"
    minimum_allowed_amount: Decimal | None = None


class VariableBudget(InternalModel):
    category: str
    weekly_amount: Decimal = Field(ge=0)
    currency: str
    source_event_ids: tuple[str, ...]
    observation_days: int = Field(ge=1)


class NormalizationPacket(InternalModel):
    packet_version: Literal["1.0.0"] = "1.0.0"
    request_id: str
    user_id: str
    route: Route
    profile: dict[str, str]
    request: dict[str, str]
    payment_options: tuple[dict[str, str], ...]
    future_events: tuple[dict, ...]
    recurrence_candidates: tuple[RecurrenceCandidate, ...]
    variable_budgets: tuple[VariableBudget, ...]
    message_files: tuple[str, ...] = ()
    image_files: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class CandidateDecision(InternalModel):
    candidate_id: str
    action: Literal["accept", "reject", "amend"] = "accept"
    amount: Decimal | None = Field(default=None, gt=0)
    anchor_date: Date | None = None
    end_date: Date | None = None
    rationale: str = Field(default="", max_length=500)
    source_refs: tuple[str, ...] = ()


class EventDecision(InternalModel):
    event_id: str
    action: Literal["keep", "exclude", "amend", "add"] = "keep"
    amount: Decimal | None = Field(default=None, gt=0)
    date: Date | None = None
    direction: Literal["credit", "debit"] | None = None
    category: str | None = None
    origin: str | None = None
    exclusion_reason: str | None = None
    rationale: str = Field(default="", max_length=500)
    source_refs: tuple[str, ...] = ()


class VariableBudgetDecision(InternalModel):
    category: str
    weekly_amount: Decimal = Field(ge=0)
    rationale: str = Field(default="", max_length=500)
    source_refs: tuple[str, ...] = ()


class NormalizationDirectives(InternalModel):
    request_id: str
    candidate_decisions: tuple[CandidateDecision, ...] = ()
    event_decisions: tuple[EventDecision, ...] = ()
    variable_budget_decisions: tuple[VariableBudgetDecision, ...] = ()
    assumptions: tuple[str, ...] = ()
    untrusted_instruction_sources: tuple[str, ...] = ()


class DraftResult(InternalModel):
    request_id: str
    valid: bool
    state_hash: str | None = None
    draft_path: str | None = None
    summary_path: str | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class BenchmarkResult(InternalModel):
    request_id: str
    model: str
    valid: bool
    duration_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    retries: int = 0
    exact_fields: int | None = None

    @model_validator(mode="after")
    def _tokens_nonnegative(self) -> "BenchmarkResult":
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts cannot be negative")
        return self
