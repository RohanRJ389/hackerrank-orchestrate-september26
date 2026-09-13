"""Deterministic decision engine.

Consumes a validated `DecisionInput` and produces one `output.csv` row. It
performs no inference about recurrence, currency, or evidence conflicts: those
belong to the normalizer.
"""

from .decide import decide
from .forecast import SpendingChange, Timeline, format_amount
from .output import CSV_COLUMNS, DecisionOutput, format_safe_amount
from .plans import Plan, candidate_plans, change_sets, eligible_installment_options
from .ranking import best_plan, completes_in_time, rank_key

__all__ = [
    "CSV_COLUMNS",
    "DecisionOutput",
    "Plan",
    "SpendingChange",
    "Timeline",
    "best_plan",
    "candidate_plans",
    "change_sets",
    "completes_in_time",
    "decide",
    "eligible_installment_options",
    "format_amount",
    "format_safe_amount",
    "rank_key",
]
