"""Compare Haiku and Sonnet on representative normalization requests."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

from engine import decide

from .claude import HAIKU_MODEL, SONNET_MODEL
from .dataset import DEFAULT_DATASET, Dataset
from .normalize import DEFAULT_TRACES, normalize_async

REPRESENTATIVE_REQUESTS = (
    "request_01",  # pure structured
    "request_02",  # salary amendment
    "request_03",  # image salary
    "request_06",  # temporary salary
    "request_10",  # pending income
    "request_11",  # flexible spending
    "request_16",  # image rent
    "request_20",  # image/pending credit
    "request_71",  # foreign currency
    "request_88",  # advance-fee scam
)
FIELDS = (
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
)


def _samples(dataset: Dataset) -> dict[str, dict[str, str]]:
    with (DEFAULT_DATASET / "sample_requests.csv").open(newline="", encoding="utf-8") as handle:
        return {row["request_id"]: row for row in csv.DictReader(handle)}


async def benchmark(
    models: tuple[str, ...],
    request_ids: tuple[str, ...] = REPRESENTATIVE_REQUESTS,
) -> list[dict]:
    dataset = Dataset()
    samples = _samples(dataset)
    results = []
    for model in models:
        for request_id in request_ids:
            started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            started = time.monotonic()
            record = {"model": model, "request_id": request_id, "started_at": started_at}
            try:
                decision_input = await normalize_async(
                    request_id, dataset=dataset, model=model, use_cache=False
                )
                row = decide(decision_input).to_row()
                wall_ms = int((time.monotonic() - started) * 1000)
                record.update(
                    {
                        "valid": True,
                        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "wall_ms": wall_ms,
                        "wall_s": round(wall_ms / 1000, 3),
                        "exact_fields": (
                            sum(row[field] == samples[request_id][field] for field in FIELDS)
                            if request_id in samples
                            else None
                        ),
                        "output": row,
                    }
                )
            except Exception as error:
                wall_ms = int((time.monotonic() - started) * 1000)
                record.update(
                    {
                        "valid": False,
                        "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "wall_ms": wall_ms,
                        "wall_s": round(wall_ms / 1000, 3),
                        "error": str(error),
                    }
                )
            results.append(record)
            print(json.dumps(record, default=str), flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=[HAIKU_MODEL, SONNET_MODEL],
    )
    parser.add_argument("--requests", nargs="+", default=list(REPRESENTATIVE_REQUESTS))
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_TRACES / "benchmark.json"
    )
    args = parser.parse_args()
    results = asyncio.run(benchmark(tuple(args.models), tuple(args.requests)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str) + "\n")
    return 0 if all(item["valid"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
