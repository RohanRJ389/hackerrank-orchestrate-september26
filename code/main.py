"""Buy or Wait? entry point: normalize, decide, write repo-root output.csv."""

from __future__ import annotations

import asyncio
import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from engine import CSV_COLUMNS, decide
from normalizer.assembler import build_decision_input
from normalizer.candidates import build_packet
from normalizer.claude import AgentRunner, HAIKU_MODEL
from normalizer.dataset import Dataset
from normalizer.models import Route
from normalizer.normalize import DEFAULT_TRACES, normalize_async
from normalizer.usage import build_report

OUTPUT_PATH = ROOT / "output.csv"
USAGE_REPORT = CODE / "evaluation" / "usage_report.md"
REQUEST_TIMEOUT_S = float(os.environ.get("NORMALIZER_REQUEST_TIMEOUT_S", "90"))


def evaluation_ids() -> list[str]:
    path = ROOT / "dataset" / "requests.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["request_id"] for row in csv.DictReader(handle)]


def write_output(request_ids: list[str], rows: dict[str, dict[str, str]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for request_id in request_ids:
            if request_id in rows:
                writer.writerow(rows[request_id])


def _fallback(dataset: Dataset, request_id: str):
    packet = build_packet(dataset, request_id)
    decision_input, _ = build_decision_input(
        dataset, packet, None, model="deterministic"
    )
    return decision_input, packet.route.value, "fallback"


async def _one(
    request_id: str,
    dataset: Dataset,
    runner: AgentRunner,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict[str, str], str, str]:
    async with semaphore:
        started = time.monotonic()
        packet = build_packet(dataset, request_id)
        source = "deterministic"
        try:
            if packet.route is Route.DETERMINISTIC_REVIEW:
                decision_input, _ = build_decision_input(
                    dataset, packet, None, model="deterministic"
                )
            else:
                decision_input = await asyncio.wait_for(
                    normalize_async(
                        request_id,
                        dataset=dataset,
                        runner=runner,
                        model=HAIKU_MODEL,
                        use_cache=True,
                    ),
                    timeout=REQUEST_TIMEOUT_S,
                )
                source = "claude"
        except Exception as error:
            print(f"{request_id} failed ({error}); using deterministic fallback", flush=True)
            decision_input, _, source = _fallback(dataset, request_id)
        row = decide(decision_input).to_row()
        elapsed = time.monotonic() - started
        print(
            f"{request_id} {packet.route.value} {source} {elapsed:.1f}s "
            f"{row['affordability_status']}/{row['recommended_payment_method']}",
            flush=True,
        )
        return request_id, row, packet.route.value, source


async def run() -> int:
    load_dotenv(ROOT / ".env")
    request_ids = evaluation_ids()
    if not request_ids:
        raise SystemExit("dataset/requests.csv has no requests")
    dataset = Dataset()
    concurrency = int(os.environ.get("NORMALIZER_CONCURRENCY", "5"))
    runner = AgentRunner(model=HAIKU_MODEL, trace_root=DEFAULT_TRACES)
    semaphore = asyncio.Semaphore(concurrency)
    ledger = DEFAULT_TRACES / "usage.jsonl"
    ledger_start = ledger.stat().st_size if ledger.exists() else 0

    rows: dict[str, dict[str, str]] = {}
    lock = asyncio.Lock()
    write_output(request_ids, rows)

    async def collect(request_id: str) -> None:
        request_id, row, _route, _source = await _one(
            request_id, dataset, runner, semaphore
        )
        async with lock:
            rows[request_id] = row
            write_output(request_ids, rows)

    await asyncio.gather(*(collect(request_id) for request_id in request_ids))
    write_output(request_ids, rows)
    if len(rows) != len(request_ids):
        raise SystemExit(f"wrote {len(rows)} rows, expected {len(request_ids)}")

    if ledger.exists() and ledger.stat().st_size > ledger_start:
        slice_path = DEFAULT_TRACES / "final-run-usage.jsonl"
        with ledger.open("rb") as handle:
            handle.seek(ledger_start)
            slice_path.write_bytes(handle.read())
        USAGE_REPORT.parent.mkdir(parents=True, exist_ok=True)
        USAGE_REPORT.write_text(build_report(slice_path), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({len(rows)} rows)", flush=True)
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
