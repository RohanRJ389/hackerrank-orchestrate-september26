"""Public normalization API, retries, cache, and bounded concurrency."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from contracts import DecisionInput, assert_valid
from dotenv import load_dotenv

from .assembler import build_decision_input
from .candidates import build_packet
from .claude import AgentRun, AgentRunner, HAIKU_MODEL, parse_max_turns
from .prompts import PROMPT_VERSION
from .dataset import DEFAULT_DATASET, Dataset
from .workspace import request_workspace

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "normalizer-cache"
DEFAULT_TRACES = ROOT / "normalizer-traces"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _fingerprint(dataset: Dataset, request_id: str, model: str) -> str:
    bundle = dataset.bundle(request_id)
    payload = {
        "request_id": request_id,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "profile": bundle.profile,
        "request": bundle.request,
        "options": bundle.payment_options,
        "events": [event.serializable() for event in bundle.events],
        "messages": [message.serializable() for message in bundle.messages],
        "images": [
            {
                **image.serializable(),
                "sha256": (
                    hashlib.sha256(image.path.read_bytes()).hexdigest()
                    if image.path.exists()
                    else None
                ),
            }
            for image in bundle.images
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _record_usage(
    path: Path,
    request_id: str,
    *,
    model: str,
    retry: int,
    wall_ms: int,
    started_at: str,
    ended_at: str,
    run: AgentRun | None = None,
    error: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": ended_at,
        "started_at": started_at,
        "ended_at": ended_at,
        "request_id": request_id,
        "model": model if run is None else run.model,
        "wall_ms": wall_ms,
        "wall_s": round(wall_ms / 1000, 3),
        "duration_ms": None if run is None else run.duration_ms,
        "turns": None if run is None else run.turns,
        "cost_usd": None if run is None else run.cost_usd,
        "usage": None if run is None else run.usage,
        "model_usage": None if run is None else run.model_usage,
        "session_id": None if run is None else run.session_id,
        "retry": retry,
        "error": error,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    timing_path = path.with_name("timing.jsonl")
    with timing_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "request_id": request_id,
                    "model": record["model"],
                    "retry": retry,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "wall_ms": wall_ms,
                    "wall_s": record["wall_s"],
                    "sdk_duration_ms": record["duration_ms"],
                    "turns": record["turns"],
                    "cost_usd": record["cost_usd"],
                    "ok": error is None,
                    "error": error,
                },
                default=str,
            )
            + "\n"
        )


def _attach_attestation(
    decision_input: DecisionInput, run: AgentRun
) -> DecisionInput:
    attestation = decision_input.financial_state.attestation.model_copy(
        update={
            "model_provider": "anthropic",
            "model_name": run.model,
            "generated_at": datetime.now(timezone.utc),
            "attested": True,
            "notes": "assembled_from_model_directives",
        }
    )
    final = decision_input.model_copy(
        update={
            "financial_state": decision_input.financial_state.model_copy(
                update={"attestation": attestation}
            )
        }
    )
    assert_valid(final)
    return final


async def normalize_async(
    request_id: str,
    *,
    dataset: Dataset | None = None,
    runner: AgentRunner | None = None,
    model: str | None = None,
    use_cache: bool = True,
    keep_workspace: bool = False,
) -> DecisionInput:
    load_dotenv(ROOT / ".env")
    dataset = dataset or Dataset(DEFAULT_DATASET)
    selected_model = model or os.environ.get("NORMALIZER_MODEL", HAIKU_MODEL)
    runner = runner or AgentRunner(
        model=selected_model,
        max_turns=parse_max_turns(os.environ.get("NORMALIZER_MAX_TURNS")),
        max_budget_usd=float(os.environ.get("NORMALIZER_MAX_BUDGET_USD", "1.0")),
        trace_root=DEFAULT_TRACES,
    )
    cache_root = DEFAULT_CACHE
    fingerprint = _fingerprint(dataset, request_id, selected_model)
    cache_path = cache_root / f"{fingerprint}.json"
    if use_cache and cache_path.exists():
        cached = DecisionInput.model_validate_json(cache_path.read_text(encoding="utf-8"))
        assert_valid(cached)
        return cached

    packet = build_packet(dataset, request_id)
    last_error: Exception | None = None
    with request_workspace(dataset, packet, keep=keep_workspace) as (workspace, staged):
        for retry in range(2):
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.monotonic()
            run: AgentRun | None = None
            try:
                run = await runner.run(
                    workspace, request_id, staged.route, packet=staged
                )
                _write_json(
                    workspace / "directives.json",
                    run.directives.model_dump(mode="json"),
                )
                assembled, _warnings = build_decision_input(
                    dataset, packet, run.directives, model=run.model
                )
                final = _attach_attestation(assembled, run)
                _record_usage(
                    DEFAULT_TRACES / "usage.jsonl",
                    request_id,
                    model=selected_model,
                    retry=retry,
                    wall_ms=int((time.monotonic() - started) * 1000),
                    started_at=started_at,
                    ended_at=datetime.now(timezone.utc).isoformat(),
                    run=run,
                )
                if use_cache:
                    cache_root.mkdir(parents=True, exist_ok=True)
                    _write_json(cache_path, final.model_dump(mode="json"))
                return final
            except Exception as error:
                last_error = error
                _record_usage(
                    DEFAULT_TRACES / "usage.jsonl",
                    request_id,
                    model=selected_model,
                    retry=retry,
                    wall_ms=int((time.monotonic() - started) * 1000),
                    started_at=started_at,
                    ended_at=datetime.now(timezone.utc).isoformat(),
                    run=run,
                    error=str(error),
                )
                _write_json(
                    workspace / "retry_feedback.json",
                    {
                        "attempt": retry + 1,
                        "error": str(error),
                        "instruction": "Fix the directives and return them again.",
                    },
                )
        raise RuntimeError(
            f"normalization failed after one retry for {request_id}: {last_error}"
        )


def normalize(request_id: str, **kwargs) -> DecisionInput:
    return asyncio.run(normalize_async(request_id, **kwargs))


async def normalize_many(
    request_ids: Iterable[str],
    *,
    concurrency: int | None = None,
    **kwargs,
) -> list[DecisionInput]:
    limit = concurrency or int(os.environ.get("NORMALIZER_CONCURRENCY", "5"))
    semaphore = asyncio.Semaphore(limit)

    async def one(request_id: str) -> DecisionInput:
        async with semaphore:
            return await normalize_async(request_id, **kwargs)

    return await asyncio.gather(*(one(request_id) for request_id in request_ids))
