"""Public normalization API, retries, cache, and bounded concurrency."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from contracts import DecisionInput, assert_valid
from dotenv import load_dotenv

from .assembler import build_decision_input, state_hash
from .candidates import build_packet
from .claude import AgentRun, AgentRunner, HAIKU_MODEL
from .dataset import DEFAULT_DATASET, Dataset
from .models import NormalizationDirectives
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
        "prompt_version": "1.0.0",
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


def _record_usage(path: Path, request_id: str, run: AgentRun, retry: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model": run.model,
        "duration_ms": run.duration_ms,
        "turns": run.turns,
        "cost_usd": run.cost_usd,
        "usage": run.usage,
        "model_usage": run.model_usage,
        "session_id": run.session_id,
        "retry": retry,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _attach_approval(
    decision_input: DecisionInput, run: AgentRun
) -> DecisionInput:
    digest = state_hash(decision_input.financial_state)
    if digest != run.approval.state_hash:
        raise ValueError(
            f"approved hash {run.approval.state_hash} does not match draft {digest}"
        )
    attestation = decision_input.financial_state.attestation.model_copy(
        update={
            "model_provider": "anthropic",
            "model_name": run.model,
            "generated_at": datetime.now(timezone.utc),
            "attested": True,
            "notes": (
                f"approved_unsigned_state_sha256={digest}; "
                f"{run.approval.notes}"
            )[:1000],
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
        max_turns=int(os.environ.get("NORMALIZER_MAX_TURNS", "12")),
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
        empty = NormalizationDirectives(request_id=request_id)
        _write_json(workspace / "directives.json", empty.model_dump(mode="json"))
        baseline, warnings = build_decision_input(
            dataset, packet, empty, model="pending-review"
        )
        _write_json(workspace / "draft.json", baseline.model_dump(mode="json"))
        _write_json(
            workspace / "summary.json",
            {
                "request_id": request_id,
                "state_hash": state_hash(baseline.financial_state),
                "warnings": warnings,
                "cash_flow_count": len(baseline.financial_state.cash_flows),
            },
        )
        for retry in range(2):
            try:
                run = await runner.run(workspace, request_id, staged.route)
                draft = DecisionInput.model_validate_json(
                    (workspace / "draft.json").read_text(encoding="utf-8")
                )
                assert_valid(draft)
                final = _attach_approval(draft, run)
                _record_usage(
                    DEFAULT_TRACES / "usage.jsonl", request_id, run, retry
                )
                if use_cache:
                    cache_root.mkdir(parents=True, exist_ok=True)
                    _write_json(cache_path, final.model_dump(mode="json"))
                return final
            except Exception as error:
                last_error = error
                _write_json(
                    workspace / "retry_feedback.json",
                    {
                        "attempt": retry + 1,
                        "error": str(error),
                        "instruction": "Correct or rebuild draft.json, then sign its exact hash.",
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
