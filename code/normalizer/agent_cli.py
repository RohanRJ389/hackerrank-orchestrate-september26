"""Deterministic utilities intended to be called by Claude through Bash."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from contracts import DecisionInput, assert_valid

from .assembler import build_decision_input, state_hash
from .candidates import build_packet
from .dataset import Dataset
from .models import NormalizationDirectives


def _workspace() -> Path:
    return Path(os.environ.get("NORMALIZER_WORKSPACE", ".")).resolve()


def _inside(path: str) -> Path:
    root = _workspace()
    candidate = (root / path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path leaves the request workspace")
    return candidate


def _dataset() -> Dataset:
    root = Path(os.environ.get("NORMALIZER_DATASET_ROOT", "dataset")).resolve()
    return Dataset(root)


def _request_id() -> str:
    request_id = os.environ.get("NORMALIZER_REQUEST_ID")
    if not request_id:
        raise RuntimeError("NORMALIZER_REQUEST_ID is required")
    return request_id


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def command_build(args: argparse.Namespace) -> int:
    directives = NormalizationDirectives.model_validate_json(
        _inside(args.directives).read_text(encoding="utf-8")
    )
    if directives.request_id != _request_id():
        raise ValueError("directives target another request")
    dataset = _dataset()
    packet = build_packet(dataset, _request_id())
    decision_input, warnings = build_decision_input(
        dataset, packet, directives, model="pending-review"
    )
    digest = state_hash(decision_input.financial_state)
    output = _inside(args.output)
    _write(output, decision_input.model_dump(mode="json"))
    summary_path = _inside(args.summary)
    _write(
        summary_path,
        {
            "request_id": _request_id(),
            "state_hash": digest,
            "cash_flow_count": len(decision_input.financial_state.cash_flows),
            "adjustable_series_count": len(
                decision_input.financial_state.adjustable_series
            ),
            "excluded_evidence_count": len(
                decision_input.financial_state.excluded_evidence
            ),
            "resolution_count": len(
                decision_input.financial_state.evidence_resolutions
            ),
            "warnings": warnings,
            "assumptions": decision_input.financial_state.assumptions,
        },
    )
    print(
        json.dumps(
            {
                "valid": True,
                "state_hash": digest,
                "draft": output.name,
                "summary": summary_path.name,
                "warnings": warnings,
            }
        )
    )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    decision_input = DecisionInput.model_validate_json(
        _inside(args.draft).read_text(encoding="utf-8")
    )
    warnings = [str(item) for item in assert_valid(decision_input)]
    print(json.dumps({"valid": True, "warnings": warnings}))
    return 0


def command_hash(args: argparse.Namespace) -> int:
    decision_input = DecisionInput.model_validate_json(
        _inside(args.draft).read_text(encoding="utf-8")
    )
    print(state_hash(decision_input.financial_state))
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    decision_input = DecisionInput.model_validate_json(
        _inside(args.draft).read_text(encoding="utf-8")
    )
    state = decision_input.financial_state
    by_origin: dict[str, int] = {}
    for flow in state.cash_flows:
        by_origin[flow.origin.value] = by_origin.get(flow.origin.value, 0) + 1
    print(
        json.dumps(
            {
                "request_id": decision_input.request.request_id,
                "window": [state.as_of_date.isoformat(), state.forecast_end_date.isoformat()],
                "opening_balance": format(state.opening_balance, "f"),
                "minimum_balance": format(state.minimum_balance_to_keep, "f"),
                "cash_flows_by_origin": by_origin,
                "adjustable_targets": [
                    item.target_event_id for item in state.adjustable_series
                ],
                "exclusions": [
                    {
                        "ref_id": item.source.ref_id,
                        "reason": item.reason.value,
                    }
                    for item in state.excluded_evidence
                ],
                "resolutions": [
                    item.summary for item in state.evidence_resolutions
                ],
                "state_hash": state_hash(state),
            },
            indent=2,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="python -m normalizer.agent_cli")
    commands = result.add_subparsers(required=True)
    build = commands.add_parser("build")
    build.add_argument("directives")
    build.add_argument("--output", default="draft.json")
    build.add_argument("--summary", default="summary.json")
    build.set_defaults(func=command_build)
    validate = commands.add_parser("validate")
    validate.add_argument("draft", default="draft.json")
    validate.set_defaults(func=command_validate)
    hashing = commands.add_parser("hash")
    hashing.add_argument("draft", default="draft.json")
    hashing.set_defaults(func=command_hash)
    summary = commands.add_parser("summarize")
    summary.add_argument("draft", default="draft.json")
    summary.set_defaults(func=command_summarize)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except Exception as error:
        print(json.dumps({"valid": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
