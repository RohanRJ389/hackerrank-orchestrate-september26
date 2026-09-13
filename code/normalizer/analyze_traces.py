"""Summarize agent behavior from redacted JSONL traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze(paths: list[Path]) -> dict:
    events = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if "event_type" not in record and "model" in record:
                    record = {
                        "request_id": record["request_id"],
                        "event_type": "usage_record",
                        "payload": record,
                    }
                events.append(record)
    counts = Counter(event["event_type"] for event in events)
    sequences: dict[str, list[str]] = defaultdict(list)
    errors = Counter()
    models = Counter()
    totals = Counter()
    tools = Counter()
    tool_signatures = Counter()
    for event in events:
        sequences[event["request_id"]].append(event["event_type"])
        payload = event.get("payload") or {}
        if event["event_type"] == "tool_call":
            tool_name = payload.get("tool_name", "unknown")
            tools[tool_name] += 1
            signature = json.dumps(
                payload.get("tool_input"), sort_keys=True, default=str
            )
            tool_signatures[(event["request_id"], tool_name, signature)] += 1
        if event["event_type"] == "assistant_message" and payload.get("model"):
            models[payload["model"]] += 1
        if event["event_type"] == "usage_record":
            models[payload["model"]] += 1
            usage = payload.get("usage") or {}
            totals["input_tokens"] += usage.get("input_tokens", 0)
            totals["output_tokens"] += usage.get("output_tokens", 0)
            totals["cost_usd"] += payload.get("cost_usd") or 0
            continue
        if event["event_type"] in {"validation_error", "agent_error"}:
            errors[str(payload)[:300]] += 1
        if event["event_type"] == "result":
            model_usage = payload.get("model_usage") or {}
            for model, usage in model_usage.items():
                models[model] += 1
                totals["input_tokens"] += usage.get("inputTokens", 0)
                totals["output_tokens"] += usage.get("outputTokens", 0)
            totals["cost_usd"] += payload.get("total_cost_usd") or 0
    duplicate_sequences = Counter(" > ".join(value) for value in sequences.values())
    return {
        "event_counts": dict(counts),
        "request_count": len(sequences),
        "common_sequences": duplicate_sequences.most_common(10),
        "common_errors": errors.most_common(10),
        "tool_counts": dict(tools),
        "duplicate_tool_calls": [
            {"request_id": key[0], "tool": key[1], "count": count}
            for key, count in tool_signatures.most_common(20)
            if count > 1
        ],
        "model_runs": dict(models),
        "totals": dict(totals),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.paths), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
