"""Aggregate the final run ledger into the required usage report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_report(ledger: Path) -> str:
    records = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_model: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "input": 0, "output": 0, "cost": 0}
    )
    for record in records:
        item = by_model[record["model"]]
        usage = record.get("usage") or {}
        item["calls"] += 1
        item["input"] += usage.get("input_tokens", 0)
        item["output"] += usage.get("output_tokens", 0)
        item["cost"] += record.get("cost_usd", 0)
    requests = len({record["request_id"] for record in records})
    total_calls = int(sum(item["calls"] for item in by_model.values()))
    total_input = int(sum(item["input"] for item in by_model.values()))
    total_output = int(sum(item["output"] for item in by_model.values()))
    total_cost = sum(item["cost"] for item in by_model.values())
    lines = [
        "# Model Usage Report",
        "",
        f"Final run requests: {requests}",
        f"Total model calls: {total_calls}",
        f"Input tokens: {total_input}",
        f"Output tokens: {total_output}",
        f"Total tokens: {total_input + total_output}",
        f"Estimated cost: ${total_cost:.4f}",
        f"Average tokens per request: {(total_input + total_output) / requests if requests else 0:.2f}",
        f"Average cost per request: ${total_cost / requests if requests else 0:.6f}",
        "",
        "## Per model",
        "",
    ]
    for model, item in sorted(by_model.items()):
        lines.extend(
            [
                f"### {model}",
                f"- Calls: {int(item['calls'])}",
                f"- Input tokens: {int(item['input'])}",
                f"- Output tokens: {int(item['output'])}",
                f"- Estimated cost: ${item['cost']:.4f}",
                "",
            ]
        )
    lines.append(
        "Costs are the Agent SDK estimates reported by the provider. "
        "No credentials or raw financial evidence are included."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(args.ledger), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
