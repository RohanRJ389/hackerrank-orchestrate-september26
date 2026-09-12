"""Regenerate the checked-in JSON Schema from the Pydantic contract.

Run from the `code/` directory with: python3 -m contracts.generate_schema
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import DecisionInput

SCHEMA_PATH = Path(__file__).with_name("financial_boundary.schema.json")


def build_schema() -> dict:
    schema = DecisionInput.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "DecisionInput"
    return schema


def main() -> None:
    SCHEMA_PATH.write_text(json.dumps(build_schema(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
