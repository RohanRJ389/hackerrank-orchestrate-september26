"""Create the disposable evidence workspace exposed to the agent."""

from __future__ import annotations

import hashlib
import json
import csv
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .dataset import Dataset
from .models import NormalizationDirectives, NormalizationPacket


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _stage_dataset(dataset: Dataset, packet: NormalizationPacket, root: Path) -> None:
    """Copy only the current user's CSV rows into the sandbox."""
    bundle = dataset.bundle(packet.request_id)
    data = root / "data"
    data.mkdir()
    _write_csv(data / "financial_profiles.csv", list(bundle.profile), [bundle.profile])
    _write_csv(data / "requests.csv", list(bundle.request), [bundle.request])
    _write_csv(data / "sample_requests.csv", list(bundle.request), [])
    _write_csv(
        data / "request_payment_options.csv",
        list(bundle.payment_options[0]),
        list(bundle.payment_options),
    )
    event_rows = [event.serializable() for event in bundle.events]
    _write_csv(data / "financial_events.csv", list(event_rows[0]), event_rows)
    message_rows = [message.serializable() for message in bundle.messages]
    message_fields = [
        "message_id", "user_id", "request_id", "related_event_id",
        "sent_at", "source_type", "message_text",
    ]
    _write_csv(data / "messages.csv", message_fields, message_rows)
    image_rows = [
        {
            "image_id": image.image_id,
            "user_id": image.user_id,
            "request_id": image.request_id,
            "related_event_id": image.related_event_id,
        }
        for image in bundle.images
    ]
    _write_csv(
        data / "images.csv",
        ["image_id", "user_id", "request_id", "related_event_id"],
        image_rows,
    )
    rate_rows = []
    seen = set()
    for rates in dataset.rates.values():
        for rate in rates:
            key = (rate.rate_date, rate.from_currency, rate.to_currency)
            if key in seen:
                continue
            seen.add(key)
            rate_rows.append(
                {
                    "rate_date": rate.rate_date.isoformat(),
                    "from_currency": rate.from_currency,
                    "to_currency": rate.to_currency,
                    "rate": format(rate.rate, "f"),
                }
            )
    _write_csv(
        data / "exchange_rates.csv",
        ["rate_date", "from_currency", "to_currency", "rate"],
        rate_rows,
    )
    media = data / "media" / "images"
    media.mkdir(parents=True)
    for image in bundle.images:
        if image.path.exists():
            shutil.copyfile(image.path, media / image.path.name)


def input_fingerprint(packet: NormalizationPacket) -> str:
    payload = packet.model_dump_json(exclude={"message_files", "image_files"})
    return hashlib.sha256(payload.encode()).hexdigest()


@contextmanager
def request_workspace(
    dataset: Dataset, packet: NormalizationPacket, keep: bool = False
) -> Iterator[tuple[Path, NormalizationPacket]]:
    root = Path(
        tempfile.mkdtemp(prefix=f"normalizer-{packet.request_id}-")
    ).resolve()
    try:
        messages_dir = root / "messages"
        images_dir = root / "images"
        messages_dir.mkdir()
        images_dir.mkdir()
        bundle = dataset.bundle(packet.request_id)
        _stage_dataset(dataset, packet, root)
        runtime = root / "runtime"
        shutil.copytree(Path(__file__).resolve().parent, runtime / "normalizer")
        shutil.copytree(Path(__file__).resolve().parents[1] / "contracts", runtime / "contracts")
        message_files = []
        for message in bundle.messages:
            path = messages_dir / f"{message.message_id}.json"
            _write_json(path, message.serializable())
            message_files.append(str(path.relative_to(root)))
        image_files = []
        for image in bundle.images:
            if not image.path.exists():
                continue
            path = images_dir / image.path.name
            shutil.copyfile(image.path, path)
            image_files.append(str(path.relative_to(root)))
        staged = packet.model_copy(
            update={
                "message_files": tuple(message_files),
                "image_files": tuple(image_files),
            }
        )
        _write_json(root / "packet.json", staged.model_dump(mode="json"))
        _write_json(root / "directives.schema.json", NormalizationDirectives.model_json_schema())
        _write_json(
            root / "directives.json",
            NormalizationDirectives(request_id=packet.request_id).model_dump(mode="json"),
        )
        _write_json(
            root / "event_history.json",
            [event.serializable() for event in bundle.events],
        )
        (root / "README.txt").write_text(
            "Do not read this file. Paths to read are listed in the user prompt.\n"
            "Return NormalizationDirectives as structured output.\n",
            encoding="utf-8",
        )
        yield root, staged
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
