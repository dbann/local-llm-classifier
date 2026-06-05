#!/usr/bin/env python3
"""
Classify abstracts stored as JSON files with a local Ollama model.

Default behaviour:
  - Reads every *.json file in ./data/input.
  - Expects each JSON file to contain a list of records with an "abstract" field.
  - Sends only the configured text field and context fields to the model.
  - Writes classified JSON to ./data/output/<input_stem>_classified.json.
  - Resumes from existing output files unless --force is used.

Run:
  python3 app.py

Useful overrides:
  python3 app.py --config classify_config.yaml
  OLLAMA_MODEL=llama3.1 python3 app.py
  python3 app.py --model gemma3:270m --concurrency 2
  python3 app.py --input data/input/example_abstracts.json --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/input")
DEFAULT_OUTPUT_DIR = Path("data/output")
DEFAULT_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:270m")
DEFAULT_QUESTION = (
    "Does the following abstract make a policy claim? Answer yes or no."
)
DEFAULT_ID_FIELD = "scopus_id"
DEFAULT_TEXT_FIELD = "abstract"
DEFAULT_CONTEXT_FIELDS = ["title"]
CONFIG_KEYS = {
    "input",
    "output_dir",
    "model",
    "base_url",
    "question",
    "concurrency",
    "timeout",
    "max_retries",
    "limit",
    "checkpoint_every",
    "force",
    "id_field",
    "text_field",
    "context_fields",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON or simple YAML config file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="A JSON file, or a directory containing JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for classified JSON output.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model tag, for example gemma3:27b or llama3.1.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Ollama base URL without a trailing path.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Classification question sent to the model.",
    )
    parser.add_argument(
        "--id-field",
        default=None,
        help="Record field used as the stable ID. Defaults to scopus_id.",
    )
    parser.add_argument(
        "--text-field",
        default=None,
        help="Record field sent to the model for classification. Defaults to abstract.",
    )
    parser.add_argument(
        "--context-fields",
        default=None,
        help="Comma-separated fields sent as extra context. Defaults to title.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of simultaneous Ollama requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Seconds to wait for each Ollama request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Retry count for transient Ollama errors.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Classify only the first N records in each input file.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Write the output JSON after this many completed classifications.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reclassify records even if they already exist in the output.",
    )
    args = parser.parse_args()
    return apply_config(args)


def default_options() -> dict[str, Any]:
    return {
        "input": DEFAULT_INPUT,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "model": DEFAULT_MODEL,
        "base_url": DEFAULT_OLLAMA_BASE_URL,
        "question": DEFAULT_QUESTION,
        "id_field": DEFAULT_ID_FIELD,
        "text_field": DEFAULT_TEXT_FIELD,
        "context_fields": DEFAULT_CONTEXT_FIELDS,
        "concurrency": int(os.environ.get("CLASSIFY_CONCURRENCY", "2")),
        "timeout": float(os.environ.get("CLASSIFY_TIMEOUT", "120")),
        "max_retries": int(os.environ.get("CLASSIFY_MAX_RETRIES", "3")),
        "limit": None,
        "checkpoint_every": 1,
        "force": False,
    }


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    options = default_options()

    if args.config:
        config = load_config(args.config)
        unknown_keys = sorted(set(config) - CONFIG_KEYS)
        if unknown_keys:
            raise ValueError(
                f"Unsupported config key(s) in {args.config}: {', '.join(unknown_keys)}"
            )
        options.update(config)

    for key in CONFIG_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            options[key] = value

    options["input"] = Path(options["input"])
    options["output_dir"] = Path(options["output_dir"])
    options["concurrency"] = int(options["concurrency"])
    options["timeout"] = float(options["timeout"])
    options["max_retries"] = int(options["max_retries"])
    options["checkpoint_every"] = int(options["checkpoint_every"])
    options["force"] = bool(options["force"])
    options["id_field"] = str(options["id_field"])
    options["text_field"] = str(options["text_field"])
    options["context_fields"] = parse_field_list(options["context_fields"])
    if options["limit"] in ("", "null", "none"):
        options["limit"] = None
    elif options["limit"] is not None:
        options["limit"] = int(options["limit"])

    return argparse.Namespace(config=args.config, **options)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {display_path(path)}")

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as file:
        if suffix == ".json":
            data = json.load(file)
        elif suffix in {".yaml", ".yml"}:
            data = parse_simple_yaml(file.read())
        else:
            raise ValueError("Config file must be .json, .yaml, or .yml")

    if not isinstance(data, dict):
        raise ValueError(
            f"Config file must contain key/value settings: {display_path(path)}"
        )
    return {str(key).replace("-", "_"): value for key, value in data.items()}


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple key/value YAML shape used by classify_config.yaml."""
    data: dict[str, Any] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line {line_number}: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip().replace("-", "_")
        if not key:
            raise ValueError(f"Missing config key on line {line_number}")
        data[key] = parse_config_value(value.strip())
    return data


def parse_config_value(value: str) -> Any:
    if not value:
        return None

    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None

    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_field_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [field.strip() for field in str(value).split(",") if field.strip()]


def discover_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Input file must be JSON: {display_path(input_path)}")
        return [input_path]

    if input_path.is_dir():
        files = sorted(path for path in input_path.glob("*.json") if path.is_file())
        if files:
            return files
        raise FileNotFoundError(f"No .json files found in {display_path(input_path)}")

    raise FileNotFoundError(f"Input path not found: {display_path(input_path)}")


def load_json_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        for key in ("records", "items", "abstracts", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"{display_path(path)} must contain a JSON list of records")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(data[:limit] if limit is not None else data):
        if not isinstance(item, dict):
            raise ValueError(
                f"{display_path(path)} item {index} is not a JSON object"
            )
        records.append(item)
    return records


def output_path_for(input_file: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_file.stem}_classified.json"


def display_path(path: Path) -> str:
    """Return a user-facing path without printing absolute local directories."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, RuntimeError, ValueError):
        pass

    if path.is_absolute():
        parent_name = path.parent.name
        if parent_name:
            return str(Path(parent_name) / path.name)
        return path.name
    return str(path)


def field_text(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if value in (None, ""):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def record_key(record: dict[str, Any], index: int, id_field: str) -> str:
    if id_field:
        value = field_text(record, id_field)
        if value:
            return f"{id_field}:{value}"

    for field in ("scopus_id", "doi", "id", "pmid", "title"):
        value = record.get(field)
        if value not in (None, ""):
            return f"{field}:{value}"
    return f"index:{index}"


def completed_classification(record: dict[str, Any]) -> bool:
    classification = record.get("classification")
    if not isinstance(classification, dict):
        return False
    label = str(classification.get("answer") or classification.get("label", "")).lower()
    return label in {"yes", "no", "unknown"}


def normalise_existing_record(
    record: dict[str, Any],
    *,
    id_field: str,
    text_field: str,
    context_fields: list[str],
) -> dict[str, Any]:
    output_record = dict(record)
    classification = output_record.get("classification")
    if not isinstance(classification, dict):
        return output_record

    label = str(classification.get("answer") or classification.get("label", "")).lower()
    if label in {"yes", "no", "unknown"}:
        normalised = {"answer": label, "label": label}
        normalised.update(classification)
        normalised["answer"] = label
        normalised["label"] = label
        normalised["id_field"] = classification.get("id_field") or id_field
        normalised["text_field"] = classification.get("text_field") or text_field
        normalised["context_fields"] = (
            parse_field_list(classification.get("context_fields")) or context_fields
        )
        output_record["classification"] = normalised
    return output_record


def classification_matches_settings(
    classification: dict[str, Any],
    *,
    question: str,
    model: str,
    text_field: str,
    context_fields: list[str],
) -> bool:
    if classification.get("question") != question:
        return False
    if classification.get("model") != model:
        return False

    existing_text_field = classification.get("text_field")
    if existing_text_field is None:
        existing_text_field = DEFAULT_TEXT_FIELD
    if existing_text_field != text_field:
        return False

    existing_context_fields = classification.get("context_fields")
    if existing_context_fields is None:
        existing_context_fields = DEFAULT_CONTEXT_FIELDS
    if parse_field_list(existing_context_fields) != context_fields:
        return False

    return True


def load_existing_by_key(
    path: Path,
    *,
    id_field: str,
    question: str,
    model: str,
    text_field: str,
    context_fields: list[str],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        return {}

    existing: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(data):
        if not isinstance(record, dict) or not completed_classification(record):
            continue
        classification = record.get("classification")
        if not isinstance(classification, dict):
            continue
        if not classification_matches_settings(
            classification,
            question=question,
            model=model,
            text_field=text_field,
            context_fields=context_fields,
        ):
            continue
        existing[record_key(record, index, id_field)] = normalise_existing_record(
            record,
            id_field=id_field,
            text_field=text_field,
            context_fields=context_fields,
        )
    return existing


def write_json_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temp_path, path)


def parse_label(raw_text: str) -> str:
    text = raw_text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            label = str(data.get("label", "")).strip().lower()
            if label in {"yes", "no"}:
                return label
    except json.JSONDecodeError:
        pass

    lowered = text.lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    if '"yes"' in lowered and '"no"' not in lowered:
        return "yes"
    if '"no"' in lowered and '"yes"' not in lowered:
        return "no"
    if "yes" in lowered and "no" not in lowered:
        return "yes"
    if "no" in lowered and "yes" not in lowered:
        return "no"
    return "unknown"


def build_prompt(
    record: dict[str, Any],
    *,
    question: str,
    text_field: str,
    context_fields: list[str],
) -> str:
    parts = [
        question,
        'Return only valid JSON in this exact shape: {"label":"yes"} or {"label":"no"}.',
    ]

    for field in context_fields:
        value = field_text(record, field).strip()
        if value:
            parts.append(f"{field}: {value}")

    parts.append(f"{text_field}: {field_text(record, text_field).strip()}")
    return "\n\n".join(parts)


def ollama_chat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise classifier of academic abstracts. "
                    "Return only JSON. The label must be yes or no."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 8,
        },
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    try:
        return str(data["message"]["content"])
    except KeyError as exc:
        raise ValueError(f"Unexpected Ollama response: {data}") from exc


def classify_record(
    *,
    record: dict[str, Any],
    index: int,
    question: str,
    id_field: str,
    text_field: str,
    context_fields: list[str],
    base_url: str,
    model: str,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    review_text = field_text(record, text_field).strip()
    output_record = dict(record)

    if not review_text:
        output_record["classification"] = {
            "answer": "unknown",
            "label": "unknown",
            "note": f"Missing text field: {text_field}",
            "question": question,
            "model": model,
            "id_field": id_field,
            "text_field": text_field,
            "context_fields": context_fields,
            "record_key": record_key(record, index, id_field),
            "classified_at": now_utc(),
        }
        return output_record

    prompt = build_prompt(
        record,
        question=question,
        text_field=text_field,
        context_fields=context_fields,
    )
    raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw = ollama_chat(
                base_url=base_url,
                model=model,
                prompt=prompt,
                timeout=timeout,
            )
            label = parse_label(raw)
            output_record["classification"] = {
                "answer": label,
                "label": label,
                "raw": raw.strip(),
                "question": question,
                "model": model,
                "id_field": id_field,
                "text_field": text_field,
                "context_fields": context_fields,
                "record_key": record_key(record, index, id_field),
                "classified_at": now_utc(),
            }
            return output_record
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            if attempt >= max_retries:
                output_record["classification"] = {
                    "answer": "error",
                    "label": "error",
                    "raw": raw.strip(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "question": question,
                    "model": model,
                    "id_field": id_field,
                    "text_field": text_field,
                    "context_fields": context_fields,
                    "record_key": record_key(record, index, id_field),
                    "classified_at": now_utc(),
                }
                return output_record
            time.sleep(2 * attempt)

    return output_record


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def classify_file(input_file: Path, args: argparse.Namespace) -> None:
    records = load_json_records(input_file, args.limit)
    output_path = output_path_for(input_file, args.output_dir)
    existing_by_key = (
        {}
        if args.force
        else load_existing_by_key(
            output_path,
            id_field=args.id_field,
            question=args.question,
            model=args.model,
            text_field=args.text_field,
            context_fields=args.context_fields,
        )
    )

    output_records: list[dict[str, Any]] = []
    todo_indices: list[int] = []

    for index, record in enumerate(records):
        key = record_key(record, index, args.id_field)
        if key in existing_by_key:
            output_records.append(existing_by_key[key])
        else:
            output_records.append(dict(record))
            todo_indices.append(index)

    print(
        f"{display_path(input_file)}: {len(records)} total | "
        f"{len(records) - len(todo_indices)} already done | {len(todo_indices)} to do"
    )

    if not todo_indices:
        write_json_atomic(output_path, output_records)
        print(f"Done. Results in {display_path(output_path)}")
        return

    checkpoint_every = max(1, args.checkpoint_every)
    completed_since_checkpoint = 0
    completed_total = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(
                classify_record,
                record=records[index],
                index=index,
                question=args.question,
                id_field=args.id_field,
                text_field=args.text_field,
                context_fields=args.context_fields,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                max_retries=args.max_retries,
            ): index
            for index in todo_indices
        }

        for future in as_completed(futures):
            index = futures[future]
            output_records[index] = future.result()
            completed_total += 1
            completed_since_checkpoint += 1

            if completed_since_checkpoint >= checkpoint_every:
                write_json_atomic(output_path, output_records)
                completed_since_checkpoint = 0

            if completed_total % 10 == 0 or completed_total == len(todo_indices):
                elapsed = max(time.time() - start, 0.001)
                rate = completed_total / elapsed
                eta_seconds = (len(todo_indices) - completed_total) / rate
                print(
                    f"  {completed_total}/{len(todo_indices)} "
                    f"({rate:.2f}/s, ETA {eta_seconds / 60:.1f} min)"
                )

    write_json_atomic(output_path, output_records)
    print(f"Done. Results in {display_path(output_path)}")


def main() -> None:
    args = parse_args()
    input_files = discover_input_files(args.input)

    print(f"Using Ollama model {args.model} at {args.base_url}")
    print(
        f"Using ID field '{args.id_field}', text field '{args.text_field}', "
        f"context fields {args.context_fields}"
    )
    for input_file in input_files:
        classify_file(input_file, args)


if __name__ == "__main__":
    main()
