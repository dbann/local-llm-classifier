# Local LLM text Classifier

This project is a general local setup for classifying text with local LLMs. 

Useful eg in (sytematic) reviews, metascience, or coding of text.

You can use multiple LLMs and check their speed / adjust concurrency setting before rolling across your entire sample. 

The current implementation uses Ollama. 

It reads JSON files from `data/input/`, sends the configured text field to a
local model, and writes classified JSON files to `data/output/`.

The default classifier question is:

```text
Does the following abstract make a policy claim? Answer yes or no.
```

The default model is `gemma3:270m`, chosen because it is fast and already small
enough for local iteration.

## Requirements

- Python 3.10 or newer
- Ollama running locally
- The selected Ollama model installed
- Optional: Jupyter, if you want to use the notebook workbook

Install the default model if needed:

```bash
ollama pull gemma3:270m
```

No Python package installation is required. The script only uses the Python
standard library.

## Input Data

Put JSON files in `data/input/`.

`data/` is listed in `.gitignore` so local datasets and generated classification
outputs are not committed by default.

The preferred input shape is a JSON array of records:

```json
[
  {
    "scopus_id": "SCOPUS_ID:123",
    "doi": "10.1234/example",
    "title": "Example title",
    "abstract": "Example abstract text..."
  }
]
```

By default, each record should have:

- `scopus_id`: used as the stable record ID
- `abstract`: the field sent to the LLM for review
- `title`: optional context sent alongside the abstract

The script preserves all original fields in the output JSON, but it does not
send every field to the LLM. Only `text_field` and `context_fields` are included
in the model prompt.

If a record has no value in the configured `text_field`, the output answer is
set to `unknown`.

The script also accepts a top-level JSON object if it contains one of these list
fields: `records`, `items`, `abstracts`, or `data`.

## Run

There are three ways to run the classifier:

| Method | Best for | File or command |
| --- | --- | --- |
| Notebook workbook | Easy interactive use | `classify_workbook.ipynb` |
| Config file | Repeatable runs | `python3 app.py --config classify_config.yaml` |
| Command line | Quick overrides | `python3 app.py --model gemma3:270m` |

## Notebook Workbook

Open `classify_workbook.ipynb` in Jupyter. Edit the settings cell:

```python
INPUT = "data/input"
OUTPUT_DIR = "data/output"
MODEL = "gemma3:270m"
BASE_URL = "http://localhost:11434"
QUESTION = "Does the following abstract make a policy claim? Answer yes or no."
ID_FIELD = "scopus_id"
TEXT_FIELD = "abstract"
CONTEXT_FIELDS = ["title"]
CONCURRENCY = 2
LIMIT = None
FORCE = False
```

Then run the cells from top to bottom. The notebook can:

- list installed Ollama models
- preview the input JSON files
- run classification through `app.py`
- report elapsed time, throughput, and rough estimates for larger sets
- summarize output counts by `classification.answer`

## Config File

For repeatable runs, edit `classify_config.yaml`:

```yaml
input: data/input
output_dir: data/output
model: gemma3:270m
base_url: http://localhost:11434
question: "Does the following abstract make a policy claim? Answer yes or no."
id_field: scopus_id
text_field: abstract
context_fields: title
concurrency: 2
limit:
force: false
```

Run it with:

```bash
python3 app.py --config classify_config.yaml
```

Command-line options override the config file. For example, this uses the config
file but only classifies the first 10 records:

```bash
python3 app.py --config classify_config.yaml --limit 10 --force
```

The YAML support is intentionally simple: use one `key: value` setting per line.

## Command Line

Classify every JSON file in `data/input/`:

```bash
python3 app.py
```

Classify one file:

```bash
python3 app.py --input data/input/example_abstracts.json
```

Try a small sample:

```bash
python3 app.py --limit 10 --force
```

Use another model:

```bash
python3 app.py --model qwen3:0.6b
```

Change the question:

```bash
python3 app.py --question "Does this abstract mention health inequality? Answer yes or no."
```

Change which JSON fields are used:

```bash
python3 app.py --id-field doi --text-field abstract --context-fields title,journal
```

## Output

Output files are written to `data/output/` using this naming pattern:

```text
data/output/<input_stem>_classified.json
```

Each output record preserves the original input fields and adds a
`classification` object:

```json
{
  "title": "Example title",
  "abstract": "Example abstract text...",
  "classification": {
    "answer": "yes",
    "label": "yes",
    "raw": "{\"label\":\"yes\"}",
    "question": "Does the following abstract make a policy claim? Answer yes or no.",
    "model": "gemma3:270m",
    "id_field": "scopus_id",
    "text_field": "abstract",
    "context_fields": ["title"],
    "record_key": "scopus_id:SCOPUS_ID:123",
    "classified_at": "2026-06-05T08:42:51+00:00"
  }
}
```

Use `classification.answer` as the answer to the prompt. It will be one of:

- `yes`
- `no`
- `unknown`

`classification.label` is kept as a duplicate for compatibility with earlier
output.

## Resuming

The classifier is resumable. If the output file already exists, records that
already have a completed `classification.answer` are skipped.

Reclassify everything from scratch with:

```bash
python3 app.py --force
```

## Parallelisation

Parallelisation is controlled by `--concurrency`.

```bash
python3 app.py --concurrency 2
```

The script sends multiple requests to Ollama at the same time using a thread
pool. Whether this improves throughput depends on the model, hardware, and how
Ollama schedules requests. On this machine, parallelisation did improve
throughput for `gemma3:270m`.

## Speed

Benchmarks below were run locally on 2026-06-05 with Ollama at
`http://localhost:11434` and model `gemma3:270m`.

| Run | Records | Concurrency | Throughput | Approx time | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Sample benchmark | 60 | 1 | 3.77 abstracts/s | 15.9 s | Temporary benchmark output |
| Sample benchmark | 60 | 2 | 5.33 abstracts/s | 11.3 s | Temporary benchmark output |

For the 60-record benchmark, `--concurrency 2` was about 1.4x faster than
`--concurrency 1`.

## Useful Options

| Option | Purpose |
| --- | --- |
| `--config` | Optional JSON or simple YAML config file. |
| `--input` | JSON file or directory of JSON files. Defaults to `data/input/`. |
| `--output-dir` | Output directory. Defaults to `data/output/`. |
| `--model` | Ollama model tag. Defaults to `gemma3:270m`. |
| `--base-url` | Ollama base URL. Defaults to `http://localhost:11434`. |
| `--question` | Classification question sent to the model. |
| `--id-field` | Record field used as the stable ID. Defaults to `scopus_id`. |
| `--text-field` | Record field sent to the LLM for classification. Defaults to `abstract`. |
| `--context-fields` | Comma-separated extra fields sent as context. Defaults to `title`. |
| `--concurrency` | Number of simultaneous Ollama requests. Defaults to `2`. |
| `--limit` | Classify only the first N records in each input file. |
| `--force` | Reclassify records even when output already exists. |
