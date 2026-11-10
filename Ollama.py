"""
Send each image in a folder to an Ollama vision model and save the replies to a JSON file.

Setup
-----
1. Install and run Ollama:  https://ollama.com/download
2. Pull the model once (from a terminal):
       ollama pull gemma3:27b        # a real local vision model, or
       ollama pull gemma4:31b-cloud  # the model you named (see note at bottom)
3. Install the Python client:
       pip install ollama

Usage
-----
    python ollama_vision.py path/to/image_folder [path/to/output.json]
"""

import argparse
import csv
import io
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import ollama
from PIL import Image

from prompts import (
    cross_dap_reid_prompt,
    cross_dap_reid_prompt_zeroshot,
    cross_dap_reid_schema,
    note_time_extraction_prompt,
    note_time_extraction_schema,
    tomato_growth_stage_prompt_AG,
    tomato_growth_stage_prompt_zeroshot,
    tomato_growth_stage_schema,
    tomato_ripeness_count_prompt,
    tomato_ripeness_count_prompt_zeroshot,
    tomato_ripeness_count_schema,
    tomato_ripeness_transition_prompt,
    tomato_ripeness_transition_prompt_zeroshot,
    tomato_ripeness_transition_schema,
    tomato_growth_transition_prompt,
    tomato_growth_transition_prompt_zeroshot,
    tomato_growth_transition_schema,
    growth_stage_set_name,
    growth_transition_name,
)

# Change this to whichever vision model you have pulled in Ollama.
MODEL = "qwen3-vl:4b"#"minicpm-v:8b"#"gemma4:cloud"#"gemma4:e4b"#"gemma4:cloud"#"gemma4:31b-cloud"

# Prompt builders per prompting condition. "agp" is attribute-guided prompting;
# "zeroshot" gives only the task and answer options. Both share the same JSON
# schemas, so their output CSVs have identical columns.
PROMPTS = {
    "agp": {
        "growth_stage": tomato_growth_stage_prompt_AG,
        "ripeness_count": tomato_ripeness_count_prompt,
        "reid": cross_dap_reid_prompt,
        "ripeness_transition": tomato_ripeness_transition_prompt,
        "growth_transition": tomato_growth_transition_prompt,
    },
    "zeroshot": {
        "growth_stage": tomato_growth_stage_prompt_zeroshot,
        "ripeness_count": tomato_ripeness_count_prompt_zeroshot,
        "reid": cross_dap_reid_prompt_zeroshot,
        "ripeness_transition": tomato_ripeness_transition_prompt_zeroshot,
        "growth_transition": tomato_growth_transition_prompt_zeroshot,
    },
}


def _prompt_fn(prompt_mode: str, task: str):
    """Return the prompt builder for `task` under `prompt_mode` ("agp" or "zeroshot")."""
    if prompt_mode not in PROMPTS:
        raise ValueError(f"prompt_mode must be one of {sorted(PROMPTS)}, got {prompt_mode!r}")
    return PROMPTS[prompt_mode][task]

# Ollama's default context window (4096 tokens) is too small once an image's
# vision tokens are added on top of the prompt text, so requests need a larger one.
NUM_CTX = 8192

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG"}

def _token_usage(response) -> dict:
    """Input/output token counts reported by Ollama for one chat response.

    `input_tokens` is Ollama's prompt_eval_count (prompt text + image tokens) and
    `output_tokens` its eval_count (generated tokens). Ollama may omit
    prompt_eval_count when the prompt was served from cache; the field is then "".
    """
    def _get(key):
        value = response.get(key) if isinstance(response, dict) else getattr(response, key, None)
        return "" if value is None else value

    return {"input_tokens": _get("prompt_eval_count"), "output_tokens": _get("eval_count")}


def ask_with_bytes(
    image_bytes: bytes, prompt: str, model: str = MODEL, format=None, usage: dict = None
) -> str:
    """Send `prompt` with one image and return the reply text. If `usage` is given,
    it is filled with the request's input_tokens/output_tokens."""
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_bytes],  # Ollama accepts bytes directly
            }
        ],
        format=format,
        options={"num_ctx": NUM_CTX},
    )
    if usage is not None:
        usage.update(_token_usage(response))
    return response["message"]["content"]

def process_image_bytes(image_bytes: bytes, model: str = MODEL) -> dict:
    prompt = tomato_growth_stage_prompt_AG()
    format = "json"

    try:
        answer = ask_with_bytes(image_bytes, prompt, model=model, format=format)
        return json.loads(_strip_code_fence(answer))
    except Exception as exc:
        return {"error": str(exc)}


def count_tomatoes_bytes(image_bytes: bytes, model: str = MODEL) -> dict:
    """Count raw/semi-ripe/ripe tomatoes in image bytes. See tomato_ripeness_count_prompt."""
    prompt = tomato_ripeness_count_prompt()
    format = "json"

    try:
        answer = ask_with_bytes(image_bytes, prompt, model=model, format=format)
        return json.loads(_strip_code_fence(answer))
    except Exception as exc:
        return {"error": str(exc)}


def ask_with_image(
    image_path: str, prompt: str, model: str = MODEL, format=None
) -> str:
    """Send `prompt` and the image at `image_path` to the model, return its text reply.

    `format` may be a JSON schema dict to have Ollama constrain the model's output
    to that structure (see https://ollama.com/blog/structured-outputs).
    """
    image_file = Path(image_path)
    if not image_file.is_file():
        raise FileNotFoundError(f"Image not found: {image_file}")

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                # The client reads the file and base64-encodes it for you.
                "images": [image_file]#[str(image_file)],
            }
        ],
        format=format,
        options={"num_ctx": NUM_CTX},
    )
    return response["message"]["content"]


def _strip_code_fence(text: str) -> str:
    """Strip a leading/trailing ```json ... ``` fence, if present, before JSON parsing."""
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return text.strip()

def process_image(image_path_str: str, model: str = MODEL) -> dict:
    """Run the tomato growth stage prompt on a single image at `image_path_str`.

    Returns a dict containing the parsed JSON result, or an "error" entry
    if the request failed or the reply wasn't valid JSON.
    """
    image_path = Path(image_path_str)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image file extension: {image_path.suffix}")

    prompt = tomato_growth_stage_prompt_AG()
    format = "json"

    print(f"Processing {image_path.name}...")
    try:
        answer = ask_with_image(image_path, prompt, model=model, format=format)
        return json.loads(_strip_code_fence(answer))
    except Exception as exc:
        return {"error": str(exc)}


def count_tomatoes(image_path_str: str, model: str = MODEL) -> dict:
    """Count raw/semi-ripe/ripe tomatoes in the image at `image_path_str`.

    Returns a dict with raw_count/semi_ripe_count/ripe_count/total_count/reasoning,
    or an "error" entry if the request failed or the reply wasn't valid JSON.
    """
    image_path = Path(image_path_str)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image file extension: {image_path.suffix}")

    prompt = tomato_ripeness_count_prompt()
    format = "json"

    print(f"Processing {image_path.name}...")
    try:
        answer = ask_with_image(image_path, prompt, model=model, format=format)
        return json.loads(_strip_code_fence(answer))
    except Exception as exc:
        return {"error": str(exc)}


def evaluate_ripeness_count(
    query_dap,
    output_csv: str,
    data_dir: str = "./data",
    labels_dir: str = "./output/labels_dap",
    id_col: str = "id",
    filename_col: str = "filename",
    angle_col: str = "angle",
    model: str = MODEL,
    prompt_mode: str = "agp",
) -> dict:
    """Run tomato_ripeness_count_prompt (or its zero-shot counterpart, per
    `prompt_mode`) once per filename in `query_dap`'s labels CSV, writing one row
    per trial to `output_csv`.

    Like count_tomatoes, but driven by a DAP's labels CSV (via _load_dap_rows/
    _load_dap_image_bytes) instead of a single path, so each output row also
    carries the plant `id`, `angle`, and DAP alongside the ripeness counts.
    There's no external ground truth for these counts, so this reports summed
    totals rather than accuracy — trials that error out (missing image, model
    failure, invalid JSON, etc.) are still written to the CSV with their error
    message in the `error` column, but excluded from the totals.

    Returns {"raw_total": int, "semi_ripe_total": int, "ripe_total": int,
    "grand_total": int, "num_trials": int, "num_errors": int}.
    """
    prompt = _prompt_fn(prompt_mode, "ripeness_count")()
    rows = _load_dap_rows(query_dap, labels_dir, id_col, filename_col, angle_col)
    if not rows:
        raise ValueError(f"No rows found for DAP {query_dap} in {labels_dir}")

    fieldnames = [
        "filename", "id", "angle", "dap",
        "raw_count", "semi_ripe_count", "ripe_count", "total_count", "reasoning", "error",
        "input_tokens", "output_tokens",
    ]

    results = []
    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            filename = row[filename_col]
            print(f"[{i}/{len(rows)}] Processing {filename}...")
            out_row = {
                "filename": filename,
                "id": row[id_col],
                "angle": row[angle_col],
                "dap": query_dap,
                "raw_count": "",
                "semi_ripe_count": "",
                "ripe_count": "",
                "total_count": "",
                "reasoning": "",
                "error": "",
                "input_tokens": "",
                "output_tokens": "",
            }
            try:
                image_bytes = _load_dap_image_bytes(query_dap, filename, data_dir)
                answer = ask_with_bytes(
                    image_bytes, prompt,
                    model=model, format=tomato_ripeness_count_schema(), usage=out_row,
                )
                parsed = json.loads(_strip_code_fence(answer))
                out_row["raw_count"] = parsed.get("raw_count", 0)
                out_row["semi_ripe_count"] = parsed.get("semi_ripe_count", 0)
                out_row["ripe_count"] = parsed.get("ripe_count", 0)
                out_row["total_count"] = parsed.get("total_count", 0)
                out_row["reasoning"] = parsed.get("reasoning", "")
            except Exception as exc:  # noqa: BLE001 - keep going with the rest of the DAP
                out_row["error"] = str(exc)

            results.append(out_row)
            writer.writerow(out_row)

    valid_results = [r for r in results if not r["error"]]
    num_errors = len(results) - len(valid_results)
    raw_total = sum(r["raw_count"] for r in valid_results)
    semi_ripe_total = sum(r["semi_ripe_count"] for r in valid_results)
    ripe_total = sum(r["ripe_count"] for r in valid_results)
    grand_total = raw_total + semi_ripe_total + ripe_total

    print(f"Saved {len(results)} trial(s) to {output_csv}")
    print(
        f"Totals — raw: {raw_total}, semi-ripe: {semi_ripe_total}, ripe: {ripe_total}, "
        f"grand total: {grand_total} ({len(valid_results)} valid trial(s), {num_errors} error(s))"
    )

    return {
        "raw_total": raw_total,
        "semi_ripe_total": semi_ripe_total,
        "ripe_total": ripe_total,
        "grand_total": grand_total,
        "num_trials": len(results),
        "num_errors": num_errors,
    }


def process_folder(folder_path: str, model: str = MODEL) -> dict:
    """Run the tomato growth stage prompt on every image in `folder_path`.

    Returns a dict mapping each image filename to its parsed JSON result, or to
    an "error" entry if the request failed or the reply wasn't valid JSON.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder}")

    prompt = tomato_growth_stage_prompt_AG()
    format = "json"

    results = {}
    image_paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    for image_path in image_paths:
        print(f"Processing {image_path.name}...")
        try:
            answer = ask_with_image(str(image_path), prompt, model=model, format=format)
            results[image_path.name] = json.loads(_strip_code_fence(answer))
        except Exception as exc:  # noqa: BLE001 - keep going with the rest of the folder
            results[image_path.name] = {"error": str(exc)}

    return results


def evaluate_growth_stage(
    query_dap,
    output_csv: str,
    data_dir: str = "./data",
    labels_dir: str = "./output/labels_dap",
    id_col: str = "id",
    filename_col: str = "filename",
    angle_col: str = "angle",
    model: str = MODEL,
    prompt_mode: str = "agp",
) -> dict:
    """Run tomato_growth_stage_prompt_AG (or its zero-shot counterpart, per
    `prompt_mode`) once per filename in `query_dap`'s labels CSV, writing one row
    per trial to `output_csv`.

    Like process_folder, but driven by a DAP's labels CSV (via _load_dap_rows/
    _load_dap_image_bytes) instead of listing a raw folder, so each output row
    also carries the plant `id`, `angle`, and DAP alongside the growth stage
    result. There's no external ground truth for growth stage, so this reports a
    stage-count breakdown rather than accuracy — trials that error out (missing
    image, model failure, invalid JSON, etc.) are still written to the CSV with
    their error message in the `error` column, but excluded from that breakdown.

    Returns {"stage_counts": dict, "num_trials": int, "num_errors": int}.
    """
    prompt = _prompt_fn(prompt_mode, "growth_stage")()
    rows = _load_dap_rows(query_dap, labels_dir, id_col, filename_col, angle_col)
    if not rows:
        raise ValueError(f"No rows found for DAP {query_dap} in {labels_dir}")

    fieldnames = [
        "filename", "id", "angle", "dap",
        "growth_stage_numbers", "growth_stage_names", "reasoning", "error",
        "input_tokens", "output_tokens",
    ]

    results = []
    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            filename = row[filename_col]
            print(f"[{i}/{len(rows)}] Processing {filename}...")
            out_row = {
                "filename": filename,
                "id": row[id_col],
                "angle": row[angle_col],
                "dap": query_dap,
                "growth_stage_numbers": "",
                "growth_stage_names": "",
                "reasoning": "",
                "error": "",
                "input_tokens": "",
                "output_tokens": "",
            }
            try:
                image_bytes = _load_dap_image_bytes(query_dap, filename, data_dir)
                answer = ask_with_bytes(
                    image_bytes, prompt,
                    model=model, format=tomato_growth_stage_schema(), usage=out_row,
                )
                print(answer)
                parsed = json.loads(_strip_code_fence(answer))
                stages = parsed.get("growth_stages", [])
                out_row["growth_stage_numbers"] = ", ".join(
                    str(s.get("growth_stage_number", "")) for s in stages
                )
                out_row["growth_stage_names"] = ", ".join(
                    s.get("growth_stage_name", "") for s in stages
                )
                out_row["reasoning"] = parsed.get("reasoning", "")
            except Exception as exc:  # noqa: BLE001 - keep going with the rest of the DAP
                out_row["error"] = str(exc)

            results.append(out_row)
            writer.writerow(out_row)

    valid_results = [r for r in results if not r["error"]]
    num_errors = len(results) - len(valid_results)
    stage_counts = dict(Counter(
        name.strip()
        for r in valid_results
        for name in r["growth_stage_names"].split(",")
        if name.strip()
    ))

    print(f"Saved {len(results)} trial(s) to {output_csv}")
    print(f"Stage counts: {stage_counts} ({len(valid_results)} valid trial(s), {num_errors} error(s))")

    return {
        "stage_counts": stage_counts,
        "num_trials": len(results),
        "num_errors": num_errors,
    }


def extract_notes_and_times_to_csv(
    folder_path: str, csv_path: str, model: str = MODEL
) -> None:
    """Extract the Note/Time overlay from every image in `folder_path` and write it to a CSV.

    Each image is first sent to the model with `note_time_extraction_prompt()` to get a
    JSON object of the form {"Note": ..., "Time": ...}, which is then flattened into a
    row of the output CSV (columns: filename, Note, Time, error).
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder}")

    prompt = note_time_extraction_prompt()
    format = "json"

    image_paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["filename", "Note", "Time", "error"])
        writer.writeheader()
        for image_path in image_paths:
            print(f"Processing {image_path.name}...")
            row = {"filename": image_path.name, "Note": "", "Time": "", "error": ""}
            try:
                answer = ask_with_image(str(image_path), prompt, model=model, format=format)
                data = json.loads(_strip_code_fence(answer))
                row["Note"] = data.get("Note", "")
                row["Time"] = data.get("Time", "")
            except Exception as exc:  # noqa: BLE001 - keep going with the rest of the folder
                row["error"] = str(exc)
            writer.writerow(row)

    print(f"Saved results for {len(image_paths)} image(s) to {csv_path}")


def _discover_daps(labels_dir: str = "./output/labels_dap", allowed_daps=None) -> list:
    """Return every DAP value with a labels CSV under `labels_dir`, named either like
    '54_notes_times.csv' or plainly '30.csv' (as written by make_dap_labels.py).

    If `allowed_daps` is given, only DAPs in that list are returned, so other label
    files in the folder are ignored.
    """
    allowed = {int(d) for d in allowed_daps} if allowed_daps is not None else None
    daps = set()
    for path in Path(labels_dir).glob("*.csv"):
        match = re.match(r"^(\d+)(_|\.csv$)", path.name)
        if match and (allowed is None or int(match.group(1)) in allowed):
            daps.add(int(match.group(1)))
    return sorted(daps)


def _find_dap_csv(dap, labels_dir: str = "./output/labels_dap") -> Path:
    """Locate the labels CSV for DAP `dap` under `labels_dir`: '<dap>.csv' if it
    exists, otherwise the first '<dap>_*.csv'."""
    exact = Path(labels_dir) / f"{dap}.csv"
    if exact.is_file():
        return exact
    matches = sorted(Path(labels_dir).glob(f"{dap}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No labels CSV for DAP {dap} found under {labels_dir}")
    return matches[0]


def _find_dap_image_dir(dap, data_dir: str = "./data") -> Path:
    """Locate the image folder under `data_dir` for DAP number `dap`.

    Matches folders whose name starts with the DAP number (e.g. '54 DAP') or that
    contain '<dap> DAP' anywhere (e.g. 'ISRO 30 DAP 22.11.2023'). The number must
    not be part of a longer one, so DAP 30 never matches '130 DAP'.
    """
    dap_str = re.escape(str(dap))
    patterns = [
        re.compile(rf"^{dap_str}(\D|$)"),
        re.compile(rf"(?<!\d){dap_str}\s*DAP\b", re.IGNORECASE),
    ]
    for entry in sorted(Path(data_dir).iterdir()):
        if entry.is_dir() and any(p.search(entry.name) for p in patterns):
            return entry
    raise FileNotFoundError(f"No image folder for DAP {dap} found under {data_dir}")


def _load_dap_rows(
    dap,
    labels_dir: str = "./output/labels_dap",
    id_col: str = "id",
    filename_col: str = "filename",
    angle_col: str = "angle",
) -> list:
    """Read DAP `dap`'s labels CSV into a list of row dicts, keyed by `id_col`/`filename_col`/`angle_col`."""
    csv_path = _find_dap_csv(dap, labels_dir)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    missing = [col for col in (id_col, filename_col, angle_col) if rows and col not in rows[0]]
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s): {missing}")

    return rows


REID_MAX_IMAGE_DIMENSION = 800


def _load_dap_image_bytes(dap, filename: str, data_dir: str = "./data") -> bytes:
    """Read `filename` from the image folder matching DAP `dap`, downscaled to JPEG.

    cross_dap_reid bundles the query plus every candidate image into a single
    request, so full-resolution originals (several MB each) can push the request
    past Ollama's body-size limit. Downscaling keeps the request small while
    preserving enough detail for structural comparison.
    """
    image_path = _find_dap_image_dir(dap, data_dir) / filename
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((REID_MAX_IMAGE_DIMENSION, REID_MAX_IMAGE_DIMENSION))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


REID_NUM_DISTRACTORS = 3


def _sample_unique_ids(rows: list, k: int, id_col: str = "id") -> list:
    """Randomly sample up to `k` rows from `rows` such that no two share the same `id_col`.

    Draws rows in random order; a draw whose id was already taken is discarded and
    the next one is drawn instead, so labels CSVs where several images share an id
    (e.g. multiple plants per plot) never yield two candidates with the same id.
    Returns fewer than `k` rows if `rows` has fewer than `k` distinct ids.
    """
    shuffled = random.sample(rows, len(rows))
    sampled, seen_ids = [], set()
    for row in shuffled:
        if row[id_col] in seen_ids:
            continue
        seen_ids.add(row[id_col])
        sampled.append(row)
        if len(sampled) == k:
            break
    return sampled


def _cross_dap_reid_for_query_row(
    query_row: dict,
    query_dap,
    angle_mode: str = "same",
    data_dir: str = "./data",
    labels_dir: str = "./output/labels_dap",
    id_col: str = "id",
    filename_col: str = "filename",
    angle_col: str = "angle",
    model: str = MODEL,
    prompt_mode: str = "agp",
    allowed_daps=None,
) -> dict:
    """Run one cross-DAP re-id trial using `query_row` (a row dict from _load_dap_rows)
    as the query. Shared by cross_dap_reid (random query) and evaluate_cross_dap_reid
    (every row in a DAP, one trial each). See cross_dap_reid for full documentation.
    """
    try:
        trial = _sample_reid_candidates(
            query_row, query_dap, angle_mode, labels_dir, id_col, filename_col, angle_col, allowed_daps,
        )
        candidate_rows = [trial["true_match_row"], *trial["distractors"]]
        random.shuffle(candidate_rows)
        return _ask_reid(trial, candidate_rows, data_dir, id_col, filename_col, angle_col, model, prompt_mode)
    except Exception as exc:
        return {"error": str(exc), "query_filename": query_row.get(filename_col)}


def _sample_reid_candidates(
    query_row: dict,
    query_dap,
    angle_mode: str = "same",
    labels_dir: str = "./output/labels_dap",
    id_col: str = "id",
    filename_col: str = "filename",
    angle_col: str = "angle",
    allowed_daps=None,
) -> dict:
    """Pick the candidate DAP, candidate angle, true-match row and distractor rows for
    one re-id trial with `query_row` as the query (see cross_dap_reid for the rules).

    Returns a dict with the query row/DAP, `candidate_dap`, `candidate_angle`,
    `angle_mode`, `true_match_row` and `distractors`; candidate ORDER is left to the
    caller. Raises ValueError if no other DAP has the query plant at a qualifying angle.
    """
    if angle_mode not in ("same", "different"):
        raise ValueError(f"angle_mode must be 'same' or 'different', got {angle_mode!r}")

    query_id = query_row[id_col]
    query_angle = query_row[angle_col]

    def _qualifying_angles(rows):
        """Angles at which `query_id` appears in `rows`, filtered by `angle_mode`."""
        angles = {row[angle_col] for row in rows if row[id_col] == query_id}
        return (angles & {query_angle}) if angle_mode == "same" else (angles - {query_angle})

    other_daps = [d for d in _discover_daps(labels_dir, allowed_daps) if d != int(query_dap)]
    daps_with_match = [
        d for d in other_daps
        if _qualifying_angles(_load_dap_rows(d, labels_dir, id_col, filename_col, angle_col))
    ]
    if not daps_with_match:
        angle_desc = (
            f"the same angle ('{query_angle}')" if angle_mode == "same"
            else f"a different angle than '{query_angle}'"
        )
        raise ValueError(
            f"No other DAP under {labels_dir} has id '{query_id}' at {angle_desc}"
        )
    candidate_dap = random.choice(daps_with_match)

    candidate_dap_rows = _load_dap_rows(candidate_dap, labels_dir, id_col, filename_col, angle_col)
    candidate_angle = random.choice(sorted(_qualifying_angles(candidate_dap_rows)))

    candidate_rows = [row for row in candidate_dap_rows if row[angle_col] == candidate_angle]
    # Several images can share an id (e.g. plots holding more than one plant),
    # so pick the true match at random among them rather than always the first.
    true_match_row = random.choice([row for row in candidate_rows if row[id_col] == query_id])
    distractor_pool = [row for row in candidate_rows if row[id_col] != query_id]
    distractors = _sample_unique_ids(distractor_pool, REID_NUM_DISTRACTORS, id_col)
    if len(distractors) < REID_NUM_DISTRACTORS:
        print(
            f"Only {len(distractors)} distractor(s) with distinct ids available at angle "
            f"'{candidate_angle}' in DAP {candidate_dap} (wanted {REID_NUM_DISTRACTORS})"
        )

    return {
        "query_row": query_row,
        "query_dap": query_dap,
        "candidate_dap": candidate_dap,
        "candidate_angle": candidate_angle,
        "angle_mode": angle_mode,
        "true_match_row": true_match_row,
        "distractors": distractors,
    }


def _ask_reid(
    trial: dict,
    candidate_rows: list,
    data_dir: str = "./data",
    id_col: str = "id",
    filename_col: str = "filename",
    angle_col: str = "angle",
    model: str = MODEL,
    prompt_mode: str = "agp",
) -> dict:
    """Send the query plus `candidate_rows` (in the given order) to the model and
    return its parsed verdict, augmented with the trial's ground truth."""
    reid_prompt = _prompt_fn(prompt_mode, "reid")
    query_row, query_dap = trial["query_row"], trial["query_dap"]
    candidate_dap = trial["candidate_dap"]
    query_id = query_row[id_col]

    query_bytes = _load_dap_image_bytes(query_dap, query_row[filename_col], data_dir)
    print(f"Query image: {query_row[filename_col]} (id={query_id}, angle={query_row[angle_col]}, DAP={query_dap})")
    for i, row in enumerate(candidate_rows, start=1):
        print(f"Candidate {i}: {row[filename_col]} (id={row[id_col]}, angle={row[angle_col]}, DAP={candidate_dap})")
    candidate_images = [
        _load_dap_image_bytes(candidate_dap, row[filename_col], data_dir)
        for row in candidate_rows
    ]
    true_index = next(
        i for i, row in enumerate(candidate_rows, start=1) if row is trial["true_match_row"]
    )

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": reid_prompt(len(candidate_rows)),
                "images": [query_bytes, *candidate_images],
            }
        ],
        format=cross_dap_reid_schema(len(candidate_rows)),
        options={"num_ctx": NUM_CTX},
    )
    result = json.loads(_strip_code_fence(response["message"]["content"]))
    result.update(_token_usage(response))

    result["query_filename"] = query_row[filename_col]
    result["query_id"] = query_id
    result["query_angle"] = query_row[angle_col]
    result["query_dap"] = query_dap
    result["candidate_dap"] = candidate_dap
    result["candidate_angle"] = trial["candidate_angle"]
    result["angle_mode"] = trial["angle_mode"]
    result["true_candidate_index"] = true_index
    result["correct"] = result.get("matching_candidate_index") == true_index
    # Which image (not slot) the model picked, so answers can be compared across orderings.
    picked = result.get("matching_candidate_index")
    result["picked_filename"] = (
        candidate_rows[picked - 1][filename_col]
        if isinstance(picked, int) and 1 <= picked <= len(candidate_rows) else ""
    )
    result["candidate_order"] = ";".join(row[filename_col] for row in candidate_rows)
    return result


def cross_dap_reid(
    query_dap,
    angle_mode: str = "same",
    data_dir: str = "./data",
    labels_dir: str = "./output/labels_dap",
    id_col: str = "id",
    filename_col: str = "filename",
    angle_col: str = "angle",
    model: str = MODEL,
    prompt_mode: str = "agp",
    allowed_daps=None,
) -> dict:
    """Test whether the model can re-identify a tomato plant across DAP snapshots.

    `angle_mode` controls how the candidate pool's camera angle relates to the
    query's:
      - "same" (default): every candidate is photographed from the SAME angle as
        the query, isolating identity from viewpoint changes.
      - "different": every candidate is photographed from a DIFFERENT angle than
        the query, testing whether identity holds when both DAP and viewpoint
        change together.

    Picks a random plant image from `query_dap` as the query, then a random OTHER
    DAP that has the same plant `id` available at a qualifying angle (per
    `angle_mode`), and a random qualifying angle within that DAP as the candidate
    angle. The candidate set is fixed at REID_NUM_DISTRACTORS + 1 images: the
    true-match row (same `id`, at the candidate angle) plus REID_NUM_DISTRACTORS
    other plants at that same angle, sampled at random (fewer if not enough
    exist). Sends the query + candidates to the model with cross_dap_reid_prompt
    and returns its parsed verdict, augmented with ground truth (`query_filename`,
    `query_id`, `query_angle`, `query_dap`, `candidate_dap`, `candidate_angle`,
    `angle_mode`, `true_candidate_index`, `correct`), or an "error" entry if
    anything failed.
    """
    try:
        query_rows = _load_dap_rows(query_dap, labels_dir, id_col, filename_col, angle_col)
        if not query_rows:
            raise ValueError(f"No rows found for DAP {query_dap} in {labels_dir}")
        query_row = random.choice(query_rows)
    except Exception as exc:
        return {"error": str(exc)}

    return _cross_dap_reid_for_query_row(
        query_row, query_dap, angle_mode, data_dir, labels_dir, id_col, filename_col, angle_col, model,
        prompt_mode, allowed_daps,
    )


def _accuracy_and_macro_f1(y_true: list, y_pred: list) -> tuple:
    """Compute accuracy and macro-averaged F1 for parallel lists of true/predicted labels.

    Macro-F1 averages per-class F1 (each distinct label seen in y_true or y_pred is
    one class) rather than pooling all trials into one binary count, so it can
    reveal e.g. a positional bias (the model favoring one candidate index) that
    plain accuracy would hide.
    """
    if not y_true:
        return 0.0, 0.0

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    labels = set(y_true) | set(y_pred)
    f1_scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    return accuracy, macro_f1


def evaluate_cross_dap_reid(
    query_dap,
    output_csv: str,
    angle_mode: str = "same",
    data_dir: str = "./data",
    labels_dir: str = "./output/labels_dap",
    id_col: str = "id",
    filename_col: str = "filename",
    angle_col: str = "angle",
    model: str = MODEL,
    prompt_mode: str = "agp",
    allowed_daps=None,
) -> dict:
    """Run cross_dap_reid once per filename in `query_dap`'s labels CSV, writing
    one row per trial to `output_csv`, then report accuracy and macro-F1.

    Unlike cross_dap_reid (which picks one random query per call), this runs every
    row in `query_dap` as its own query exactly once, so results cover the whole
    DAP rather than a single sample. Trials that error out (missing image, no
    qualifying candidate DAP, model failure, etc.) are still written to the CSV
    with their error message in the `error` column, but excluded from the metrics.

    Returns {"accuracy": float, "f1": float, "num_trials": int, "num_errors": int}.
    """
    query_rows = _load_dap_rows(query_dap, labels_dir, id_col, filename_col, angle_col)
    if not query_rows:
        raise ValueError(f"No rows found for DAP {query_dap} in {labels_dir}")

    fieldnames = [
        "query_filename", "query_id", "query_angle", "query_dap",
        "candidate_dap", "candidate_angle", "angle_mode",
        "matching_candidate_index", "true_candidate_index", "confidence",
        "correct", "reasoning", "error", "input_tokens", "output_tokens",
    ]

    results = []
    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i, query_row in enumerate(query_rows, start=1):
            print(f"[{i}/{len(query_rows)}] Evaluating {query_row[filename_col]}...")
            result = _cross_dap_reid_for_query_row(
                query_row, query_dap, angle_mode, data_dir, labels_dir,
                id_col, filename_col, angle_col, model, prompt_mode, allowed_daps,
            )
            results.append(result)
            writer.writerow({field: result.get(field, "") for field in fieldnames})

    valid_results = [r for r in results if "error" not in r]
    num_errors = len(results) - len(valid_results)

    y_true = [r["true_candidate_index"] for r in valid_results]
    y_pred = [r.get("matching_candidate_index") for r in valid_results]
    accuracy, f1 = _accuracy_and_macro_f1(y_true, y_pred)

    print(f"Saved {len(results)} trial(s) to {output_csv}")
    print(
        f"Accuracy: {accuracy:.3f}, Macro-F1: {f1:.3f} "
        f"({len(valid_results)} valid trial(s), {num_errors} error(s))"
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "num_trials": len(results),
        "num_errors": num_errors,
    }


REID_NUM_SHUFFLES = 3


def _shuffled_orders(true_match_row: dict, distractors: list, num_shuffles: int) -> list:
    """Return `num_shuffles` candidate orderings of the same candidate set.

    The true match is placed in a different slot in each ordering (as long as there
    are at least as many slots as shuffles), and the distractors are shuffled around
    it, so a model that favours one slot cannot be right in every ordering by chance.
    """
    num_slots = len(distractors) + 1
    if num_shuffles <= num_slots:
        true_slots = random.sample(range(num_slots), num_shuffles)
    else:
        true_slots = [random.randrange(num_slots) for _ in range(num_shuffles)]
    orders = []
    for slot in true_slots:
        others = random.sample(distractors, len(distractors))
        orders.append(others[:slot] + [true_match_row] + others[slot:])
    return orders


def evaluate_cross_dap_reid_shuffled(
    query_dap,
    output_csv: str,
    num_shuffles: int = REID_NUM_SHUFFLES,
    angle_mode: str = "same",
    data_dir: str = "./data",
    labels_dir: str = "./output/labels_dap",
    id_col: str = "id",
    filename_col: str = "filename",
    angle_col: str = "angle",
    model: str = MODEL,
    prompt_mode: str = "agp",
    allowed_daps=None,
) -> dict:
    """Re-identification with the candidate order shuffled `num_shuffles` times.

    For every query in `query_dap`, the candidate set (true match + distractors) is
    sampled ONCE and then presented to the model `num_shuffles` times, each time in a
    different order with the true match in a different slot. One CSV row is written
    per (query, shuffle), with a `shuffle` column (1..num_shuffles).

    Accuracy and macro-F1 over the candidate index are computed separately for each
    shuffle round and then averaged. Also reported: `consistency`, the share of
    queries for which the model picked the same image in every ordering. A
    position-invariant model is consistent; one driven by slot position is not.
    Trials that error out are written with their error message and excluded.

    Returns {"accuracy": mean, "f1": mean, "accuracy_per_shuffle": [...],
    "f1_per_shuffle": [...], "consistency": float, "num_queries": int, "num_errors": int}.
    """
    query_rows = _load_dap_rows(query_dap, labels_dir, id_col, filename_col, angle_col)
    if not query_rows:
        raise ValueError(f"No rows found for DAP {query_dap} in {labels_dir}")

    fieldnames = [
        "query_filename", "query_id", "query_angle", "query_dap",
        "candidate_dap", "candidate_angle", "angle_mode", "shuffle", "candidate_order",
        "matching_candidate_index", "true_candidate_index", "picked_filename", "confidence",
        "correct", "reasoning", "error", "input_tokens", "output_tokens",
    ]

    results = []  # one entry per (query, shuffle)
    picks_per_query = []  # filenames picked across shuffles, for queries with no errors
    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i, query_row in enumerate(query_rows, start=1):
            print(f"[{i}/{len(query_rows)}] Evaluating {query_row[filename_col]} ({num_shuffles} orderings)...")
            try:
                trial = _sample_reid_candidates(
                    query_row, query_dap, angle_mode, labels_dir, id_col, filename_col, angle_col, allowed_daps,
                )
                orders = _shuffled_orders(trial["true_match_row"], trial["distractors"], num_shuffles)
            except Exception as exc:  # noqa: BLE001 - no candidate set for this query
                orders, trial_error = [], str(exc)
            else:
                trial_error = ""

            query_results = []
            for shuffle in range(1, num_shuffles + 1):
                if trial_error:
                    result = {"error": trial_error, "query_filename": query_row.get(filename_col)}
                else:
                    try:
                        result = _ask_reid(
                            trial, orders[shuffle - 1], data_dir, id_col, filename_col, angle_col,
                            model, prompt_mode,
                        )
                    except Exception as exc:  # noqa: BLE001 - keep going with the other orderings
                        result = {"error": str(exc), "query_filename": query_row.get(filename_col)}
                result["shuffle"] = shuffle
                query_results.append(result)
                writer.writerow({field: result.get(field, "") for field in fieldnames})
            results.extend(query_results)
            if all("error" not in r for r in query_results):
                picks_per_query.append([r["picked_filename"] for r in query_results])

    accuracy_per_shuffle, f1_per_shuffle = [], []
    for shuffle in range(1, num_shuffles + 1):
        valid = [r for r in results if r["shuffle"] == shuffle and "error" not in r]
        acc, f1 = _accuracy_and_macro_f1(
            [r["true_candidate_index"] for r in valid],
            [r.get("matching_candidate_index") for r in valid],
        )
        accuracy_per_shuffle.append(acc)
        f1_per_shuffle.append(f1)

    accuracy = sum(accuracy_per_shuffle) / num_shuffles
    f1 = sum(f1_per_shuffle) / num_shuffles
    consistency = (
        sum(1 for picks in picks_per_query if len(set(picks)) == 1) / len(picks_per_query)
        if picks_per_query else 0.0
    )
    num_errors = sum(1 for r in results if "error" in r)

    print(f"Saved {len(results)} trial(s) ({len(query_rows)} queries x {num_shuffles} orderings) to {output_csv}")
    for shuffle, (acc, f) in enumerate(zip(accuracy_per_shuffle, f1_per_shuffle), start=1):
        print(f"  Shuffle {shuffle}: accuracy {acc:.3f}, macro-F1 {f:.3f}")
    print(
        f"Average over {num_shuffles} orderings: accuracy {accuracy:.3f}, macro-F1 {f1:.3f}; "
        f"consistency {consistency:.3f} ({num_errors} error(s))"
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "accuracy_per_shuffle": accuracy_per_shuffle,
        "f1_per_shuffle": f1_per_shuffle,
        "consistency": consistency,
        "num_queries": len(query_rows),
        "num_errors": num_errors,
    }


def _detect_ripeness_transition_for_row(
    first_row: dict,
    plant_dap,
    angle_mode: str = "same",
    data_dir: str = "./data",
    labels_dir: str = "./output/labels_dap",
    id_col: str = "id",
    filename_col: str = "filename",
    angle_col: str = "angle",
    model: str = MODEL,
    prompt_mode: str = "agp",
    allowed_daps=None,
) -> dict:
    """Run one ripeness-transition trial using `first_row` (a row dict from
    _load_dap_rows) as one endpoint. Shared by detect_ripeness_transition (random
    row) and evaluate_ripeness_transition (every row in a DAP, one trial each).
    See detect_ripeness_transition for full documentation.
    """
    if angle_mode not in ("same", "different"):
        raise ValueError(f"angle_mode must be 'same' or 'different', got {angle_mode!r}")
    transition_prompt = _prompt_fn(prompt_mode, "ripeness_transition")

    try:
        plant_id = first_row[id_col]
        first_angle = first_row[angle_col]

        def _qualifying_angles(rows):
            """Angles at which `plant_id` appears in `rows`, filtered by `angle_mode`."""
            angles = {row[angle_col] for row in rows if row[id_col] == plant_id}
            return (angles & {first_angle}) if angle_mode == "same" else (angles - {first_angle})

        other_daps = [d for d in _discover_daps(labels_dir, allowed_daps) if d != int(plant_dap)]
        daps_with_match = [
            d for d in other_daps
            if _qualifying_angles(_load_dap_rows(d, labels_dir, id_col, filename_col, angle_col))
        ]
        if not daps_with_match:
            angle_desc = (
                f"the same angle ('{first_angle}')" if angle_mode == "same"
                else f"a different angle than '{first_angle}'"
            )
            raise ValueError(
                f"No other DAP under {labels_dir} has id '{plant_id}' at {angle_desc}"
            )
        second_dap = random.choice(daps_with_match)

        second_dap_rows = _load_dap_rows(second_dap, labels_dir, id_col, filename_col, angle_col)
        second_angle = random.choice(sorted(_qualifying_angles(second_dap_rows)))
        # Several images can share an id and angle (e.g. plots holding more than one
        # plant), so pick one at random rather than always the first.
        second_row = random.choice([
            row for row in second_dap_rows
            if row[id_col] == plant_id and row[angle_col] == second_angle
        ])

        # Order chronologically: the smaller DAP value happened earlier.
        if int(plant_dap) <= int(second_dap):
            earlier_dap, earlier_row = plant_dap, first_row
            later_dap, later_row = second_dap, second_row
        else:
            earlier_dap, earlier_row = second_dap, second_row
            later_dap, later_row = plant_dap, first_row

        earlier_bytes = _load_dap_image_bytes(earlier_dap, earlier_row[filename_col], data_dir)
        later_bytes = _load_dap_image_bytes(later_dap, later_row[filename_col], data_dir)
        print(
            f"Earlier image: {earlier_row[filename_col]} (id={plant_id}, "
            f"angle={earlier_row[angle_col]}, DAP={earlier_dap})"
        )
        print(
            f"Later image: {later_row[filename_col]} (id={plant_id}, "
            f"angle={later_row[angle_col]}, DAP={later_dap})"
        )

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": transition_prompt(earlier_dap, later_dap),
                    "images": [earlier_bytes, later_bytes],
                }
            ],
            format=tomato_ripeness_transition_schema(),
            options={"num_ctx": NUM_CTX},
        )
        result = json.loads(_strip_code_fence(response["message"]["content"]))
        result.update(_token_usage(response))

        result["plant_id"] = plant_id
        result["earlier_dap"] = earlier_dap
        result["earlier_angle"] = earlier_row[angle_col]
        result["earlier_filename"] = earlier_row[filename_col]
        result["later_dap"] = later_dap
        result["later_angle"] = later_row[angle_col]
        result["later_filename"] = later_row[filename_col]
        result["angle_mode"] = angle_mode

        return result
    except Exception as exc:
        return {"error": str(exc), "first_filename": first_row.get(filename_col)}


def detect_ripeness_transition(
    plant_dap,
    angle_mode: str = "same",
    data_dir: str = "./data",
    labels_dir: str = "./output/labels_dap",
    id_col: str = "id",
    filename_col: str = "filename",
    angle_col: str = "angle",
    model: str = MODEL,
    prompt_mode: str = "agp",
    allowed_daps=None,
) -> dict:
    """Test whether the model can articulate the directional ripeness change of a
    tomato plant between two DAP snapshots.

    Picks a random plant image from `plant_dap`, then a random OTHER DAP with the
    same plant `id` available at a qualifying angle: `angle_mode="same"` (default)
    requires the same camera angle in both images, `"different"` requires a
    different angle. The two DAP values are then ordered chronologically (the
    smaller DAP is "earlier") and sent to the model as Image 1/Image 2 with
    prompts.tomato_ripeness_transition_prompt, which asks it to name each image's
    predominant ripeness stage and classify the transition between them. Returns
    the parsed verdict augmented with (`plant_id`, `earlier_dap`, `earlier_angle`,
    `earlier_filename`, `later_dap`, `later_angle`, `later_filename`,
    `angle_mode`), or an "error" entry if anything failed.
    """
    try:
        first_rows = _load_dap_rows(plant_dap, labels_dir, id_col, filename_col, angle_col)
        if not first_rows:
            raise ValueError(f"No rows found for DAP {plant_dap} in {labels_dir}")
        first_row = random.choice(first_rows)
    except Exception as exc:
        return {"error": str(exc)}

    return _detect_ripeness_transition_for_row(
        first_row, plant_dap, angle_mode, data_dir, labels_dir, id_col, filename_col, angle_col, model,
        prompt_mode, allowed_daps,
    )


_STAGE_RANK = {"Raw": 0, "Semi-ripe": 1, "Ripe": 2}


def evaluate_ripeness_transition(
    query_dap,
    output_csv: str,
    angle_mode: str = "same",
    data_dir: str = "./data",
    labels_dir: str = "./output/labels_dap",
    id_col: str = "id",
    filename_col: str = "filename",
    angle_col: str = "angle",
    model: str = MODEL,
    prompt_mode: str = "agp",
    allowed_daps=None,
) -> dict:
    """Run detect_ripeness_transition once per filename in `query_dap`'s labels CSV,
    writing one row per trial to `output_csv`, then report self-consistency.

    Unlike evaluate_cross_dap_reid, there is no external ground truth for ripeness
    (no one hand-labeled the "true" stage per image), so this can't report
    accuracy/F1. Instead it reports how often the model's own answer is internally
    consistent: ripening should never look like it went backwards between an
    earlier and later DAP under normal conditions, so a trial is flagged
    consistent when `transition_type` isn't "Regression" AND the later stage's
    rank (Raw < Semi-ripe < Ripe) is >= the earlier stage's rank. This surfaces
    self-contradictory or biologically implausible answers, not correctness.

    Trials that error out (missing image, no qualifying candidate DAP, model
    failure, etc.) are still written to the CSV with their error message in the
    `error` column, but excluded from the consistency rate.

    Returns {"consistency_rate": float, "transition_counts": dict, "num_trials": int,
    "num_errors": int}.
    """
    query_rows = _load_dap_rows(query_dap, labels_dir, id_col, filename_col, angle_col)
    if not query_rows:
        raise ValueError(f"No rows found for DAP {query_dap} in {labels_dir}")

    fieldnames = [
        "plant_id", "earlier_dap", "earlier_angle", "earlier_filename",
        "later_dap", "later_angle", "later_filename", "angle_mode",
        "earlier_stage", "later_stage", "transition_type", "stage_order_consistent",
        "reasoning", "error", "input_tokens", "output_tokens",
    ]

    results = []
    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(query_rows, start=1):
            print(f"[{i}/{len(query_rows)}] Evaluating {row[filename_col]}...")
            result = _detect_ripeness_transition_for_row(
                row, query_dap, angle_mode, data_dir, labels_dir,
                id_col, filename_col, angle_col, model, prompt_mode, allowed_daps,
            )
            if "error" not in result:
                earlier_rank = _STAGE_RANK.get(result.get("earlier_stage"))
                later_rank = _STAGE_RANK.get(result.get("later_stage"))
                result["stage_order_consistent"] = (
                    result.get("transition_type") != "Regression"
                    and earlier_rank is not None
                    and later_rank is not None
                    and later_rank >= earlier_rank
                )
            results.append(result)
            writer.writerow({field: result.get(field, "") for field in fieldnames})

    valid_results = [r for r in results if "error" not in r]
    num_errors = len(results) - len(valid_results)

    consistency_rate = (
        sum(1 for r in valid_results if r["stage_order_consistent"]) / len(valid_results)
        if valid_results else 0.0
    )
    transition_counts = dict(Counter(r["transition_type"] for r in valid_results))

    print(f"Saved {len(results)} trial(s) to {output_csv}")
    print(
        f"Self-consistency: {consistency_rate:.3f} "
        f"({len(valid_results)} valid trial(s), {num_errors} error(s))"
    )
    print(f"Transition type counts: {transition_counts}")

    return {
        "consistency_rate": consistency_rate,
        "transition_counts": transition_counts,
        "num_trials": len(results),
        "num_errors": num_errors,
    }


GROWTH_TRANSITION_MIN_SUPPORT = 5
DEFAULT_PAIRS_DIR = Path(__file__).resolve().parent / "ground_truth"


def macro_f1_min_support(y_true: list, y_pred: list, min_support: int = GROWTH_TRANSITION_MIN_SUPPORT) -> tuple:
    """Macro-F1 over the classes that occur at least `min_support` times in `y_true`.

    Rare classes are left out of the average (their F1 would be mostly noise), but
    every prediction still counts: a wrong prediction of a kept class is a false
    positive for it. Returns (macro_f1, kept_classes); macro_f1 is None if no class
    has enough support.
    """
    support = Counter(y_true)
    classes = sorted(c for c, n in support.items() if n >= min_support)
    if not classes:
        return None, []
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s), classes


def growth_transition_metrics(rows: list, min_support: int = GROWTH_TRANSITION_MIN_SUPPORT) -> dict:
    """Exact accuracy and macro-F1 (classes with >= min_support pairs) for Task 5 result
    rows, using only rows that have ground truth and no error."""
    scored = [r for r in rows if not r.get("error") and r.get("gt_transition")]
    y_true = [r["gt_transition"] for r in scored]
    y_pred = [r["pred_transition"] for r in scored]
    macro_f1, classes = macro_f1_min_support(y_true, y_pred, min_support)
    return {
        "n": len(scored),
        "exact_accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / len(scored) if scored else None,
        "macro_f1": macro_f1,
        "macro_f1_classes": len(classes),
    }


def _load_pair_image(row: dict, which: str, data_dir: str) -> bytes:
    """Image bytes for the `which` ("earlier"/"later") image of a pair row. The image
    folder is looked up by time point (combined dataset, e.g. "50 DAP") and, failing
    that, by the session DAP (per-cycle data, e.g. "54 DAP_2 plants")."""
    try:
        return _load_dap_image_bytes(row[f"{which}_time_point"], row[f"{which}_filename"], data_dir)
    except FileNotFoundError:
        return _load_dap_image_bytes(row[f"{which}_source_dap"], row[f"{which}_filename"], data_dir)


def evaluate_growth_transition(
    pairs_csv: str,
    output_csv: str,
    data_dir: str = "./data",
    model: str = MODEL,
    prompt_mode: str = "agp",
    min_support: int = GROWTH_TRANSITION_MIN_SUPPORT,
) -> dict:
    """Task 5: growth stage transition between two images of the same plant.

    Reads the image pairs from `pairs_csv` (ground_truth/growth_transition_pairs_angle_*.csv,
    built by make_transition_pairs.py), sends each pair to the model with the earlier
    image first and the true session DAPs in the prompt, and writes one row per pair to
    `output_csv`.

    The predicted transition (`pred_transition`) is rebuilt from the model's own stage
    lists with growth_transition_name, so it compares exactly with `gt_transition`;
    `transition_consistent` records whether the model's free-text `transition` matched
    its stage lists. Accuracy and macro-F1 use pairs with ground truth and no error;
    macro-F1 averages only classes with at least `min_support` ground-truth pairs.

    Returns {"exact_accuracy", "macro_f1", "macro_f1_classes", "n", "by_cycle",
    "num_trials", "num_errors"}.
    """
    pairs = list(csv.DictReader(open(pairs_csv, newline="", encoding="utf-8")))
    if not pairs:
        raise ValueError(f"No pairs in {pairs_csv}")
    prompt_fn = _prompt_fn(prompt_mode, "growth_transition")

    fieldnames = [
        "cycle", "track_id", "plant_id", "angle",
        "earlier_time_point", "earlier_source_dap", "earlier_filename",
        "later_time_point", "later_source_dap", "later_filename",
        "gt_earlier_stages", "gt_later_stages", "gt_transition",
        "pred_earlier_stages", "pred_later_stages", "pred_transition", "model_transition",
        "transition_consistent", "earlier_correct", "later_correct", "correct",
        "reasoning", "error", "input_tokens", "output_tokens",
    ]
    results = []
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i, pair in enumerate(pairs, start=1):
            print(f"[{i}/{len(pairs)}] {pair['track_id']}: {pair['earlier_filename']} "
                  f"({pair['earlier_source_dap']} DAP) -> {pair['later_filename']} ({pair['later_source_dap']} DAP)")
            out = {f: pair.get(f, "") for f in fieldnames}
            out.update({f: "" for f in fieldnames if f.startswith(("pred_", "model_", "transition_"))
                        or f in ("earlier_correct", "later_correct", "correct", "reasoning", "error",
                                 "input_tokens", "output_tokens")})
            try:
                images = [_load_pair_image(pair, "earlier", data_dir), _load_pair_image(pair, "later", data_dir)]
                response = ollama.chat(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": prompt_fn(pair["earlier_source_dap"], pair["later_source_dap"]),
                        "images": images,
                    }],
                    format=tomato_growth_transition_schema(),
                    options={"num_ctx": NUM_CTX},
                )
                parsed = json.loads(_strip_code_fence(response["message"]["content"]))
                out.update(_token_usage(response))
                earlier = parsed.get("earlier_growth_stages") or []
                later = parsed.get("later_growth_stages") or []
                out["pred_earlier_stages"] = growth_stage_set_name(earlier)
                out["pred_later_stages"] = growth_stage_set_name(later)
                out["pred_transition"] = growth_transition_name(earlier, later)
                out["model_transition"] = parsed.get("transition", "")
                out["transition_consistent"] = (
                    " ".join(out["model_transition"].split()).lower() == out["pred_transition"].lower()
                )
                out["reasoning"] = parsed.get("reasoning", "")
                if pair.get("gt_transition"):
                    out["earlier_correct"] = out["pred_earlier_stages"] == pair["gt_earlier_stages"]
                    out["later_correct"] = out["pred_later_stages"] == pair["gt_later_stages"]
                    out["correct"] = out["pred_transition"] == pair["gt_transition"]
            except Exception as exc:  # noqa: BLE001 - keep going with the remaining pairs
                out["error"] = str(exc)
            results.append(out)
            writer.writerow(out)

    metrics = growth_transition_metrics(results, min_support)
    by_cycle = {c: growth_transition_metrics([r for r in results if r["cycle"] == c], min_support)
                for c in sorted({r["cycle"] for r in results})}
    num_errors = sum(1 for r in results if r["error"])

    fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
    print(f"Saved {len(results)} pair(s) to {output_csv} ({num_errors} error(s))")
    print(f"Exact accuracy {fmt(metrics['exact_accuracy'])}, macro-F1 {fmt(metrics['macro_f1'])} "
          f"over {metrics['macro_f1_classes']} class(es) with >= {min_support} pairs (n={metrics['n']})")
    for cycle, m in by_cycle.items():
        print(f"  {cycle}: exact accuracy {fmt(m['exact_accuracy'])}, macro-F1 {fmt(m['macro_f1'])} "
              f"({m['macro_f1_classes']} classes, n={m['n']})")

    return {**metrics, "by_cycle": by_cycle, "num_trials": len(results), "num_errors": num_errors}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python ollama_vision.py <image_folder> [output_path] [--notes]")
        sys.exit(1)

    args = sys.argv[1:]
    notes_mode = "--notes" in args
    if notes_mode:
        args.remove("--notes")

    folder_path = args[0]
    folder_prefix = Path(folder_path).name.split(" ")[0]
    print("prefix: ", folder_prefix)

    if notes_mode:
        if len(args) > 1:
            output_path = Path(args[1])
        else:
            output_path = f".//output//labels_dap//{folder_prefix}_notes_times.csv"
        try:
            extract_notes_and_times_to_csv(folder_path, str(output_path))
        except Exception as exc:  # noqa: BLE001 - surface any client/model error to the user
            print(f"Error: {exc}")
            sys.exit(1)
        return

    if len(args) > 1:
        output_path_DIR = Path(args[1])
        output_path = Path(output_path_DIR) / f"{folder_prefix}_tomato_growth_stages.json"
    else:
        output_path = Path(folder_path) / f"{folder_prefix}_tomato_growth_stages.json"

    try:
        results = process_folder(folder_path)
    except Exception as exc:  # noqa: BLE001 - surface any client/model error to the user
        print(f"Error: {exc}")
        sys.exit(1)

    output_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results for {len(results)} image(s) to {output_path}")


# Benchmark tasks run by run_experiments: task name -> (evaluate function,
# output subfolder, CSV filename prefix). The subfolders/prefixes match the
# existing output/AGP/<model>/ layout, so compute_benchmark_metrics.py works on
# either prompting condition's folder.
EXPERIMENT_TASKS = {
    "growth_stage": (evaluate_growth_stage, "growthStageClassification", "gs_eval"),
    "ripeness_transition": (evaluate_ripeness_transition, "ripeningTransitionResults", "rt_eval"),
    "reid": (evaluate_cross_dap_reid, "reid", "reid_{angle_mode}_eval"),
    "ripeness_count": (evaluate_ripeness_count, "ripenessCount", "rpc_eval"),
    # Own folder, so its multi-row-per-query CSVs aren't mixed into reid results.
    "reid_shuffled": (evaluate_cross_dap_reid_shuffled, "reidShuffled", "reid_{angle_mode}_shuffled_eval"),
    # Task 5: runs per camera angle from ground_truth/growth_transition_pairs_angle_<x>.csv.
    "growth_transition": (evaluate_growth_transition, "growthTransition", "gtr_eval_angle"),
}

# Tasks that only run when named in --tasks (not part of "all").
OPT_IN_TASKS = {"reid_shuffled", "growth_transition"}

PROMPT_MODE_DIRS = {"agp": "AGP", "zeroshot": "ZeroShot"}

DEFAULT_DATA_DIR = "C:\\Users\\Deeksha Aggarwal\\Documents\\IIRS\\Fruits\\"
# Anchored to this script's folder so results land in TemporalVLM\output\<AGP|ZeroShot>\...
# no matter which directory the command is run from.
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
DEFAULT_DAPS = ["54", "75", "97", "131", "30", "50", "72"]


def run_experiments(argv=None) -> None:
    """Command-line entry point for running benchmark tasks under a prompting condition.

    Examples:
        python Ollama.py --mode zeroshot
        python Ollama.py --mode agp --tasks reid --daps 54 75
        python Ollama.py --mode zeroshot --tasks growth_stage ripeness_count --seed 0
        python Ollama.py --mode agp --tasks reid_shuffled --daps 54 75 97 131 --seed 0
        python Ollama.py --mode zeroshot ^
            --data-dir "C:\\Users\\Deeksha Aggarwal\\Documents\\IIRS\\RRSC-ADVRD_TOMATO" ^
            --labels-dir "C:\\Users\\Deeksha Aggarwal\\Documents\\IIRS\\RRSC-ADVRD_TOMATO\\labels" ^
            --output-root ./output/RRSC-ADVRD_TOMATO

    Results go to <output-root>/<AGP|ZeroShot>/<model>/<task folder>/<prefix>_<dap>.csv.
    Existing CSVs are skipped unless --overwrite is given, so hand-added ground-truth
    columns in earlier runs aren't lost by accident.
    """
    parser = argparse.ArgumentParser(
        description="Run TomatoTemporal benchmark tasks with zero-shot or attribute-guided (AGP) prompts."
    )
    parser.add_argument("--mode", choices=sorted(PROMPTS), required=True,
                        help="prompting condition: 'zeroshot' (basic) or 'agp' (attribute-guided)")
    parser.add_argument("--tasks", nargs="+", choices=[*EXPERIMENT_TASKS, "all"], default=["all"],
                        help="tasks to run (default: all)")
    parser.add_argument("--daps", nargs="+", default=DEFAULT_DAPS,
                        help=f"DAP values to evaluate; DAPs without a labels CSV in --labels-dir "
                             f"are skipped (default: {' '.join(DEFAULT_DAPS)})")
    parser.add_argument("--model", default=MODEL, help=f"Ollama model name (default: {MODEL})")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="folder containing one image subfolder per DAP")
    parser.add_argument("--labels-dir", default="./output/labels_dap",
                        help="folder containing the label files, named <DAP>.csv or "
                             "<DAP>_*.csv (e.g. 54_notes_times.csv)")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT),
                        help=f"root folder for result CSVs (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--angle-mode", choices=["same", "different"], default="same",
                        help="camera-angle pairing for the reid and ripeness_transition tasks")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for candidate/pair sampling; reusing a seed gives "
                             "both prompting conditions the same trials")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite result CSVs that already exist")
    parser.add_argument("--num-shuffles", type=int, default=REID_NUM_SHUFFLES,
                        help=f"candidate orderings per query for the reid_shuffled task "
                             f"(default: {REID_NUM_SHUFFLES})")
    parser.add_argument("--angles", nargs="+", default=["a", "b", "c"],
                        help="camera angles to run for growth_transition (default: a b c)")
    parser.add_argument("--pairs-dir", default=str(DEFAULT_PAIRS_DIR),
                        help="folder with growth_transition_pairs_angle_<x>.csv files "
                             f"(default: {DEFAULT_PAIRS_DIR})")
    args = parser.parse_args(argv)

    # "all" runs the four benchmark tasks; opt-in tasks only run when asked for by name.
    tasks = [t for t in EXPERIMENT_TASKS if t not in OPT_IN_TASKS] if "all" in args.tasks else args.tasks
    model_dir = re.sub(r"[^A-Za-z0-9]", "", args.model)
    run_dir = Path(args.output_root) / PROMPT_MODE_DIRS[args.mode] / model_dir
    summary = []

    if "growth_transition" in tasks:
        evaluate, subfolder, prefix = EXPERIMENT_TASKS["growth_transition"]
        task_dir = run_dir / subfolder
        task_dir.mkdir(parents=True, exist_ok=True)
        for angle in args.angles:
            pairs_csv = Path(args.pairs_dir) / f"growth_transition_pairs_angle_{angle}.csv"
            output_csv = task_dir / f"{prefix}_{angle}.csv"
            if not pairs_csv.exists():
                print(f"Skipping growth_transition angle {angle}: {pairs_csv} not found")
                continue
            if output_csv.exists() and not args.overwrite:
                print(f"Skipping growth_transition angle {angle}: {output_csv} exists (use --overwrite to replace)")
                continue
            print(f"\n=== growth_transition | angle {angle} | mode={args.mode} | model={args.model} ===")
            try:
                result = evaluate(str(pairs_csv), str(output_csv), data_dir=args.data_dir,
                                  model=args.model, prompt_mode=args.mode)
            except Exception as exc:  # noqa: BLE001 - keep going with the remaining angles
                print(f"Error running growth_transition for angle {angle}: {exc}")
                result = {"error": str(exc)}
            summary.append(("growth_transition", f"angle {angle}", result))
        tasks = [t for t in tasks if t != "growth_transition"]

    daps = []
    if tasks:
        # Only DAPs from the list that have a labels CSV are run. They are also the only
        # DAPs reid/ripeness_transition may pair with, so other label files are never read.
        available = {str(d) for d in _discover_daps(args.labels_dir)}
        daps = [d for d in args.daps if d in available]
        missing = [d for d in args.daps if d not in available]
        if missing:
            print(f"Skipping DAP(s) with no labels CSV in {args.labels_dir}: {' '.join(missing)}")
        if not daps:
            parser.error(f"none of the DAPs {' '.join(args.daps)} have a labels CSV in {args.labels_dir}")
        print(f"DAPs: {' '.join(daps)}")

    if args.seed is not None:
        random.seed(args.seed)

    for task in tasks:
        evaluate, subfolder, prefix = EXPERIMENT_TASKS[task]
        task_dir = run_dir / subfolder
        task_dir.mkdir(parents=True, exist_ok=True)
        for dap in daps:
            output_csv = task_dir / f"{prefix.format(angle_mode=args.angle_mode)}_{dap}.csv"
            if output_csv.exists() and not args.overwrite:
                print(f"Skipping {task} DAP {dap}: {output_csv} exists (use --overwrite to replace)")
                continue

            print(f"\n=== {task} | DAP {dap} | mode={args.mode} | model={args.model} ===")
            kwargs = dict(
                data_dir=args.data_dir, labels_dir=args.labels_dir,
                model=args.model, prompt_mode=args.mode,
            )
            if task in ("reid", "ripeness_transition", "reid_shuffled"):
                kwargs["angle_mode"] = args.angle_mode
                kwargs["allowed_daps"] = daps
            if task == "reid_shuffled":
                kwargs["num_shuffles"] = args.num_shuffles
            try:
                result = evaluate(dap, str(output_csv), **kwargs)
            except Exception as exc:  # noqa: BLE001 - keep going with the remaining tasks/DAPs
                print(f"Error running {task} for DAP {dap}: {exc}")
                result = {"error": str(exc)}
            summary.append((task, f"DAP {dap}", result))

    print("\n=== Summary ===")
    for task, dap, result in summary:
        print(f"{task} | {dap}: {result}")

    shuffled = [r for task, _, r in summary if task == "reid_shuffled" and "error" not in r]
    if shuffled:
        mean = lambda key: sum(r[key] for r in shuffled) / len(shuffled)
        print(
            f"reid_shuffled | average over {len(shuffled)} DAP(s): accuracy {mean('accuracy'):.3f}, "
            f"macro-F1 {mean('f1'):.3f}, consistency {mean('consistency'):.3f}"
        )


if __name__ == "__main__":
    run_experiments()

# Replicates the exact image -> bytes -> process_image_bytes path used in inference.py's
# /process_image route, so Ollama's cloud/vision behavior can be tested in isolation.
# image_path = sys.argv[1] if len(sys.argv) > 1 else "./IMG_6650.jpg"
# print(process_image(image_path))
# print(count_tomatoes(image_path, model=MODEL))
# image = Image.open(image_path).convert("RGB")
# image = image.resize((400, 400))

# ollama_byte_array = io.BytesIO()
# image.save(ollama_byte_array, format="JPEG")
# resized_bytes = ollama_byte_array.getvalue()

# ollama_response = process_image_bytes(resized_bytes)
# print("#########", ollama_response)

# if "error" in ollama_response:
#     raise SystemExit(f"Ollama error: {ollama_response['error']}")

# stages = ollama_response.get("growth_stages", [])
# if len(stages) > 1:
#     result = stages[0]["growth_stage_name"] + ", " + stages[1]["growth_stage_name"]
# elif len(stages) == 1:
#     result = stages[0]["growth_stage_name"]
# else:
#     result = "Unknown"

#python Ollama.py --mode zeroshot --tasks growth_stage --data-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\RRSC-ADVRD_TOMATO" --labels-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\RRSC-ADVRD_TOMATO\labels"
r"""
$data   = "C:\Users\Deeksha Aggarwal\Documents\IIRS\RRSC-ADVRD_TOMATO"
$labels = "C:\Users\Deeksha Aggarwal\Documents\IIRS\RRSC-ADVRD_TOMATO\labels"

Zero-shot
# T1 Growth stage -> output\ZeroShot\gemma4cloud\growthStageClassification\gs_eval_30.csv ...
python Ollama.py --mode zeroshot --tasks growth_stage --data-dir $data --labels-dir $labels

# T2 Ripening transition -> ...\ripeningTransitionResults\rt_eval_30.csv ...
python Ollama.py --mode zeroshot --tasks ripeness_transition --data-dir $data --labels-dir $labels --seed 0

python Ollama.py --mode zeroshot --tasks ripeness_transition --data-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\RRSC-ADVRD_TOMATO" --labels-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\RRSC-ADVRD_TOMATO\labels"

# T3 Re-identification -> ...\reid\reid_same_eval_30.csv ...
python Ollama.py --mode zeroshot --tasks reid --data-dir $data --labels-dir $labels --seed 0
python Ollama.py --mode zeroshot --tasks reid --data-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\RRSC-ADVRD_TOMATO" --labels-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\RRSC-ADVRD_TOMATO\labels"

# T4 Ripeness counting -> ...\ripenessCount\rpc_eval_30.csv ...
python Ollama.py --mode zeroshot --tasks ripeness_count --data-dir $data --labels-dir $labels

AGP#

python Ollama.py --mode zeroshot --task growth_stage --data-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO" --labels-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO\labels" --daps 30 50 70 95 130 --output-root ./output/combined --seed 0
python Ollama.py --mode zeroshot --task ripeness_transition --data-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO" --labels-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO\labels" --daps50 70 95 130 --output-root ./output/combined --seed 0

python Ollama.py --mode agp --tasks reid_shuffled --daps 54 75 97 131 --seed 0
python Ollama.py --mode zeroshot --tasks reid_shuffled --daps 54 75 97 131 --seed 0
python Ollama.py --mode zeroshot --tasks reid_shuffled --data-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO_masked" --labels-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO_masked\labels" --daps 50 70 95 130 --output-root ./output/combined_masked --seed 0

python mask_overlay.py                                   # Combined_TOMATO -> Combined_TOMATO_masked
python mask_overlay.py <input_folder> <output_folder>    # any other dataset folder

python Ollama.py --mode agp --tasks growth_transition --model gemma4:cloud --data-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO_masked" --output-root ./output/combined_masked
python Ollama.py --mode zeroshot --tasks growth_transition --model gemma4:cloud --data-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO_masked" --output-root ./output/combined_masked
python Ollama.py --mode agp --tasks growth_transition --angles b --model gemma4:cloud --data-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO_masked" --output-root ./output/combined_masked

"""