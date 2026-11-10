# TomatoTemporal: running the code

Code for **TomatoTemporal**, a benchmark that tests vision-language models (VLMs) on
temporal reasoning in greenhouse tomato, and for **CropEval**, which scores both their
accuracy and their reasoning. Models are run locally or in the cloud through
[Ollama](https://ollama.com).

## Contents

1. [Setup](#1-setup)
2. [Data](#2-data)
3. [Preparing the datasets](#3-preparing-the-datasets)
4. [Running the tasks](#4-running-the-tasks-ollamapy)
5. [Ground truth](#5-ground-truth)
6. [Evaluation (CropEval)](#6-evaluation-cropeval)
7. [Tips and troubleshooting](#7-tips-and-troubleshooting)
8. [Script reference](#8-script-reference)

---

## 1. Setup

**Python environment.** The project's virtual environment `tempEnv` has the required
packages (`ollama`, `Pillow`). Activate it before running anything:

```bat
cd /d "C:\Users\Deeksha Aggarwal\Documents\TemporalVLM"
tempEnv\Scripts\activate.bat
```

Other Python installs on this machine do not have `ollama` installed, so always activate
`tempEnv` (or call `tempEnv\Scripts\python.exe` directly).

**Ollama.** Install Ollama and make sure it is running. Then pick a model:

- **Cloud models** (tags ending in `cloud`, e.g. `gemma4:cloud`, `qwen3-vl:235b-cloud`)
  run on Ollama's servers and need you to be signed in to Ollama. No download needed.
- **Local models** must be downloaded first, otherwise Ollama returns
  *404 model not found*:

  ```bat
  ollama pull qwen3-vl:4b
  ollama list
  ```

  This laptop's GPU (RTX 4050, 6 GB) fits models up to about 4–5 GB fully: e.g.
  `qwen3-vl:2b`, `qwen3-vl:4b`, `gemma4:e2b`. Larger models partly run on the CPU and are
  much slower. Check where a running model sits with `ollama ps`.

---

## 2. Data

Images are in `C:\Users\Deeksha Aggarwal\Documents\IIRS\`:

| Folder | Contents |
|---|---|
| `Fruits\` | **Cycle B** (2025): 20 plants, sessions at 54, 75, 97, 131 DAP (plus others not used). Each image has a burned-in "Time / Note" overlay with the plant code and DAP. |
| `RRSC-ADVRD_TOMATO\` | **Cycle A** (2023–24): 60 plants in 20 plots (3 per plot), sessions at 30, 40, 50, 63, 72 DAP. |
| `Combined_TOMATO\` | Both cycles pooled into five time points: `30 DAP`, `50 DAP` (50 + 54), `70 DAP` (72 + 75), `95 DAP`, `130 DAP`, with `labels\<time point>.csv`. |
| `Combined_TOMATO_masked\` | Same as above with the overlay masked (grey boxes). **Use this for experiments.** |

Every image is photographed from three camera angles, `a`, `b` and `c`.

---

## 3. Preparing the datasets

These steps only need to be re-run if the raw data changes.

```bat
:: 1. Labels for the Cycle A folders (filename, DAP, id, angle) -> RRSC-ADVRD_TOMATO\labels\<DAP>.csv
python make_dap_labels.py

:: 2. Pool both cycles into five time points -> IIRS\Combined_TOMATO
python make_combined_dataset.py

:: 3. Mask the burned-in overlay -> IIRS\Combined_TOMATO_masked
python mask_overlay.py
```

`mask_overlay.py` can mask any dataset folder (one subfolder per DAP plus `labels\`).
Put paths in double quotes because they contain a space:

```bat
python mask_overlay.py "C:/Users/Deeksha Aggarwal/Documents/IIRS/Fruits_subset/CycleA" "C:/Users/Deeksha Aggarwal/Documents/IIRS/Fruits_subset_masked/CycleA"
```

---

## 4. Running the tasks (`Ollama.py`)

### Tasks

| Task | `--tasks` name | Input | Output folder |
|---|---|---|---|
| T1 Growth stage classification | `growth_stage` | 1 image | `growthStageClassification` |
| T2 Ripening transition | `ripeness_transition` | 2 images of one plant | `ripeningTransitionResults` |
| T3 Re-identification | `reid` | query + 4 candidates | `reid` |
| T3 Re-identification, 3 candidate orderings | `reid_shuffled` | query + 4 candidates, ×3 | `reidShuffled` |
| T4 Ripeness-stratified counting | `ripeness_count` | 1 image | `ripenessCount` |
| T5 Growth stage transition | `growth_transition` | 2 images of one plant | `growthTransition` |

`--tasks all` runs T1–T4. `reid_shuffled` and `growth_transition` only run when named.

### Prompting conditions

- `--mode zeroshot`: task and answer options only.
- `--mode agp`: attribute-guided prompting (visual cues to look for).

All prompts are in `prompts.py`. Every answer is returned as JSON with a `reasoning`
field, which CropEval scores.

### Commands

The examples use **cmd**. Run each command on one line.

**T1 growth stage, combined masked dataset:**

```bat
python Ollama.py --mode zeroshot --tasks growth_stage --model gemma4:cloud --data-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO_masked" --labels-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO_masked\labels" --daps 30 50 70 95 130 --output-root ./output/combined_masked
```

**T2 ripening transition, T4 counting:** same command with `--tasks ripeness_transition`
or `--tasks ripeness_count`. Add `--seed 0` to T2 so the random pairs can be reproduced.

**T3 re-identification with 3 candidate orderings** (recommended over plain `reid`):

```bat
python Ollama.py --mode agp --tasks reid_shuffled --model gemma4:cloud --data-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO_masked" --labels-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO_masked\labels" --daps 50 70 95 130 --output-root ./output/combined_masked --seed 0
```

**T5 growth stage transition** (reads the pairs from `ground_truth\`; no `--labels-dir`
or `--daps` needed):

```bat
python Ollama.py --mode agp --tasks growth_transition --model gemma4:cloud --data-dir "C:\Users\Deeksha Aggarwal\Documents\IIRS\Combined_TOMATO_masked" --output-root ./output/combined_masked
```

Add `--angles b` to run one angle only (216 pairs).

Replace `--mode agp` with `--mode zeroshot`, and `--model` with any other model, to run
the other conditions.

### Both modes in one go (`run_task.sh`)

From **Git Bash**, `run_task.sh` runs one task for zero-shot and then AGP with one model,
on the masked combined dataset with `--seed 0`:

```bash
./run_task.sh growth_stage gemma4:cloud
./run_task.sh reid_shuffled qwen3-vl:4b
./run_task.sh growth_transition gemma4:cloud --parallel --angles b   # both modes at once
DAPS="50 70" ./run_task.sh ripeness_count minicpm-v:8b --overwrite   # override settings
DRY_RUN=1 ./run_task.sh growth_stage gemma4:cloud                    # show commands only
```

Anything after the model (and `--parallel`) is passed on to `Ollama.py`. The environment
variables `DATA_DIR`, `LABELS_DIR`, `OUTPUT_ROOT`, `DAPS` and `SEED` override the defaults.
The console output of each mode is saved to `logs/<task>_<model>_<mode>.log`.

### Options

| Option | Meaning | Default |
|---|---|---|
| `--mode` | `zeroshot` or `agp` (required) | — |
| `--tasks` | one or more task names, or `all` | `all` |
| `--model` | Ollama model tag | the `MODEL` value at the top of `Ollama.py` (currently `qwen3-vl:4b`) |
| `--data-dir` | dataset folder (one image subfolder per DAP) | `IIRS\Fruits` |
| `--labels-dir` | folder with `<DAP>.csv` / `<DAP>_*.csv` label files | `output\labels_dap` |
| `--daps` | DAPs / time points to run | `54 75 97 131 30 50 72` |
| `--output-root` | where results are written | `output` |
| `--seed` | random seed for pair / candidate sampling; the same seed gives the same trials | none |
| `--angle-mode` | `same` or `different` camera angle for T2/T3 candidates | `same` |
| `--num-shuffles` | candidate orderings for `reid_shuffled` | 3 |
| `--angles` | camera angles for `growth_transition` | `a b c` |
| `--pairs-dir` | folder with `growth_transition_pairs_angle_<x>.csv` | `ground_truth` |
| `--overwrite` | replace existing result files (otherwise they are skipped) | off |

For the combined dataset, always pass `--daps 30 50 70 95 130` (or the subset you want):
the default list does not contain 70, 95 or 130.

### Output

Results are written to

```
<output-root>\<AGP|ZeroShot>\<model>\<task folder>\<file>.csv
```

where `<model>` is the model tag without punctuation (`gemma4:cloud` → `gemma4cloud`).
Each task writes one CSV per DAP (T5: one per angle). Every row includes the model's
answer, its `reasoning`, any `error`, and the token counts `input_tokens` /
`output_tokens`.

---

## 5. Ground truth

All labels are in **`ground_truth\labels.csv`**: one row per image of the combined dataset
(both cycles, 652 images), blank where no label exists.

| Column | Meaning |
|---|---|
| `filename`, `cycle`, `dataset`, `time_point`, `source_dap`, `id`, `angle` | image identity. Cycle A reuses filenames across sessions, so an image is identified by `filename` + `time_point`. |
| `growth_stage` | T1 / T5 label: one or two stage numbers, e.g. `2, 3` (1 side shooting, 2 flowering, 3 fruit development, 4 fruit ripening, 5 senescence) |
| `ripeness_stage` | T2 per-image dominant ripeness: `Raw`, `Semi-ripe`, `Ripe` |
| `raw_count`, `semi_ripe_count`, `ripe_count` | T4 per-class fruit counts |
| `notes` | e.g. conflicting labels to resolve |

T3 needs no labels: the correct candidate comes from the plant IDs.

**Edit `labels.csv` directly** to add or correct labels. `make_ground_truth.py` rebuilds it
from the original label columns and will not overwrite it without `--overwrite` (which
would discard hand edits).

Derived files (re-run these after editing `labels.csv`):

```bat
:: One row per plant and angle, with its filename at every time point -> ground_truth\plant_tracks.csv
python make_plant_tracks.py

:: T5 image pairs, one file per angle -> ground_truth\growth_transition_pairs_angle_<a|b|c>.csv
python make_transition_pairs.py
```

---

## 6. Evaluation (CropEval)

```bat
python cropeval.py
```

CropEval reports for each task and prompting condition:

- **Reasoning validity:** share of answers whose `reasoning` follows the task rubric in
  [`output/reasoning_score_rubric.md`](output/reasoning_score_rubric.md). This needs a
  `score` column (1 or 0) in the result CSVs; scores are added after a run by reviewing
  each reasoning against the rubric.
- **Accuracy** against `ground_truth\labels.csv`.
- **Grounded accuracy:** share of answers that are both correct and validly reasoned.

Metrics are computed per DAP (T5: per angle) and averaged. T5 additionally reports
macro-F1 over transition classes with at least 5 pairs. T1 and T2 are read from
`output\combined`, T3 and T4 from `output\AGP` and `output\ZeroShot`, and T5 from any
`growthTransition` folder under `output\`.

`cropeval.py` can also be imported to score any other results folder, e.g.:

```python
import cropeval as c
r = c.evaluate("gs", c.task_rows(c.OUT / "combined_masked" / "ZeroShot" / "gemma4e4b" / "growthStageClassification"), c.gs_correct)
print(r["reasoning"], r["accuracy"], r["grounded"])
```

---

## 7. Tips and troubleshooting

- **Paths with spaces:** always put them in double quotes. An unquoted
  `C:\Users\Deeksha Aggarwal\...` is split into two arguments.
- **cmd vs PowerShell:** the `$data` / `$labels` shortcuts only work in PowerShell; in cmd,
  write the full paths.
- **404 model not found:** the model is not downloaded. Run `ollama pull <model>`.
- **`ModuleNotFoundError: ollama`:** the wrong Python is active. Activate `tempEnv`.
- **Permission denied when writing a CSV:** the file is open in Excel. Close it.
- **Existing results are skipped:** add `--overwrite` to replace them.
- **Running two jobs at once:** fine in two terminals, as long as they write different
  files (different mode, model, task, DAP or angle). With cloud models this speeds things
  up; with local models on the 6 GB GPU the requests queue, so there is little gain.
- **Context window:** `NUM_CTX` (8192) in `Ollama.py` is enough for all tasks with images
  downscaled to 800 px. If a row's `input_tokens + output_tokens` gets close to it, raise
  `NUM_CTX` (e.g. 16384), and use 32768 for "thinking" models such as `qwen3.8`.
- **Known data gaps:** `Fruits\131 DAP_2 plants\IMG_7277.JPG` is an empty (0-byte) file, so
  trials using it always fail; R4T1 angle c was never photographed at 131 DAP.

---

## 8. Script reference

| Script | Purpose |
|---|---|
| `Ollama.py` | Runs the benchmark tasks with Ollama (see section 4) |
| `prompts.py` | Zero-shot and attribute-guided prompts and JSON schemas for all tasks |
| `cropeval.py` | CropEval: reasoning validity, accuracy, grounded accuracy |
| `make_dap_labels.py` | Label CSVs for the Cycle A image folders |
| `make_combined_dataset.py` | Builds the combined five-time-point dataset |
| `mask_overlay.py` | Masks the burned-in overlay in every image of a dataset |
| `make_ground_truth.py` | Builds `ground_truth\labels.csv` from the original label columns |
| `make_plant_tracks.py` | Builds `ground_truth\plant_tracks.csv` |
| `make_transition_pairs.py` | Builds the T5 pair files |
| `compute_benchmark_metrics.py` | Earlier metrics script for the first AGP run (superseded by `cropeval.py`) |
| `preprocess_images.py` | Early helper: reads the plant code from the image overlay with Tesseract OCR |
| `postProcess.py` | Early helper: converts a growth-stage JSON output to CSV |
"# temporalVLM" 
