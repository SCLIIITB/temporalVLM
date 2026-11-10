#!/usr/bin/env bash
# Run one TomatoTemporal task for both prompting modes (zeroshot, then agp) with one model.
#
# Usage:
#   ./run_task.sh <task> <model> [--parallel] [extra Ollama.py options...]
#
# Examples:
#   ./run_task.sh growth_stage gemma4:cloud
#   ./run_task.sh reid_shuffled qwen3-vl:4b
#   ./run_task.sh growth_transition gemma4:cloud --parallel --angles b
#   DAPS="50 70" ./run_task.sh ripeness_count minicpm-v:8b --overwrite
#   DRY_RUN=1 ./run_task.sh growth_stage gemma4:cloud     # print the commands only
#
# Tasks: growth_stage, ripeness_transition, reid, reid_shuffled, ripeness_count,
#        growth_transition
#
# --parallel runs both modes at the same time (useful for cloud models; with a local
# model on a small GPU the requests just queue). Both modes use the same seed, so they
# see the same trials.
#
# Settings (override by setting the variable before the command):
#   DATA_DIR     dataset folder         (default: IIRS\Combined_TOMATO_masked)
#   LABELS_DIR   label CSV folder       (default: $DATA_DIR/labels)
#   OUTPUT_ROOT  results folder         (default: ./output/combined_masked)
#   DAPS         time points to run     (default: 30 50 70 95 130; reid tasks: 50 70 95 130)
#   SEED         random seed            (default: 0)
#   DRY_RUN=1    print the commands without running them
#
# Output: <OUTPUT_ROOT>/<ZeroShot|AGP>/<model>/<task folder>/...; console output of each
# mode is also saved to logs/<task>_<model>_<mode>.log.

set -uo pipefail
cd "$(dirname "$0")"

TASKS="growth_stage ripeness_transition reid reid_shuffled ripeness_count growth_transition"
if [[ $# -lt 2 ]]; then
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
TASK=$1
MODEL=$2
shift 2
if [[ " $TASKS " != *" $TASK "* ]]; then
    echo "Unknown task '$TASK'. Choose one of: $TASKS" >&2
    exit 1
fi
PARALLEL=0
if [[ ${1:-} == "--parallel" ]]; then
    PARALLEL=1
    shift
fi
EXTRA=("$@")

# Python from the project environment (it has the ollama package).
if [[ -x tempEnv/Scripts/python.exe ]]; then
    PY=tempEnv/Scripts/python.exe
else
    PY=python
fi

DATA_DIR=${DATA_DIR:-"C:/Users/Deeksha Aggarwal/Documents/IIRS/Combined_TOMATO_masked"}
LABELS_DIR=${LABELS_DIR:-"$DATA_DIR/labels"}
OUTPUT_ROOT=${OUTPUT_ROOT:-./output/combined_masked}
SEED=${SEED:-0}
if [[ -z ${DAPS:-} ]]; then
    case $TASK in
        reid|reid_shuffled|ripeness_transition) DAPS="50 70 95 130" ;;
        *) DAPS="30 50 70 95 130" ;;
    esac
fi

ARGS=(--tasks "$TASK" --model "$MODEL" --data-dir "$DATA_DIR" --output-root "$OUTPUT_ROOT" --seed "$SEED")
if [[ $TASK != "growth_transition" ]]; then
    # growth_transition reads its image pairs from ground_truth/ instead of label files.
    read -r -a DAP_LIST <<< "$DAPS"
    ARGS+=(--labels-dir "$LABELS_DIR" --daps "${DAP_LIST[@]}")
fi

MODEL_SLUG=${MODEL//[^A-Za-z0-9]/}
mkdir -p logs

run_mode() {
    local mode=$1
    local log="logs/${TASK}_${MODEL_SLUG}_${mode}.log"
    local cmd=("$PY" -u Ollama.py --mode "$mode" "${ARGS[@]}" "${EXTRA[@]}")
    echo ">>> [$mode] ${cmd[*]}"
    if [[ ${DRY_RUN:-0} == 1 ]]; then
        return 0
    fi
    if [[ $PARALLEL == 1 ]]; then
        "${cmd[@]}" > "$log" 2>&1
    else
        "${cmd[@]}" 2>&1 | tee "$log"
        return "${PIPESTATUS[0]}"
    fi
}

echo "Task: $TASK | model: $MODEL | data: $DATA_DIR | output: $OUTPUT_ROOT | seed: $SEED"
start=$SECONDS
status=0

if [[ $PARALLEL == 1 ]]; then
    run_mode zeroshot & pid_zs=$!
    run_mode agp & pid_agp=$!
    if [[ ${DRY_RUN:-0} != 1 ]]; then
        echo "Running both modes in parallel; follow progress with:"
        echo "  tail -f logs/${TASK}_${MODEL_SLUG}_zeroshot.log logs/${TASK}_${MODEL_SLUG}_agp.log"
    fi
    wait "$pid_zs" || { echo "zeroshot run failed (see log)" >&2; status=1; }
    wait "$pid_agp" || { echo "agp run failed (see log)" >&2; status=1; }
else
    run_mode zeroshot || { echo "zeroshot run failed" >&2; status=1; }
    run_mode agp || { echo "agp run failed" >&2; status=1; }
fi

echo "Done in $(( (SECONDS - start) / 60 )) min $(( (SECONDS - start) % 60 )) s (exit status $status)."
[[ ${DRY_RUN:-0} == 1 ]] || echo "Logs: logs/${TASK}_${MODEL_SLUG}_zeroshot.log, logs/${TASK}_${MODEL_SLUG}_agp.log"
exit $status
