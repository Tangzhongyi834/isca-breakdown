#!/usr/bin/env bash
# Sequential model -> precision -> lengths. The Python runner also locks GPU UUIDs.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
PRECISIONS="${PRECISIONS:-fp16 w8a8 w4a4}"
read -r -a precision_list <<< "$PRECISIONS"
models=("$@")
if [ "${#models[@]}" -eq 0 ]; then
    models=("meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-v0.1")
fi
failed=0
for model in "${models[@]}"; do
    for precision in "${precision_list[@]}"; do
        if "$PYTHON_BIN" profile_llm.py --model "$model" --precision "$precision" \
            --seq-lens 512 1024 2048 4096 --num-samples "${NUM_SAMPLES:-32}" \
            --warmup "${WARMUP:-5}" --repeats "${REPEATS:-3}" --phase prefill \
            --device "$DEVICE" --output-dir "$OUTPUT_DIR"; then
            printf 'Completed: %s %s\n' "$model" "$precision"
        else
            printf 'FAILED (no substituted data): %s %s\n' "$model" "$precision" >&2
            failed=1
        fi
    done
done
exit "$failed"
