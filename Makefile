# Local WikiText-2 profiling. Override settings with: make run GPU=2
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
# Keep experiments sequential even when invoked with make -j.
.NOTPARALLEL:
.DEFAULT_GOAL := help

PYTHON ?= /home/tangzhongyi/miniconda3/envs/million/bin/python
GPU ?= 0

MISTRAL_MODEL ?= /home/shared/models/mistralai/Mistral-7B-v0.1
LLAMA2_MODEL ?= /home/shared/models/meta-llama/Llama-2-7b-hf
LLAMA3_MODEL ?= /home/shared/models/meta-llama/Meta-Llama-3-8B-Instruct
MODELS ?= $(MISTRAL_MODEL) $(LLAMA2_MODEL) $(LLAMA3_MODEL)
DATASET_PATH ?= /home/shared/datasets/wikitext-2-raw-v1

# True W4A4 is unavailable; never substitute another precision.
PRECISIONS ?= fp16 w8a8
SEQ_LENS ?= 512 1024 2048 4096
NUM_SAMPLES ?= 32
WARMUP ?= 5
REPEATS ?= 3
CROSS_SEQ_LEN ?= 4096

CACHE_DIR ?= data/wikitext2_local
OUTPUT_DIR ?= results/local_wikitext
FIGURE_DIR ?= figures

export CUDA_VISIBLE_DEVICES := $(GPU)
export HF_HUB_OFFLINE := 1
export HF_DATASETS_OFFLINE := 1
export PYTHONUNBUFFERED := 1
export MODELS DATASET_PATH PRECISIONS SEQ_LENS NUM_SAMPLES WARMUP REPEATS CACHE_DIR OUTPUT_DIR

# Export the Python body as a variable: GNU Make can strip indentation from
# recipe heredocs, which would break nested Python blocks.
define PROFILE_LOCAL_PYTHON
import os
import shlex
from unittest.mock import patch
from datasets import load_from_disk
from profile_llm import main

dataset_path = os.environ["DATASET_PATH"]
test_dataset = load_from_disk(dataset_path)["test"]
if "text" not in test_dataset.column_names:
    raise SystemExit("Local WikiText-2 test split must contain a text column")
models = shlex.split(os.environ["MODELS"])
precisions = shlex.split(os.environ["PRECISIONS"])
seq_lens = shlex.split(os.environ["SEQ_LENS"])
if not models or not precisions or not seq_lens:
    raise SystemExit("MODELS, PRECISIONS and SEQ_LENS must not be empty")
print(f"Local dataset: {dataset_path}, test rows: {len(test_dataset)}", flush=True)

# Redirect the current CLI's Hub loader only within this process.
with patch("datasets.load_dataset", return_value=test_dataset):
    for model in models:
        for precision in precisions:
            print(f"Running {model} / {precision}", flush=True)
            status = main([
                "--model", model,
                "--precision", precision,
                "--seq-lens", *seq_lens,
                "--num-samples", os.environ["NUM_SAMPLES"],
                "--batch-size", "1",
                "--warmup", os.environ["WARMUP"],
                "--repeats", os.environ["REPEATS"],
                "--phase", "prefill",
                "--device", "cuda:0",
                "--local-files-only",
                "--cache-dir", os.environ["CACHE_DIR"],
                "--output-dir", os.environ["OUTPUT_DIR"],
            ])
            if status != 0:
                raise SystemExit(status)
endef
export PROFILE_LOCAL_PYTHON

.PHONY: help check run plots plot-breakdown plot-nonlinear plot-cross-model all

help:
	@printf '%s\n' \
	  'make run GPU=0          Run all three local models, FP16 then W8A8.' \
	  'make plots              Plot existing measured results (PDF + PNG).' \
	  'make all GPU=0          Run experiments, then generate all figures.' \
	  'make check              Check embedded Python syntax without running experiments.' \
	  'make plot-breakdown     Generate Figure 1(a) only.' \
	  'make plot-nonlinear     Generate Figure 1(b) only.' \
	  'make plot-cross-model   Generate Figure 1(c) only; needs >= 2 models.' \
	  '' \
	  'Overrides: PYTHON, MODELS, DATASET_PATH, PRECISIONS, SEQ_LENS,' \
	  'NUM_SAMPLES, WARMUP, REPEATS, CACHE_DIR, OUTPUT_DIR, FIGURE_DIR,' \
	  'CROSS_SEQ_LEN. Use a separate OUTPUT_DIR for different settings.' \
	  '' \
	  'Example: make run GPU=2 PRECISIONS=fp16' \
	  'Default Python: million environment. No conda activation required.' \
	  'W4A4 is unavailable; missing measurements stay explicitly missing.'

check:
	@"$(PYTHON)" -c 'import os; compile(os.environ["PROFILE_LOCAL_PYTHON"], "<Makefile:run>", "exec"); print("Makefile Python syntax: OK (no experiments started)")'

run:
	@"$(PYTHON)" -c "$$PROFILE_LOCAL_PYTHON"

plot-breakdown:
	@"$(PYTHON)" plotting/plot_runtime_breakdown.py \
	  --input "$(OUTPUT_DIR)/summary/figure1_data.csv" \
	  --seq-lens $(SEQ_LENS) --output-dir "$(FIGURE_DIR)" --allow-incomplete

plot-nonlinear:
	@"$(PYTHON)" plotting/plot_nonlinear_fraction.py \
	  --input "$(OUTPUT_DIR)/summary/figure1_data.csv" \
	  --seq-lens $(SEQ_LENS) --output-dir "$(FIGURE_DIR)" --allow-incomplete

plot-cross-model:
	@"$(PYTHON)" plotting/plot_cross_model.py \
	  --input "$(OUTPUT_DIR)/summary/figure1_data.csv" \
	  --seq-len $(CROSS_SEQ_LEN) --output-dir "$(FIGURE_DIR)" --allow-incomplete

plots: plot-breakdown plot-nonlinear plot-cross-model

# .NOTPARALLEL guarantees this prerequisite order, including make -j all.
all: run plots
