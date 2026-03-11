# Beyond Tokens: Semantic-Aware Speculative Decoding for Efficient Inference by Probing Internal States

This repository contains code for the paper, including data generation, speculative decoding, and probe training.

## Overview

The codebase is organized around two main workflows:

1. `Speculative decoding`
   This part implements semantic-aware speculative decoding for faster inference.

2. `Probe training`
   This part contains the training pipeline for the probe models used during decoding.

## Main Directories

### `speculative/`

This directory contains the main semantic speculative decoding code.

files:


- `semantic_speculative.py`: offline sglang implementation for semantic speculative decoding
- `speculative_hf_decoding.py`: Hugging Face implementation for semantic speculative decoding
- `speculative_sglang_decoding.py`: online sglang implementation for semantic speculative decoding
- `eval.py`: evaluation script for generated results

### `training_limo_s1/`

This directory contains probe training and data preparation code.

files:

- `training_limo_s1/generate_data.py`: generating prefixes
- `training_limo_s1/generate_sample_s1_limo.py`: generating data for clustering based on prefixes
- `training_limo_s1/train_probe.py`: probe training pipeline
- `training_limo_s1/labeling.py`: semantic labeling pipeline


## Semantic Speculative Decoding

All probe weights are stored in the `speculative/weight` directory.
### Run the sglang decoding script


```bash
python speculative/semantic_speculative.py \
  --dataset gpqa \
  --seed 9870 \
  --start_dataset 0 \
  --end_dataset 198
```

### Run the Hugging Face decoding variant

```bash
python speculative/speculative_hf_decoding.py \
  --dataset math-500 \
  --seed 3210 \
  --start_dataset 0 \
  --end_dataset 500
```

## Evaluation

Use the evaluation script after generation is finished:

```bash
python speculative/eval.py \
  --start 0 \
  --end 30 \
  --dataset aime \
  --eval_path /path/to/speculative/results \
  --seed 123
```

## Notes

- `--dataset` selects the evaluation dataset, such as `gpqa`, `math-500`, or `aime`
- `--seed` controls reproducibility
- `--start_dataset` and `--end_dataset` define the evaluation range

## Installation

Create environment and install dependencies:

```bash
pip install -r requirements.txt