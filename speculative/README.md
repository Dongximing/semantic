# Semantic Speculative Decoding

This branch is a cleaned-up version of the original `cyber11` experiments.

## Main Entry

Use [`semantic_speculative.py`](/home/semantic/speculative/semantic_speculative.py) as the primary script.

`off.py` is kept only as a compatibility wrapper.

## What This Repo Contains

- `semantic_speculative.py`: main offline speculative decoding runner based on `sglang`
- `eval.py`: evaluate generated results
- `utils.py`: seeding and GPQA prompt helpers
- `paser.py`: answer parsing helpers used by evaluation
- `speculative_hf_decoding.py`: Hugging Face based variant
- `speculative_sglang_decoding.py`: older sglang variant
- `weight/`: probe checkpoints

## Quick Start

Example:

```bash
python semantic_speculative.py \
  --dataset gpqa \
  --seed 9870 \
  --start_dataset 0 \
  --end_dataset 198
```

Math-500 example:

```bash
python semantic_speculative.py \
  --dataset math-500 \
  --seed 3210 \
  --start_dataset 0 \
  --end_dataset 500
```

## Important Args

- `--dataset`: `math-500`, `aime`, `amc23`, or `gpqa`
- `--probe_device`: device for the probe models, default `cuda:4`
- `--small_device`: visible CUDA device for the speculative model, default `4`
- `--big_device`: visible CUDA device for the target model, default `5`

## Notes

- The code expects local model paths under `/home/original_models/`.
- GPQA loading currently uses the local dataset path `/home/semantic/baseline/gpqa`.
- Output is written under `{dataset}{data_dir}{seed}/spec_<dataset>_<index>/spec_generation.json`.
