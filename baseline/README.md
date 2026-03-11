# Baseline Experiments

This directory contains the cleaned baseline runners used for direct generation and evaluation.

## Main Files

- `baseline.py`: Hugging Face baseline runner for `math-500`, `aime`, `amc23`, and `gpqa`
- `sgl_baseline.py`: sglang baseline runner
- `traditional_spec.py`: traditional speculative decoding baseline
- `eagle.py`: EAGLE-style baseline
- `offline_sgl_baseline.py`: offline sglang baseline variant
- `baseline_eval.py`: evaluate generated outputs
- `eval_length.py`: length statistics
- `utils.py`: shared prompt helpers
- `paser.py`: answer parsing helpers

## Dataset Assets

- `gpqa/`: local GPQA dataset files

## Quick Start

Run the standard Hugging Face baseline:

```bash
python baseline.py \
  --dataset math-500 \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --seed 123 \
  --start 0 \
  --end 50
```

Run GPQA:

```bash
python baseline.py \
  --dataset gpqa \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --seed 9870 \
  --start 0 \
  --end 198
```

## Notes

- Output folders default to `/home/<model>_<dataset>_seed<seed>/`.
- `baseline.py` now covers the old small-model baseline use case through `--model`.
- GPQA loading uses the local `gpqa/` folder when present, and falls back to `Idavidrein/gpqa` otherwise.
- Several duplicate experiment copies were removed to keep this directory maintainable.
