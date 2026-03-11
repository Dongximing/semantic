# Semantic Speculative Decoding


## Quick Start

sglang Example:

```bash
python semantic_speculative.py \
  --dataset gpqa \
  --seed 9870 \
  --start_dataset 0 \
  --end_dataset 198
```

HF version:

```bash
python speculative_hf_decoding.py \
  --dataset math-500 \
  --seed 3210 \
  --start_dataset 0 \
  --end_dataset 500
```


```bash
# --dataset: evaluation dataset (e.g., gpqa or math-500)
# --seed: random seed for reproducibility
# --start_dataset: starting index of the dataset to evaluate
# --end_dataset: ending index of the dataset to evaluate
all the probe weights are stored in the speculaitve/weight

you can evalation using speculaitve/eval.py 

python run_eval.py \
  --start 0 \
  --end 30 \
  --dataset aime \
  --eval_path /data/semantic/speculative/spec_result_math-500_seed_456 \
  --seed 123


