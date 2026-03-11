# Beyond Tokens: Semantic-Aware Speculative Decoding for Efficient Inference by Probing Internal States

This repository contains data generation, baseline inference, speculative decoding, and probe-training code for semantic uncertainty experiments.

## Status

- Source comments and documentation are now English-only.
- The repository was scanned for Python syntax errors with `ast.parse`; the current `.py` files passed.
- Several GPQA dataset paths were fixed from `/home/semantic/...` to `/home/semantic/...`.

## Root Files

- `find_stop_words.py`: utility for stop-token discovery and analysis.
- `generate_prefix_data.py`: generates prefix-level data used for downstream processing.
- `generate_samples.py`: Hugging Face generation pipeline that saves hidden states and token-level metadata.
- `inference_hg.py`: standalone Hugging Face inference runner for math and AIME tasks.
- `labeling.py`: semantic clustering and entropy labeling pipeline for generated samples.

## `baseline/`

- `baseline/README.md`: local README for the baseline directory.
- `baseline/baseline.py`: main Hugging Face baseline runner for `math-500`, `aime`, `amc23`, and `gpqa`.
- `baseline/baseline_eval.py`: evaluates saved baseline outputs.
- `baseline/eagle.py`: OpenAI-compatible / vLLM-style baseline runner.
- `baseline/eval.sh`: batch evaluation commands.
- `baseline/gpqa.py`: GPQA-specific speculative-style experiment script kept from the original experiments.
- `baseline/hg_spe.py`: Hugging Face speculative decoding baseline using an assistant model.
- `baseline/offline_sgl_baseline.py`: offline sglang baseline runner for saved datasets.
- `baseline/paser.py`: answer parsing helpers used by baseline evaluation.
- `baseline/run.sh`: baseline launch commands for retained runners.
- `baseline/run_wax.sh`: GPQA baseline launch commands.
- `baseline/sgl_baseline.py`: sglang baseline runner.
- `baseline/traditional_spec.py`: traditional speculative decoding baseline.
- `baseline/utils.py`: multiple-choice prompt and answer helpers.

## `speculative/`

- `speculative/.gitignore`: ignores caches and transient files in the speculative directory.
- `speculative/README.md`: local README for speculative decoding code.
- `speculative/aime.sh`: AIME / AMC-style launch commands for the speculative runner.
- `speculative/analysis.py`: post-run analysis helpers.
- `speculative/eval.py`: evaluation script for speculative outputs.
- `speculative/off.py`: compatibility wrapper that forwards to `semantic_speculative.py`.
- `speculative/paser.py`: answer parsing helpers used by speculative evaluation.
- `speculative/run.sh`: retained speculative launch commands.
- `speculative/run_cyber.sh`: retained speculative launch commands for the cyber setup.
- `speculative/semantic_speculative.py`: main cleaned speculative decoding entry point.
- `speculative/small_model_api.py`: helper code for small-model API or service interaction.
- `speculative/sp_de.py`: hidden-state extraction and sample generation for speculative training data.
- `speculative/speculative_hf_decoding.py`: Hugging Face-based speculative decoding implementation.
- `speculative/speculative_sglang_decoding.py`: sglang-based speculative decoding implementation.
- `speculative/utils.py`: shared speculative helpers.

## `speculative/weight/`

- `speculative/weight/combine_s1_valid_h100_1.51b-100data_math_output_last_hidden_list_best_probe_mse.pt`: combined math probe checkpoint.
- `speculative/weight/combine_s1_valid_h100_1.5r1b-100data_math_output_last_hidden_list_best_probe_mse.pt`: combined math probe checkpoint.
- `speculative/weight/combine_s1_valid_h100_32r1b-100data_math_output_last_hidden_list_best_probe_mse.pt`: combined math probe checkpoint.
- `speculative/weight/s1_valid_h100_1.5b_math_last_hidden_state_best_probe_mse.pt`: 1.5B last-hidden-state probe checkpoint.
- `speculative/weight/s1_valid_h100_32b_gpqa_output_last_hidden_list_best_probe_mse.pt`: 32B GPQA output-hidden probe checkpoint.
- `speculative/weight/s1_valid_h100_32b_math_last_hidden_state_best_probe_mse.pt`: 32B math last-hidden-state probe checkpoint.
- `speculative/weight/s1_valid_h100_32r1b-200data_math_output_last_hidden_list_best_probe_mse.pt`: 32B/R1B combined probe checkpoint.
- `speculative/weight/s1_valid_h100_32r1b_math_output_last_hidden_list_best_probe_mse.pt`: 32B/R1B math probe checkpoint.
- `speculative/weight/s1_valid_h100_r1.5b_gpqa_output_last_hidden_list_best_probe_mse.pt`: R1 1.5B GPQA probe checkpoint.
- `speculative/weight/s1_valid_h100_r1.5b_math_output_last_hidden_list_best_probe_mse.pt`: R1 1.5B math probe checkpoint.

## `training_limo_s1/`

- `training_limo_s1/cob.py`: legacy training script for probe regression over combined datasets.
- `training_limo_s1/filter.py`: filters generated segment folders before training.
- `training_limo_s1/gen.py`: hidden-state extraction runner for saved generation JSON files.
- `training_limo_s1/generate_sample.py`: sample generation pipeline for training data.
- `training_limo_s1/generate_sample_s1_limo.py`: dataset-to-segment conversion for S1/LIMO science data.
- `training_limo_s1/labeling.py`: DeBERTa-based semantic labeling pipeline for large and small model outputs.
- `training_limo_s1/plot.py`: plotting script for accuracy, length, and latency comparisons.
- `training_limo_s1/train_combine.py`: probe training for combined hidden-state datasets.
- `training_limo_s1/train_probe.py`: main probe training script.
- `training_limo_s1/training.py`: another retained probe-training variant.

## `training_limo_s1/` Checkpoints

- `training_limo_s1/combine_s1_valid_h100_1.5deep4096r1b-100data_math_output_last_hidden_list_best_probe_mse.pt`: combined probe checkpoint.
- `training_limo_s1/combine_s1_valid_h100_1.5r1b-100data_math_last_hidden_state_best_probe_mse.pt`: combined probe checkpoint.
- `training_limo_s1/combine_s1_valid_h100_32r1b-200data_math_last_hidden_state_best_probe_mse.pt`: combined probe checkpoint.
- `training_limo_s1/s1_valid_h100_1.5b_math_last_hidden_state_best_probe_mse.pt`: 1.5B math checkpoint.
- `training_limo_s1/s1_valid_h100_32b_gpqa_last_hidden_state_best_probe_mse.pt`: 32B GPQA checkpoint.
- `training_limo_s1/s1_valid_h100_32b_gpqa_output_last_hidden_list_best_probe_mse.pt`: 32B GPQA output-hidden checkpoint.
- `training_limo_s1/s1_valid_h100_32b_math_last_hidden_state_best_probe_mse.pt`: 32B math checkpoint.
- `training_limo_s1/s1_valid_h100_32bqwq_gpqa_last_hidden_state_best_probe_mse.pt`: QwQ GPQA checkpoint.
- `training_limo_s1/s1_valid_h100_32bqwq_gpqa_output_last_hidden_list_best_probe_mse.pt`: QwQ GPQA output-hidden checkpoint.
- `training_limo_s1/s1_valid_h100_32bqwq_math_last_hidden_state_best_probe_mse.pt`: QwQ math checkpoint.
- `training_limo_s1/s1_valid_h100_r1.5b_gpqa_last_hidden_state_best_probe_mse.pt`: R1 1.5B GPQA checkpoint.
- `training_limo_s1/s1_valid_h100_r1.5b_gpqa_output_last_hidden_list_best_probe_mse.pt`: R1 1.5B GPQA output-hidden checkpoint.
- `training_limo_s1/s1_valid_h100_r1.5b_math_last_hidden_state_best_probe_mse.pt`: R1 1.5B math checkpoint.
- `training_limo_s1/s1_valid_h100_r1.5b_math_output_last_hidden_list_best_probe_mse.pt`: R1 1.5B math output-hidden checkpoint.
- `training_limo_s1/s1_valid_h100_r1_mathgpqa_output_last_hidden_list_best_probe_mse.pt`: R1 math/GPQA checkpoint.
- `training_limo_s1/s1_valid_h100_r1_mathqwq_output_last_hidden_list_best_probe_mse.pt`: R1 math/QwQ checkpoint.

## Main Findings

- `labeling.py` imports `semantic_entropy`, but that module is currently absent from the remaining tracked files. This is a functional blocker if `labeling.py` is executed.
- Several scripts still rely on hard-coded absolute paths, CUDA device ids, and local directory layouts. The main affected areas are `baseline/`, `speculative/`, and `training_limo_s1/`.
- A few filenames are still legacy and not self-explanatory: `paser.py`, `gen.py`, `cob.py`, `sp_de.py`, `hg_spe.py`, `off.py`. They should be renamed only with coordinated import and script updates.
- `baseline/gpqa.py` overlaps heavily with `speculative/semantic_speculative.py` and is a good candidate for consolidation.
- `training_limo_s1/` still contains multiple near-duplicate training scripts (`cob.py`, `training.py`, `train_probe.py`, `train_combine.py`) that should eventually be merged behind one configurable entry point.

## Recommended Next Cleanup

1. Replace hard-coded paths and GPU ids with CLI arguments or environment variables.
2. Consolidate duplicate speculative and probe-training runners.
3. Rename legacy filenames after updating all imports and shell scripts.
4. Reintroduce or replace the missing `semantic_entropy` dependency used by `labeling.py`.
