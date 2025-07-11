#!/bin/bash


#python speculative_hf_decoding.py --seed 98  --dataset aime --start_dataset 0 --end_dataset 30
#python speculative_hf_decoding.py --seed 498  --dataset aime --start_dataset 0 --end_dataset 30
#python speculative_hf_decoding.py --seed 398  --dataset aime --start_dataset 0 --end_dataset 30
#python speculative_hf_decoding.py --seed 98  --dataset math-500 --start_dataset 353 --end_dataset 500


python speculative_hf_decoding.py --seed 298  --dataset math-500 --start_dataset 306 --end_dataset 500
python speculative_hf_decoding.py --seed 198  --dataset math-500 --start_dataset 100 --end_dataset 500