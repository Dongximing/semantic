#!/bin/bash
python sgl_baseline.py --seed 123 --start 100 --end 500 --model DeepSeek-R1-Distill-1.5b --dataset math-500
python sgl_baseline.py --seed 456 --start 100 --end 500 --model DeepSeek-R1-Distill-1.5b --dataset math-500
python sgl_baseline.py --seed 789 --start 100 --end 500 --model DeepSeek-R1-Distill-1.5b --dataset math-500