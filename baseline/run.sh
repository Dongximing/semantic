#!/bin/bash
python sgl_baseline.py --seed 123 --start 100 --end 500 --model Qwen/QwQ-32B
python sgl_baseline.py --seed 456 --start 100 --end 500 --model Qwen/QwQ-32B
python sgl_baseline.py --seed 789 --start 100 --end 500 --model Qwen/QwQ-32B

#python baseline.py --seed 2015 --start 0 --end 30 --model Qwen/QwQ-32B
#python baseline.py --seed 2015 --start 0 --end 30 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
#
#python baseline.py --seed 2016 --start 0 --end 30 --model Qwen/QwQ-32B
#python baseline.py --seed 2016 --start 0 --end 30 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B