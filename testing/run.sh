#!/bin/bash




python test.py --seed 123 --model unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit
python test.py --seed 456 --model unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit
python test.py --seed 789 --model unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit
python test.py --seed 43 --model unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit
python test.py --seed 42 --model unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit


python test.py --seed 123 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
python test.py --seed 456 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
python test.py --seed 789 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
python test.py --seed 43 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
python test.py --seed 42 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B


python test.py --seed 123 --model unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit
python test.py --seed 456 --model unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit
python test.py --seed 789 --model unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit
python test.py --seed 43 --model unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit
python test.py --seed 42 --model unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit