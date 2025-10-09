# python traditional_spec1.py --seed 3210 --dataset amc23 --start 0 --end 40
# python traditional_spec1.py --seed 6540 --dataset amc23 --start 0 --end 40
# python traditional_spec1.py --seed 9870 --dataset amc23 --start 0 --end 40
# python traditional_spec1.py --seed 3210 --dataset aime --start 0 --end 30
# python traditional_spec1.py --seed 6540 --dataset aime --start 0 --end 30
# python traditional_spec1.py --seed 9870 --dataset aime --start 0 --end 30

python sgl_b.py --seed 3210 --start 0 --end 198 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
python sgl_b.py --seed 9870 --start 0 --end 198 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
