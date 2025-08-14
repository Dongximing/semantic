#!/bin/bash

TARGET_PID=774742
CHECK_INTERVAL=10  # 每10秒检查一次

echo "Waiting for process $TARGET_PID to finish..."

# 循环检查进程是否存在
while kill -0 $TARGET_PID 2>/dev/null; do
    sleep $CHECK_INTERVAL
done

echo "Process $TARGET_PID has ended. Running your script..."

python sgl_baseline.py --seed 1234 --start 100 --end 500
python sgl_baseline.py --seed 5678 --start 100 --end 500
python sgl_baseline.py --seed 9101112 --start 100 --end 500
#python sgl_baseline.py --seed 789 --start 156 --end 500 --model Qwen/QwQ-32B

#python baseline.py --seed 2015 --start 0 --end 30 --model Qwen/QwQ-32B
#python baseline.py --seed 2015 --start 0 --end 30 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
#
#python baseline.py --seed 2016 --start 0 --end 30 --model Qwen/QwQ-32B
#python baseline.py --seed 2016 --start 0 --end 30 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B