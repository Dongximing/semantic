#!/bin/bash

TARGET_PID=1524468
CHECK_INTERVAL=10  # 每10秒检查一次

echo "Waiting for process $TARGET_PID to finish..."

# 循环检查进程是否存在
while kill -0 $TARGET_PID 2>/dev/null; do
    sleep $CHECK_INTERVAL
done

echo "Process $TARGET_PID has ended. Running your script..."
python speculative_hf_decoding.py --seed 298  --dataset aime --start_dataset 0 --end_dataset 30
#python speculative_hf_decoding.py --seed 498  --dataset aime --start_dataset 0 --end_dataset 30
#python speculative_hf_decoding.py --seed 398  --dataset aime --start_dataset 0 --end_dataset 30
