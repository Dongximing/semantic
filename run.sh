#!/bin/bash

TARGET_PID=4118532
CHECK_INTERVAL=10  # 每10秒检查一次
echo "Waiting for process $TARGET_PID to finish..."

# 循环检查进程是否存在
while kill -0 $TARGET_PID 2>/dev/null; do
    sleep $CHECK_INTERVAL
done

echo "Process $TARGET_PID has ended. Running your script..."
python generate_prefix_data.py
python slg/generate_samples_sglang.py