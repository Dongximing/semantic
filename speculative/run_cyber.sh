#!/bin/bash

TARGET_PID=1297161
CHECK_INTERVAL=10  # 每10秒检查一次

echo "Waiting for process $TARGET_PID to finish..."

# 循环检查进程是否存在
while kill -0 $TARGET_PID 2>/dev/null; do
    sleep $CHECK_INTERVAL
done

echo "Process $TARGET_PID has ended. Running your script..."


#python offline_sgl_spec.py  --seed 123   --dataset aime --start_dataset 0 --end_dataset 30 --target_probe /home/ximing/semantic/probe_weight_big/valid_new_full_size_qwq_32b_aime_output_last_hidden_list_best_probe_mse


#python offline_sgl_spec.py  --seed 456   --dataset aime --start_dataset 0 --end_dataset 30 --target_probe /home/ximing/semantic/probe_weight_big/valid_new_full_size_qwq_32b_aime_output_last_hidden_list_best_probe_mse


python offline_sgl_spec.py  --seed 30010   --dataset math-500 --start_dataset 100 --end_dataset 500
python offline_sgl_spec.py  --seed 30020   --dataset math-500 --start_dataset 100 --end_dataset 500
python offline_sgl_spec.py  --seed 9981  --dataset math-500 --start_dataset 100 --end_dataset 500
#python offline_sgl_spec.py  --seed 789   --dataset aime --start_dataset 0 --end_dataset 30 --target_probe /home/ximing/semantic/probe_weight_big/valid_new_full_size_qwq_32b_aime_output_last_hidden_list_best_probe_mse
# python offline_sgl_spec.py  --seed 298   --dataset amc23 --start_dataset 1 --end_dataset 40
# python offline_sgl_spec.py  --seed 398   --dataset amc23 --start_dataset 0 --end_dataset 40
# python offline_sgl_spec.py  --seed 998   --dataset amc23 --start_dataset 0 --end_dataset 40
