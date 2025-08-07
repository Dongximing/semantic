#!/bin/bash

TARGET_PID=234912
CHECK_INTERVAL=10  # 每10秒检查一次

echo "Waiting for process $TARGET_PID to finish..."

# 循环检查进程是否存在
while kill -0 $TARGET_PID 2>/dev/null; do
    sleep $CHECK_INTERVAL
done

echo "Process $TARGET_PID has ended. Running your script..."
#python speculative_hf_decoding.py --seed 981   --dataset aime --start_dataset 0 --end_dataset 29 --data_dir /data/semantic/speculative/speed_up_spec_result_aime-30_deepseek_r132_deepseek1.5seed_
#python speculative_hf_decoding.py --seed 20981  --dataset aime --start_dataset 0 --end_dataset 29 --data_dir /data/semantic/speculative/speed_up_spec_result_aime-30_deepseek_r132_deepseek1.5seed_
#python speculative_hf_decoding.py --seed 30981   --dataset aime --start_dataset 0 --end_dataset 29 --data_dir /data/semantic/speculative/speed_up_spec_result_aime-30_deepseek_r132_deepseek1.5seed_
#python speculative_hf_decoding.py --seed 2022   --dataset math-500 --start_dataset 100 --end_dataset 500 --data_dir /data/semantic/speculative/speed_up_spec_result_math-500_deepseek_r132_deepseek1.5seed_
#python speculative_hf_decoding.py --seed 2023   --dataset math-500 --start_dataset 100 --end_dataset 500 --data_dir /data/semantic/speculative/speed_up_spec_result_math-500_deepseek_r132_deepseek1.5seed_
#python speculative_sglang_decoding.py --seed 2013   --dataset math-500 --start_dataset 100 --end_dataset 500
#python speculative_sglang_decoding.py --seed 2025   --dataset math-500 --start_dataset 101 --end_dataset 500
#python speculative_sglang_decoding.py --seed 30   --dataset math-500 --start_dataset 169 --end_dataset 500
python speculative_sglang_decoding.py --seed 79   --dataset math-500 --start_dataset 100 --end_dataset 500
#python slg/generate_samples_sglang.py

#python speculative_hf_decoding.py --seed 2004   --dataset aime --start_dataset 0 --end_dataset 30 --data_dir /data/semantic/speculative/speed_up_spec_result_aime-30_deepseek_r132_deepseek1.5seed_
#python speculative_hf_decoding.py --seed 2004   --dataset math-500 --start_dataset 100 --end_dataset 500 --data_dir /data/semantic/speculative/speed_up_spec_result_math-500_deepseek_r132_deepseek1.5seed_  --target_probe /data/semantic/training/new_deepseekr132b_math-500_output_last_hidden_list_best_probe_mse
#


#python speculative_hf_decoding.py --seed 2981  --dataset aime --start_dataset 0 --end_dataset 30
#python speculative_hf_decoding.py --seed 3981  --dataset aime --start_dataset 0 --end_dataset 30
#python speculative_hf_decoding.py --seed 1981  --dataset aime --start_dataset 0 --end_dataset 30



#python speculative_hf_decoding.py --seed 75  --dataset aime --start_dataset 0 --end_dataset 30
#python speculative_hf_decoding.py --seed 30  --dataset aime --start_dataset 0 --end_dataset 30