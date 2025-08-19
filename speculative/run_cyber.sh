python speculative_sglang_decoding.py --seed 123   --dataset math-500 --start_dataset 100 --end_dataset 500
python speculative_sglang_decoding.py --seed 123   --dataset aime --start_dataset 0 --end_dataset 30 --target_probe /home/ximing/semantic-1/probe_weight_big/valid_new_full_size_qwq_32b_aime_output_last_hidden_list_best_probe_mse
python speculative_sglang_decoding.py --seed 123   --dataset amc23 --start_dataset 0 --end_dataset 40

python speculative_sglang_decoding.py --seed 456   --dataset math-500 --start_dataset 0 --end_dataset 500
python speculative_sglang_decoding.py --seed 456   --dataset aime --start_dataset 0 --end_dataset 30 --target_probe /home/ximing/semantic-1/probe_weight_big/valid_new_full_size_qwq_32b_aime_output_last_hidden_list_best_probe_mse
python speculative_sglang_decoding.py --seed 456   --dataset amc23 --start_dataset 0 --end_dataset 40


python speculative_sglang_decoding.py --seed 789   --dataset math-500 --start_dataset 0 --end_dataset 500
python speculative_sglang_decoding.py --seed 789   --dataset aime --start_dataset 0 --end_dataset 30 --target_probe /home/ximing/semantic-1/probe_weight_big/valid_new_full_size_qwq_32b_aime_output_last_hidden_list_best_probe_mse
python speculative_sglang_decoding.py --seed 789   --dataset amc23 --start_dataset 0 --end_dataset 40