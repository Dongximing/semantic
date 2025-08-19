import datetime
import torch
import json
import os
import pickle
import numpy as np
import traceback
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import argparse
import requests
STOP_TOKENS = [
    ' \n\n', '.\n\n', ':\n\n', '\n\n',
    ')\n\n', '?\n\n', ']\n\n', ').\n\n'
]
AIME_STOP_TOKENS = [
    ' \n\n', '.\n\n', ':\n\n', '\n\n',
    ')\n\n', '?\n\n', ']\n\n', ').\n\n',
]



def predict(tokenizer, model, input_data, temperature, return_full=False, return_latent=False,gpu=0):


    sampling_params = {
        "temperature": temperature,
        "top_p": 0.95,
        "max_new_tokens": 500,
        "stop_token_ids":[4710,382,1447,271,692,1939,2533,3593,13824,14190],
        "no_stop_trim": True
    }
    json_data_check = {
        "text": [input_data],
        "sampling_params": sampling_params,
        "return_hidden_states": True,
    }
    checking_outputs = requests.post(
        f"http://0.0.0.0:{8080}/generate",
        json=json_data_check,
    )
    checking_outputs = checking_outputs.json()
    checking_output = checking_outputs[0]
    for i in range(len(checking_output["meta_info"]["hidden_states"])):
        checking_output["meta_info"]["hidden_states"][i] = torch.tensor(
            checking_output["meta_info"]["hidden_states"][i], dtype=torch.bfloat16
        )
    hidden_states = torch.cat(
        [
            i.unsqueeze(0) if len(i.shape) == 1 else i
            for i in checking_output["meta_info"]["hidden_states"]
        ]
    )
    Completion_tokens = checking_output['meta_info']['completion_tokens']


    real_answer = checking_output['text']
    # print('checking_output',real_answer)
    hidden_states = hidden_states[-Completion_tokens:,:] #len *hidden


    last_token_hidden = hidden_states[-1]
    sec_last_input = hidden_states[-2]
    last_tok_bef_gen_input = hidden_states[0]
    output_hidden_states = hidden_states


    hidden_states = (
        last_token_hidden,sec_last_input,last_tok_bef_gen_input,output_hidden_states
    )
    return (real_answer, hidden_states)




def process_file_to_pickle(json_path, out_pkl_path,  num_generations):
    with open(json_path, 'r', encoding='utf-8') as f:
        alldata = json.load(f)
    all_generations = []

    log_file = out_pkl_path.replace('.pkl', '.log')
    n = int(len(alldata))
    data = alldata[:n - 1]
    print(len(data))

    for index, element in enumerate(data):
        input_text = element['text']
        for i in range(num_generations + 1):
            log_entry = {
                "time": datetime.datetime.now().isoformat(),
                "sample_index": index,
                "generation_index": i,
                "input_text_preview": input_text[:80] + ("..." if len(input_text) > 80 else ""),
            }
            # try:
            if i == 0:
                (
                    most_real_answer,
                    ( most_last_token_hidden,most_sec_last_input,most_last_tok_bef_gen_input,most_output_hidden_states)
                ) = predict( input_text,temperature=0.1)
            else:
                (
                    real_answer,
                    ( last_token_hidden,sec_last_input,last_tok_bef_gen_input,output_hidden_states)
                ) = predict(input_text,temperature=0.6)


            if i == 0:
                all_generations.append({
                    "most_input_text": input_text,
                    'most_real_answer': most_real_answer,
                    "most_last_hidden_state": most_last_token_hidden.cpu(),
                    "most_sec_last_hidden_state": most_sec_last_input.cpu(),
                    "most_last_input_token_state": most_last_tok_bef_gen_input.cpu(),
                    "most_output_last_hidden_list": most_output_hidden_states.cpu(),

                    "most_generation_index": -1,
                    "most_sample_index": index,
                })
            else:

                all_generations.append({
                    "input_text": input_text,
                    'real_answer': real_answer,
                    "last_hidden_state": last_token_hidden.cpu(),
                    "sec_last_hidden_state": sec_last_input.cpu(),
                    "last_input_token_state": last_tok_bef_gen_input.cpu(),
                    "output_last_hidden_list": output_hidden_states.cpu(),
                    "generation_index": i - 1,
                    "sample_index": index,
                })
            # except Exception as e:
            #     log_entry.update({
            #         "status": "error",
            #         "error_msg": str(e),
            #         "traceback": traceback.format_exc()
            #     })
            #     if i == 0:
            #         all_generations.append({
            #             "most_input_text": input_text,
            #             "most_real_answer": None,
            #             "most_predicted_answer": None,
            #             "most_output_last_hidden_list": None,
            #             "most_ppl": None,
            #             "most_log_likelihoods": None,
            #             "most_embedding": None,
            #             "most_last_hidden_state": None,
            #             "most_sec_last_hidden_state": None,
            #             "most_last_input_token_state": None,
            #             "most_generation_index": -1,
            #             "most_sample_index": index,
            #             'triggered_stop': None
            #         })
            #     else:
            #         all_generations.append({
            #             "input_text": input_text,
            #             "real_answer": None,
            #             "predicted_answer": None,
            #             "output_last_hidden_list": None,
            #             "ppl": None,
            #             "log_likelihoods": None,
            #             "embedding": None,
            #             "last_hidden_state": None,
            #             "sec_last_hidden_state": None,
            #             "last_input_token_state": None,
            #             "generation_index": i - 1,
            #             "sample_index": index,
            #             'triggered_stop': None
            #         })

            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    with open(out_pkl_path, "wb") as f:
        pickle.dump(all_generations, f)
    print(f"Saved {out_pkl_path} with {len(all_generations)} generations.")



# wail /home/cs/staff/shaowei/hf/math-result_left
# quail /data/ximing/math-result_left
def inference_model_pickle(task_name: str, base_dir,
                           start=9, end=50, num_generations=20):
    for number in tqdm(range(start, end)):

        dirname = f'data-877_{number}'
        dir_path = os.path.join(base_dir, dirname)
        json_path = os.path.join(dir_path, f'generation.json')
        out_pkl_path = os.path.join(dir_path, f'new_generations_{number}.pkl')

        if not os.path.isfile(json_path):
            print(f"[Warning] {json_path} does not exist! Skipping...")
            continue
        if os.path.isfile(out_pkl_path):
            print(f"[Warning] {out_pkl_path} exist! Skipping...")
            continue

        print(f"[Info] Processing file: {json_path}")
        process_file_to_pickle(json_path, out_pkl_path, num_generations)

    print("[Info] Processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--start", type=int, help="dataset", default=0)
    parser.add_argument("--end", type=int, help="dataset", default=105)  #
    parser.add_argument("--base_dir", type=str, help="dataset",
                        default='/data/semantic/training_limo_s1/data_s1_200_science_big')
    args = parser.parse_args()
    # /home/cs/staff/shaowei/semantic/aime
    # /data/ximing/aime
    # /home/cs/staff/shaowei/semantic/deepseek-32b_r1_awq_math
    inference_model_pickle(task_name='science', base_dir=args.base_dir,
                           start=args.start, end=args.end)
    print("done")
