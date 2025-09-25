import sglang as sgl
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import numpy as np
import random
import torch
import time
import requests
import json
import math
MATH_PROMPT = "\nPlease reason step by step, and put your final answer within \\boxed{}."
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


NUMBER = 0
def extract_potential_ids(input_top_logprobs, draft_len_output):
    # 直接取最后 draft_len_output 个元素
    last_items = input_top_logprobs[-draft_len_output:]

    potential_ids = []
    for i, item in enumerate(last_items):
        if item is None or len(item) == 0:
            raise ValueError(f"第 {i} 个元素为空或无候选项: {item}")
        top1_id = item[0][1]  # 取第一个 tuple 的第二个元素（token_id）
        potential_ids.append(top1_id)

    return potential_ids



def speculative_decoding(llm_big,llm_small,target_tokenizer,speculative_tokenizer,problem,answer,max_new_tokens):
    start_time = time.time()
    messages = [
            {"role": "user", "content": problem + MATH_PROMPT}
        ]
    # apply the pattern for speculative model and target model
    target_text = target_tokenizer.apply_chat_template(  # big
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    valid_checking_target_text = target_text 
    speculative_text = speculative_tokenizer.apply_chat_template( #small
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    valid_draft_text = speculative_text
    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 3,

    }
    def checking_is_finish(generated_ids, max_new_tokens):
        return len(speculative_tokenizer.encode(generated_ids))  < max_new_tokens
    checking_sampling_params = {"temperature": 0.1,"max_new_tokens": 1}
    print('\n')
    while checking_is_finish(max_new_tokens=14000, generated_ids=valid_draft_text):
        print('valid_draft_text', valid_draft_text)
        speculative_output = llm_small.generate(
            [valid_draft_text], sampling_params=sampling_params, return_logprob=True
        )
        unvalid_speculative_real_output_text = speculative_output[0]['text']
        
        
        draft_len_output = speculative_output[0]['meta_info']['completion_tokens']
        # print(speculative_output[0]['meta_info']['output_token_logprobs'])
        print('unvalid_speculative_real_output_text',unvalid_speculative_real_output_text)
        unvalid_id = speculative_tokenizer(valid_draft_text+unvalid_speculative_real_output_text)["input_ids"][-draft_len_output:]
        print('unvalid_id', unvalid_id)
        # check whether to accept the speculative output
        prob_small_result = [{"id": tid, "prob": math.exp(lp)} for lp, tid, _ in speculative_output[0]['meta_info']['output_token_logprobs']]
        print('prob_small_result:',prob_small_result)
        for item in prob_small_result:
            if item["id"] == 151643:
                valid_draft_text = valid_draft_text + unvalid_speculative_real_output_text
                break

        unvalid_checking_target_text = valid_checking_target_text+unvalid_speculative_real_output_text
        print('\n')
        print('unvalid_checking_target_text:', unvalid_checking_target_text)
        checking_outputs = llm_big.generate([unvalid_checking_target_text],
                        sampling_params = {"temperature": 0.1,"max_new_tokens": 1},
                        return_logprob=True,
                        logprob_start_len=0,top_logprobs_num=2
            
                    )
        print('==================big model ===============')
        # print(checking_outputs[0]['meta_info'])

        potential_ids = extract_potential_ids(checking_outputs[0]['meta_info']['input_top_logprobs'], draft_len_output)
        prob_big_result = [
                    {"id": tid, "prob": math.exp(lp)}
                    for lp, tid, _ in checking_outputs[0]['meta_info']['input_token_logprobs'][-draft_len_output:]
                ]
        print('prob_big_result', prob_big_result)
        if len(prob_small_result) != len(prob_big_result):
            raise ValueError("结果列表长度不一致，无法逐项比较")
        i = 0 
        for index, r in enumerate(prob_small_result):
            small = prob_small_result[index]
            big = prob_big_result[index]

            if small["id"] != big["id"]:
                raise ValueError(f"第 {index} 项 id 不匹配：small_id={small['id']} ≠ big_id={big['id']}")
            if big["prob"] < small["prob"]:
                valid_id= unvalid_id[:index]
                i=index
                break
        else:

            valid_id = unvalid_id
            boundus = checking_outputs[0]['text']
            valid_checking_target_text = unvalid_checking_target_text+boundus
            
            valid_draft_text = valid_draft_text+unvalid_speculative_real_output_text+boundus
            print('all good!--------------------------------------')
            continue
        print('i--------------->',i)

        
        encoded_context_ids = speculative_tokenizer(valid_draft_text)["input_ids"]
        if len(valid_id) > 0:
            encoded_context_ids = torch.cat([
                    torch.tensor(encoded_context_ids, dtype=torch.long)
                    if isinstance(encoded_context_ids, list) else encoded_context_ids,
                    torch.tensor(valid_id, dtype=torch.long)]
                        , dim=0)

            encoded_context_ids = torch.cat([
                    torch.tensor(encoded_context_ids, dtype=torch.long) if isinstance(encoded_context_ids, list) else encoded_context_ids,
                     torch.tensor([potential_ids[i]], dtype=torch.long)
                            ], dim=0)

            print('valid_id', valid_id)
            print('potential_ids', potential_ids)
            print('using big model', potential_ids[i])
            valid_draft_text = speculative_tokenizer.decode(encoded_context_ids, skip_special_tokens=True)

            valid_checking_target_text_ids = target_tokenizer(unvalid_checking_target_text)["input_ids"]
            valid_checking_target_text_ids = torch.cat([
    torch.tensor(valid_checking_target_text_ids[:-(draft_len_output - i)], dtype=torch.long)
        if isinstance(valid_checking_target_text_ids, list) else valid_checking_target_text_ids[:-(draft_len_output - i)],
    torch.tensor([potential_ids[i]], dtype=torch.long)
], dim=0)
            valid_checking_target_text = target_tokenizer.decode(valid_checking_target_text_ids, skip_special_tokens=True)

            
        else:
            print('valid_id', valid_id)
            print('potential_ids', potential_ids)
            print('using big model', potential_ids[i])
            encoded_context_ids = torch.cat([
    torch.tensor(encoded_context_ids, dtype=torch.long)
        if isinstance(encoded_context_ids, list) else encoded_context_ids,
    torch.tensor([potential_ids[i]], dtype=torch.long)
], dim=0)
            valid_draft_text = speculative_tokenizer.decode(encoded_context_ids, skip_special_tokens=True)
            valid_checking_target_text_id = target_tokenizer(valid_checking_target_text)["input_ids"]
            valid_checking_target_text_id = torch.cat([
    torch.tensor(valid_checking_target_text_id, dtype=torch.long)
        if isinstance(valid_checking_target_text_id, list) else valid_checking_target_text_id,
    torch.tensor([potential_ids[i]], dtype=torch.long)
], dim=0)

            valid_checking_target_text = target_tokenizer.decode(valid_checking_target_text_id, skip_special_tokens=True)
        


        end_time = time.time()
        

    real_answer_len = speculative_tokenizer(valid_draft_text, return_tensors="pt")["input_ids"]
    return  valid_draft_text, valid_draft_text, problem,len(real_answer_len),end_time - start_time

def process_file_to_json(dir_path, target_tokenizer, speculative_tokenizer,llm_big,llm_small,problem, answer):
    all_generations = []
    max_new_tokens = 10
    real_answer, full_answer, input_data,full_answer_len,execution_time = speculative_decoding(llm_big,llm_small,target_tokenizer,speculative_tokenizer,problem,answer,max_new_tokens)
    all_generations.append({
        "input_text": input_data,
        "real_answer": real_answer,
        "full_answer": full_answer,
        "tokens_full_answer":full_answer_len,
        "answer": answer,
        "execution_time":execution_time
    })


    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, "generation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_generations, f, ensure_ascii=False, indent=2)

def inference_model_pickle(task_name: str,  base_dir,target_tokenizer,
        speculative_tokenizer,
        llm_small,
        llm_big,
        start=0, end=10,seed=42):
    if task_name == "math-500":
        ds = load_dataset("HuggingFaceH4/MATH-500")['test']
    elif task_name == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    elif args.dataset == "amc23":
        ds = load_dataset("zwhe99/amc23", split="test")
    else:
        raise ValueError(f"Unknown task: {task_name}")

    ds = ds.select(range(start, end))
    if args.dataset == "amc23":
        problems_and_answers = [{"problem": item["question"], "answer": item["answer"]} for item in ds]
    else:
        problems_and_answers = [{"problem": item["problem"], "answer": item["answer"]} for item in ds]

    for idx, number in enumerate(tqdm(range(start, end))):
        dirname = f'seed_{seed}_baseline_{task_name}_{number}'
        dir_path = os.path.join(base_dir, dirname)
        problem = problems_and_answers[idx]['problem']
        answer = problems_and_answers[idx]['answer']
        process_file_to_json(dir_path, target_tokenizer, speculative_tokenizer,llm_big,llm_small,problem, answer)

    print("[Info] Processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", default='amc23')  # math-500
    parser.add_argument("--seed", type=int, help="seed", default=123)
    parser.add_argument("--model", type=str, help="model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--start", type=int, help="start", default=0)
    parser.add_argument("--end", type=int, help="end", default=1)
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":
        model_name = "DeepSeek-R1-Distill-Qwen-32B"
    if args.model == "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit":
        model_name = "DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"
    elif args.model == "Qwen/QwQ-32B-AWQ":
        model_name = "QwQ-32B-AWQ"
    elif args.model == "Qwen/QwQ-32B":
        model_name = "QwQ-32B"
    elif args.model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
        model_name = "DeepSeek-R1-Distill-1.5b"

    Tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"  
    llm_small = sgl.Engine(
    model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    mem_fraction_static=0.7,
    tp_size=1   
    
)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
    llm_big = sgl.Engine(
    model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    mem_fraction_static=0.9,
   tp_size=1 
)   
    target_tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        trust_remote_code=True
    )

    speculative_tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        trust_remote_code=True
    )




    base_dir = f'../baseline/sglang_spec_{model_name}_{args.dataset}_seed{args.seed}/'
    inference_model_pickle(
        task_name=args.dataset,
        base_dir=base_dir,
        target_tokenizer=target_tokenizer,
        speculative_tokenizer=speculative_tokenizer,
        llm_small=llm_small,
        llm_big=llm_big,
        start=args.start,
        end=args.end,
        seed=args.seed,
    )
    print("done")

 