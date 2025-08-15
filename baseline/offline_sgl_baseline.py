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

def predict(tokenizer, input_data, model):
    start_time = time.time()
    messages = [
        {"role": "user", "content": input_data + MATH_PROMPT}
    ]
    # apply the pattern for speculative model and target model
    target_text = tokenizer.apply_chat_template(  # big
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 2,
    }
    speculative_output = model.generate(
        target_text, sampling_params=sampling_params, return_hidden_states=True
    )
    print(speculative_output)

    speculative_real_output_text = speculative_output[0]['text']
    end_time = time.time()
    len_output = speculative_output[0]['meta_info']['completion_tokens']
    print(speculative_real_output_text)
    return speculative_real_output_text, target_text+speculative_real_output_text, input_data,len_output,end_time-start_time

def process_file_to_json(save_path, tokenizer, problem, answer,model):
    all_generations = []

    real_answer, full_answer, input_data,full_answer_len,execution_time = predict(tokenizer, problem, model)
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

def inference_model_pickle(task_name: str, tokenizer, base_dir,model,
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
        process_file_to_json(dir_path, tokenizer, problem, answer,model)

    print("[Info] Processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", default='amc23')  # math-500
    parser.add_argument("--seed", type=int, help="seed", default=123)
    parser.add_argument("--model", type=str, help="model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--start", type=int, help="start", default=0)
    parser.add_argument("--end", type=int, help="end", default=30)
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
    llm = sgl.Engine(  model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
,        enable_return_hidden_states=True,
    )



    base_dir = f'/home/cs/staff/shaowei/semantic/baseline/offline_{model_name}_{args.dataset}_seed{args.seed}/'
    inference_model_pickle(
        task_name=args.dataset,
        base_dir=base_dir,
        tokenizer=Tokenizer,
        model=llm,
        start=args.start,
        end=args.end,
        seed=args.seed,

    )
    print("done")
