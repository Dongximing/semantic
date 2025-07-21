import datetime
import torch
import json
import os
import traceback
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import numpy as np
import random
import torch
import time

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

def predict(tokenizer, model, input_data, temperature):
    max_new_tokens = 2000
    messages = [
        {"role": "user", "content": input_data + MATH_PROMPT}
    ]
    # apply the pattern for speculative model and target model
    target_text = tokenizer.apply_chat_template(  # big
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(target_text, return_tensors="pt").to(f"cuda:{NUMBER}")
    initial_length = len(inputs['input_ids'][0])
    start_time  = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    execution_time = time.time() - start_time

    full_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    full_answer_len = outputs.shape[1]
    real_answer = tokenizer.decode(outputs[0][initial_length:], skip_special_tokens=True)
    return real_answer, full_answer, input_data,full_answer_len,execution_time

def process_file_to_json(save_path, tokenizer, model, problem, answer):
    all_generations = []
    try:
        real_answer, full_answer, input_data,full_answer_len,execution_time = predict(tokenizer, model, problem, temperature=0.6)
        all_generations.append({
            "input_text": input_data,
            "real_answer": real_answer,
            "full_answer": full_answer,
            "tokens_full_answer":full_answer_len,
            "answer": answer,
            "execution_time":execution_time
        })
    except Exception as e:
        print('ggggg')
        all_generations.append({
            "input_text": problem,
            "real_answer": None,
            "full_answer": None,
            "answer": answer,
            "tokens_full_answer":None,
            "error": traceback.format_exc()
        })

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, "generation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_generations, f, ensure_ascii=False, indent=2)

def inference_model_pickle(task_name: str, model, tokenizer, base_dir,
                           start=0, end=10,seed=42):
    if task_name == "math-500":
        ds = load_dataset("HuggingFaceH4/MATH-500")['test']

    elif task_name == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    else:
        raise ValueError(f"Unknown task: {task_name}")

    ds = ds.select(range(start, end))
    problems_and_answers = [{"problem": item["problem"], "answer": item["answer"]} for item in ds]

    for idx, number in enumerate(tqdm(range(start, start+1))):
        dirname = f'test_2000tokens_baseline_{task_name}_{number}'
        dir_path = os.path.join(base_dir, dirname)
        problem = problems_and_answers[idx]['problem']
        answer = problems_and_answers[idx]['answer']
        process_file_to_json(dir_path, tokenizer, model, problem, answer)

    print("[Info] Processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", default='aime')  # math-500
    parser.add_argument("--seed", type=int, help="seed", default=123)
    parser.add_argument("--model", type=str, help="model", default="Qwen/QwQ-32B-AWQ")
    parser.add_argument("--start", type=int, help="start", default=4)
    parser.add_argument("--end", type=int, help="end", default=30)
    args = parser.parse_args()
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if args.model =="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":
      model_name = "DeepSeek-R1-Distill-Qwen-32B"
    elif args.model == "Qwen/QwQ-32B-AWQ":
      model_name ="QwQ-32B-AWQ"
    elif args.model == "Qwen/QwQ-32B":
        model_name ="QwQ-32B"
    elif args.model == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
        model_name ="DeepSeek-R1-Distill-Qwen-1.5B"
    elif args.model == "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit":
        model_name ="DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"
    wrong_list = [21, 25, 27, 28, 29]
    base_dir = f'test_{model_name}_{args.dataset}_seed{args.seed}/'
    for idx, number in enumerate(tqdm(wrong_list, total=len(wrong_list))):


        inference_model_pickle(
            task_name=args.dataset,
            model=model,
            tokenizer=tokenizer,
            base_dir=base_dir,
            start=args.start,
            end=args.end,
            seed=args.seed
        )
    print("done")
