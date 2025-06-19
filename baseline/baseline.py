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
SEED = 42
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
    max_new_tokens = 15000
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

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

    full_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    full_answer_len = outputs.shape[1]
    real_answer = tokenizer.decode(outputs[0][initial_length:], skip_special_tokens=True)
    return real_answer, full_answer, input_data,full_answer_len

def process_file_to_json(save_path, tokenizer, model, problem, answer):
    all_generations = []
    try:
        real_answer, full_answer, input_data,full_answer_len = predict(tokenizer, model, problem, temperature=0.6)
        all_generations.append({
            "input_text": input_data,
            "real_answer": real_answer,
            "full_answer": full_answer,
            "tokens_full_answer":full_answer_len,
            "answer": answer
        })
    except Exception as e:
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

    for idx, number in enumerate(tqdm(range(start, end))):
        dirname = f'seed_{seed}_baseline_{task_name}_{number}'
        dir_path = os.path.join(base_dir, dirname)
        problem = problems_and_answers[idx]['problem']
        answer = problems_and_answers[idx]['answer']
        process_file_to_json(dir_path, tokenizer, model, problem, answer)

    print("[Info] Processing completed.")

if __name__ == "__main__":
    seed_everything(123)
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        torch_dtype=torch.float16,
        device_map=f"cuda:{NUMBER}"
    )

    base_dir = '/home/cs/staff/shaowei/semantic/r1_1.5B_baseline_math_500_seed123/'
    inference_model_pickle(
        task_name="math-500",
        model=model,
        tokenizer=tokenizer,
        base_dir=base_dir,
        start=100,
        end=500,
        seed=SEED
    )
    print("done")
