import argparse
import json
import os
import random
import time
import traceback

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_GPQA_multiple_choice_answers

MATH_PROMPT = "\nPlease reason step by step, and put your final answer within \\boxed{}."
DATASET_CHOICES = ["math-500", "aime", "amc23", "gpqa"]
LOCAL_GPQA_PATH = "/home/semantic/baseline/gpqa"
MODEL_ALIASES = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-R1-Distill-Qwen-32B",
    "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit": "DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/QwQ-32B-AWQ": "QwQ-32B-AWQ",
    "Qwen/QwQ-32B": "QwQ-32B",
}


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="math-500", choices=DATASET_CHOICES)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=14000)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--max_memory_gpu", type=str, default="79GB")
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()


def build_prompt(problem, dataset_name):
    if dataset_name == "gpqa":
        return problem
    return problem + MATH_PROMPT


def predict(tokenizer, model, problem, dataset_name, temperature, max_new_tokens, device):
    messages = [{"role": "user", "content": build_prompt(problem, dataset_name)}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    execution_time = time.time() - start_time

    full_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    real_answer = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return {
        "real_answer": real_answer,
        "full_answer": full_answer,
        "tokens_full_answer": int(outputs.shape[1]),
        "execution_time": execution_time,
    }


def write_generation(save_path, payload):
    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, "generation.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump([payload], handle, ensure_ascii=False, indent=2)


def load_dataset_slice(dataset_name, start, end):
    if dataset_name == "math-500":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
        dataset = dataset.select(range(start, end))
        return [{"index": start + i, "problem": item["problem"], "answer": item["answer"]} for i, item in enumerate(dataset)]

    if dataset_name == "aime":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        dataset = dataset.select(range(start, end))
        return [{"index": start + i, "problem": item["problem"], "answer": item["answer"]} for i, item in enumerate(dataset)]

    if dataset_name == "amc23":
        dataset = load_dataset("zwhe99/amc23", split="test")
        dataset = dataset.select(range(start, end))
        return [{"index": start + i, "problem": item["question"], "answer": item["answer"]} for i, item in enumerate(dataset)]

    if os.path.exists(LOCAL_GPQA_PATH):
        loaded = load_dataset(LOCAL_GPQA_PATH, "gpqa_diamond")
    else:
        loaded = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
    dataset = loaded["train"].select(range(start, end))
    rows = dataset.to_pandas().to_dict("records")
    problems = []
    for offset, row in enumerate(rows):
        options, correct_answer = get_GPQA_multiple_choice_answers(row)
        problems.append(
            {
                "index": start + offset,
                "problem": (
                    "Return your final response within \\boxed{{}} and only include the letter choice "
                    "(A, B, C, or D) as your final response. "
                    f"{row['Question']}\n{options}"
                ),
                "answer": correct_answer,
            }
        )
    return problems


def get_model_name(model_name):
    return MODEL_ALIASES.get(model_name, model_name.split("/")[-1])


def get_output_dir(args):
    if args.output_dir:
        return args.output_dir
    model_name = get_model_name(args.model)
    return f"/home/{model_name}_{args.dataset}_seed{args.seed}"


def main():
    args = parse_args()
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={int(args.device.split(":")[-1]): args.max_memory_gpu},
    )

    output_dir = get_output_dir(args)
    problems = load_dataset_slice(args.dataset, args.start, args.end)

    for item in tqdm(problems):
        dirname = f"seed_{args.seed}_baseline_{args.dataset}_{item['index']}"
        save_path = os.path.join(output_dir, dirname)
        try:
            result = predict(
                tokenizer=tokenizer,
                model=model,
                problem=item["problem"],
                dataset_name=args.dataset,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
            payload = {
                "input_text": item["problem"],
                "real_answer": result["real_answer"],
                "full_answer": result["full_answer"],
                "tokens_full_answer": result["tokens_full_answer"],
                "answer": item["answer"],
                "execution_time": result["execution_time"],
            }
        except Exception:
            payload = {
                "input_text": item["problem"],
                "real_answer": None,
                "full_answer": None,
                "tokens_full_answer": None,
                "answer": item["answer"],
                "error": traceback.format_exc(),
            }
        write_generation(save_path, payload)


if __name__ == "__main__":
    main()
