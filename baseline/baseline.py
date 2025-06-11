import datetime
import torch
import json
import os
import traceback
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

NUMBER = 2

def predict(tokenizer, model, input_data, temperature):
    max_new_tokens = 15000
    inputs = tokenizer(input_data, return_tensors="pt").to(f"cuda:{NUMBER}")
    initial_length = len(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

    full_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    real_answer = tokenizer.decode(outputs[0][initial_length:], skip_special_tokens=True)
    return real_answer, full_answer, input_data

def process_file_to_json(save_path, tokenizer, model, problem, answer):
    all_generations = []
    try:
        real_answer, full_answer, input_data = predict(tokenizer, model, problem, temperature=0.1)
        all_generations.append({
            "input_text": input_data,
            "real_answer": real_answer,
            "full_answer": full_answer,
            "answer": answer
        })
    except Exception as e:
        all_generations.append({
            "input_text": problem,
            "real_answer": None,
            "full_answer": None,
            "answer": answer,
            "error": traceback.format_exc()
        })

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, "generation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_generations, f, ensure_ascii=False, indent=2)

def inference_model_pickle(task_name: str, model, tokenizer, base_dir,
                           start=0, end=10):
    if task_name == "math-500":
        ds = load_dataset("HuggingFaceH4/MATH-500")["train"]
    elif task_name == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    else:
        raise ValueError(f"Unknown task: {task_name}")

    ds = ds.select(range(start, end))
    problems_and_answers = [{"problem": item["problem"], "answer": item["answer"]} for item in ds]

    for idx, number in enumerate(tqdm(range(start, end))):
        dirname = f'baseline_{task_name}_{number}'
        dir_path = os.path.join(base_dir, dirname)
        problem = problems_and_answers[idx]['problem']
        answer = problems_and_answers[idx]['answer']
        process_file_to_json(dir_path, tokenizer, model, problem, answer)

    print("[Info] Processing completed.")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/QwQ-32B-AWQ",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/QwQ-32B-AWQ",
        torch_dtype=torch.float16,
        device_map=f"cuda:{NUMBER}"
    )

    base_dir = '/data/ximing/semantic'
    inference_model_pickle(
        task_name="math-500",
        model=model,
        tokenizer=tokenizer,
        base_dir=base_dir,
        start=100,
        end=200
    )
    print("done")
