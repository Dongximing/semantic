import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import pandas as pd
import os
import datetime
from datasets import load_dataset
from tqdm import tqdm
MATH_PROMPT = "\nPlease reason step by step, and put your final answer within \\boxed{}."

def inference_model(task_name: str, model, tokenizer):
    if task_name == 'math-500':
        df = pd.read_json(f"/home/shaowei/test.jsonl", lines=True)
        prompts = df['problem'][60:100].tolist()
        answers = df['answer'][60:100].tolist()
        start_number = 60
        result_base_dir = "./qwq32b_math"
    elif task_name == 'aime':
        dataset = load_dataset("AI-MO/aimo-validation-aime")
        dataset = dataset['train']
        new_dataset = dataset.select(range(0,60)) # 2022 2023
        prompts = new_dataset['problem'][:]
        answers = new_dataset['answer'][:]
        start_number = 0
        result_base_dir = "./qwq32b_aime"

    for index, prompt in tqdm(enumerate(prompts), total=len(prompts)):

        if  task_name == 'math-500':
            number = start_number + index
            batch_dir_name = f"data-500-temp0_{number}"
            batch_dir_path = os.path.join(result_base_dir, batch_dir_name)
            file_name = os.path.join(batch_dir_path, f"data-500_{number}.json")
            log_file = os.path.join(batch_dir_path, "log.txt")
        elif task_name == 'aime':
            batch_dir_name = f"data-60-temp0_{index}"
            batch_dir_path = os.path.join(result_base_dir, batch_dir_name)
            file_name = os.path.join(batch_dir_path, f"data-60_{index}.json")
            log_file = os.path.join(batch_dir_path, "log.txt")

        if os.path.exists(file_name):
            print(f"Sample {index} already exists, skip.")
            continue

        os.makedirs(batch_dir_path, exist_ok=True)

        log_dict = {
            "time": str(datetime.datetime.now()),
            "prompt": prompt,
            "answer": answers[index] if index < len(answers) else None
        }
        try:
            messages = [
                {"role": "user", "content": prompt+MATH_PROMPT}
            ]
            formatted_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer([formatted_input], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=8192,
                    temperature=0.6,
                    return_dict_in_generate=True
                )

            input_lengths = [len(ids) for ids in model_inputs['input_ids']]
            gen_only_ids = [seq[inlen:] for seq, inlen in zip(outputs.sequences, input_lengths)]
            input_ids = [seq[:inlen] for seq, inlen in zip(outputs.sequences, input_lengths)]
            input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            response = tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)



            result = {
                "prompt": prompt,
                "answer": answers[index] if index < len(answers) else None,
                "large_model_output": response[0],
                "large_model_output_ids": gen_only_ids[0].tolist() if torch.is_tensor(gen_only_ids[0]) else list(gen_only_ids[0]),
                "large_model_input_ids": input_ids[0].tolist() if torch.is_tensor(input_ids[0]) else list(input_ids[0]),
                "input_prompt_with_template": input_texts[0]
            }
            with open(file_name, 'w') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"Saved {file_name}.")

            log_dict.update({
                "status": "success",
                "output_file": file_name,
                "output_preview": response[0][:120]  
            })

        except Exception as e:
            log_dict.update({
                "status": "error",
                "error_msg": str(e)
            })
            error_log = os.path.join(result_base_dir, "error.log")
            with open(error_log, 'a') as f:
                f.write(f"[{datetime.datetime.now()}] Sample {index} error: {str(e)}\n")


        with open(log_file, 'a') as f:
            f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-model-path",
        type=str,
        default="Qwen/QwQ-32B",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="math-500",
    )
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.main_model_path,
        torch_dtype=torch.float16,
        device_map="auto",

    )
    tokenizer = AutoTokenizer.from_pretrained(args.main_model_path)
    inference_model(task_name=args.task, model=model, tokenizer=tokenizer)

if __name__ == "__main__":
    main()
