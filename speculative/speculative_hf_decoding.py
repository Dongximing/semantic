import transformers
import torch
import argparse
import json 
import os
import time
from tqdm import tqdm
from datasets import load_dataset

TARGET_model= 1
SPEC_model = 0
TARGET_probe = 2
SPEC_probe = 3
def speculative_decoding(target_model, target_tokenizer, speculative_model,speculative_tokenizer,problem, target_temperature,speculative_temperature,max_new_tokens):
    messages = [
        {"role": "user", "content": problem}
    ]
    target_text = target_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(target_text)

    model_inputs = target_tokenizer([target_text], return_tensors="pt").to(target_model.device)




    pass
def process_file_to_json(dir_path, target_model, target_tokenizer,speculative_model, speculative_tokenizer,problem, answer,target_temperature,speculative_temperature,max_new_tokens):
    all_generations = []
    try:
        start_time = time.time()
        result = speculative_decoding(target_model, target_tokenizer, speculative_model, speculative_tokenizer, problem, target_temperature, speculative_temperature,max_new_tokens)
        end_time = time.time()
        real_answer, full_answer, input_data = result
        all_generations.append({
            "input_text": input_data,
            "real_answer": real_answer,
            "full_answer": full_answer,
            "answer": answer,
            "execution_time": f"{end_time - start_time:.2f}s"
        })
    except Exception as e:
        all_generations.append({
            "input_text": problem,
            "real_answer": None,
            "full_answer": None,
            "answer": answer,
        })

    os.makedirs(dir_path, exist_ok=True)
    out_path = os.path.join(dir_path, "spec_generation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_generations, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,  help="dataset",default='math-500')
    parser.add_argument("--target_model", type=str,  help="target_model",default="Qwen/QwQ-32B-AWQ")
    parser.add_argument("--speculative_model", type=str,  help="speculative_model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--data_dir", type=str,  help="data_dir")
    parser.add_argument("--start_dataset", type=str, help="the beginning of the dataset",default=0)
    parser.add_argument("--end_dataset", type=str, help="the end of the dataset",default=1)
    parser.add_argument("--target_probe", type=str, help="target_probe",default="")
    parser.add_argument("--speculative_probe", type=str, help="speculative_probe",default="")
    parser.add_argument("--target_temperature", type=float, help="target_temperature",default=0.1)
    parser.add_argument("--speculative_temperature", type=float, help="speculative_temperature",default=0.6)
    parser.add_argument("--max_new_tokens", type=int, help="max_new_tokens",default=32000)
    parser.add_argument("--top_p", type=float, help="top_p",default=0.9)
    parser.add_argument("--top_k", type=int, help="top_k",default=50)
    args = parser.parse_args()
    target_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
        device_map=f"cuda:{TARGET_model}"
    )
    target_tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.target_model,
        trust_remote_code=True
    )
    speculative_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.speculative_model,
        torch_dtype=torch.float16,
        device_map=f"cuda:{SPEC_model}"

    )
    speculative_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.speculative_model,
        trust_remote_code=True
    )
    if args.dataset == "math-500":
        ds = load_dataset("HuggingFaceH4/MATH-500")['test']

    elif args.dataset == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    else:
        raise ValueError(f"Unknown task: {args.dataset}")

    ds = ds.select(range(args.start_dataset, args.end_dataset))
    problems_and_answers = [{"problem": item["problem"], "answer": item["answer"]} for item in ds]

    for idx, number in enumerate(tqdm(range(args.start_dataset, args.end_dataset))):
        dirname = f'baseline_{args.task_name}_{number}'
        dir_path = os.path.join(args.data_dir, dirname)
        problem = problems_and_answers[idx]['problem']
        answer = problems_and_answers[idx]['answer']
        process_file_to_json(dir_path, target_model, target_tokenizer,speculative_model, speculative_tokenizer, problem,answer,args.target_temperature,args.speculative_temperature,args.max_new_tokens)