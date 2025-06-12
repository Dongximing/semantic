import transformers
import torch
import argparse
import json 
import os
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import StoppingCriteria, StoppingCriteriaList
MATH_PROMPT = "\nPlease reason step by step, and put your final answer within \\boxed{}."
TARGET_model= 1
SPEC_model = 0
TARGET_probe = 2
SPEC_probe = 3


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer, initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.triggered_stop = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
        for stop in self.stops:
            if stop in generation:
                self.triggered_stop = stop
                return True
        return False
STOP_TOKENS = [
    ' \n\n', '.\n\n', ':\n\n', '\n\n', ' Wait', 'Alternatively', 'Wait', ' But',
    ')\n\n', '?\n\n', ']\n\n', ').\n\n'
]

def generate_with_partial_kv(
        model, tokenizer, input_ids, past_key_values=None, max_new_tokens=10,
        temperature=1.0, top_k=50, top_p=0.95
):
    device = input_ids.device


    if input_ids.numel() == 0 or input_ids.shape[1] == 0:
        raise ValueError("input_ids cannot be empty")

    seq_len = input_ids.shape[1]

    if past_key_values is None:

        if seq_len > 1:
            with torch.no_grad():
                outputs = model(input_ids=input_ids[:, :-1], use_cache=True, return_dict=True)
                past_key_values = outputs.past_key_values
    else:

        cached_len = past_key_values[0][0].shape[2]
        if cached_len < seq_len - 1:
            new_input_ids = input_ids[:, cached_len:-1]
            if new_input_ids.shape[1] > 0:
                with torch.no_grad():
                    outputs = model(input_ids=new_input_ids, past_key_values=past_key_values, use_cache=True,
                                    return_dict=True)
                    past_key_values = outputs.past_key_values

    do_sample = temperature > 0 and (top_k > 0 or top_p < 1.0)

    stopping_criteria_obj = StoppingCriteriaSub(
        stops=STOP_TOKENS,
        initial_length=len(input_ids[0]),
        tokenizer=tokenizer
    )
    stopping_criteria = StoppingCriteriaList([
        stopping_criteria_obj
    ])


    try:
        output = model.generate(
            input_ids=input_ids,
            attention_mask=(input_ids != tokenizer.pad_token_id).long(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
    except Exception as e:
        print(f"Error in model.generate: {e}")
        print(f"past_key_values type: {type(past_key_values)}")
        if past_key_values is not None:
            print(f"past_key_values length: {len(past_key_values)}")
            print(f"first layer shape: {past_key_values[0][0].shape if len(past_key_values) > 0 else 'N/A'}")
    generated_ids = output.sequences
    past_key_values = output.past_key_values

    return generated_ids, past_key_values
def speculative_decoding(target_model, target_tokenizer, speculative_model,speculative_tokenizer,problem, target_temperature,speculative_temperature,max_new_tokens):
    messages = [
        {"role": "user", "content": problem + MATH_PROMPT}
    ]
    generated_ids = target_text = target_tokenizer.apply_chat_template( #big
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    speculative_text = speculative_tokenizer.apply_chat_template( #small
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # we start at the target model
    start_model_inputs = target_tokenizer(target_text, return_tensors="pt").to(target_model.device)
    prompt_len = start_model_inputs.shape[1]
    spec_kv, tgt_kv = None, None
    correct_tokens, try_correct_num = [], 0
    begin = True




    if change_flag:
        try_correct_num = try_correct_num + 1
        generated_ids, tgt_kv = generate_with_partial_kv(
        target_model, target_tokenizer, generated_ids, tgt_kv_candidate,
            max_new_tokens=change_tokens, temperature=0.6, top_k=50, top_p=0.95
        )



    return




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
    parser.add_argument("--data_dir", type=str,  help="data_dir",default='/data/semantic/speculative/spec_reasult_math-500')
    parser.add_argument("--start_dataset", type=str, help="the beginning of the dataset",default=0)
    parser.add_argument("--end_dataset", type=str, help="the end of the dataset",default=1)
    parser.add_argument("--target_probe", type=str, help="target_probe",default="")
    parser.add_argument("--speculative_probe", type=str, help="speculative_probe",default="")
    parser.add_argument("--target_temperature", type=float, help="target_temperature",default=0.1)
    parser.add_argument("--speculative_temperature", type=float, help="speculative_temperature",default=0.6)
    parser.add_argument("--max_new_tokens", type=int, help="max_new_tokens",default=32000)
    parser.add_argument("--top_p", type=float, help="top_p",default=0.9)
    parser.add_argument("--top_k", type=int, help="top_k",default=50)
    parser.add_argument("--target_probe", type=str, help="num_return_sequences",default=1)
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
        dirname = f'spec_{args.dataset}_{number}'
        dir_path = os.path.join(args.data_dir, dirname)
        problem = problems_and_answers[idx]['problem']
        answer = problems_and_answers[idx]['answer']
        process_file_to_json(dir_path, target_model, target_tokenizer,speculative_model, speculative_tokenizer, problem,answer,args.target_temperature,args.speculative_temperature,args.max_new_tokens)