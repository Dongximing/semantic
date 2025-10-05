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

def speculative_accept(qi, pi, threshold_min=0.7):
    ratio = qi / pi if pi > 0 else 0
    threshold = min(1.0, ratio)
    r = random.uniform(0, 1)
    
    # print(f"qi: {qi}, pi: {pi}, ratio: {ratio}, threshold: {threshold}, r: {r}")
    return r < threshold
NUMBER = 0
def extract_potential_ids(input_top_logprobs, input_token_logprobs, draft_len_output):
    """
    返回形如 [[top1_id, top2_id], ...] 的列表，长度为 draft_len_output。
    回退规则：
      - 若 top 为 None/空：用对应的 token 元组里的 id（若也拿不到则 0），并复制两次 [id, id]
      - 若 top 有候选但缺少第二个候选：第二个用第一个顶上（复制）
      - 若某个候选的结构不完整（不是 tuple/list 或长度不足）：跳过并回退到 token 的 id
    """
    # 取最后 draft_len_output 个，逐位置对齐
    last_top  = input_top_logprobs[-draft_len_output:]
    last_tok  = input_token_logprobs[-draft_len_output:]
    potential_ids = []

    def get_token_id_from_tuple(t):
        # 期望 t 形如 (logprob, token_id, ...)
        return t[1] if isinstance(t, (list, tuple)) and len(t) > 1 else None

    for top, tok in zip(last_top, last_tok):
        # 先尝试从 top（top-k 列表）里拿前两个候选的 id
        top1_id = top2_id = None

        if top and isinstance(top, (list, tuple)) and len(top) > 0:
            # 第一个候选
            head1 = top[0]
            if isinstance(head1, (list, tuple)) and len(head1) > 1:
                top1_id = head1[1]

            # 第二个候选（可能没有）
            if len(top) > 1:
                head2 = top[1]
                if isinstance(head2, (list, tuple)) and len(head2) > 1:
                    top2_id = head2[1]

        # 若 top1 为空，回退到 token 的 id
        if top1_id is None:
            top1_id = get_token_id_from_tuple(tok)

        # 若 top2 为空，优先复制 top1；再不行回退到 token 的 id；再不行用 0
        if top2_id is None:
            top2_id = top1_id if top1_id is not None else get_token_id_from_tuple(tok)

        # 最后兜底
        if top1_id is None:
            top1_id = 0
        if top2_id is None:
            top2_id = top1_id  # 复制

        potential_ids.append([top1_id, top2_id])

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
        "max_new_tokens": 20,

    }
    break_flag = False
    def checking_is_finish(generated_ids, max_new_tokens):
        # print(len(speculative_tokenizer.encode(generated_ids)))
        return len(speculative_tokenizer.encode(generated_ids))  < max_new_tokens
    checking_sampling_params = {"temperature": 0.1,"max_new_tokens": 1}
    # print('\n')
    prev_valid_draft_text = None # 用于存储上一次的 valid_draft_text
    lens = []
    while checking_is_finish(max_new_tokens=14000, generated_ids=valid_draft_text):
        lens.append(len(speculative_tokenizer.encode(valid_draft_text)))
        if len(lens) >= 10 and lens[-1] == lens[-2]:
            print("No progress made, stopping to avoid infinite loop.")
            break

        speculative_output = llm_small.generate(
            [valid_draft_text], sampling_params=sampling_params, return_logprob=True
        )
        unvalid_speculative_real_output_text = speculative_output[0]['text']
        
        
        draft_len_output = speculative_output[0]['meta_info']['completion_tokens']
        # print(speculative_output[0]['meta_info']['output_token_logprobs'])
        # print('unvalid_speculative_real_output_text',unvalid_speculative_real_output_text)
        unvalid_id = speculative_tokenizer(valid_draft_text+unvalid_speculative_real_output_text)["input_ids"][-draft_len_output:]
        # print('draft_len_output',draft_len_output)
        # print('unvalid_id',unvalid_id)
        prob_small_result = [
                {"id": unvalid_id[index], "prob": math.exp(lp)}
                for index, (lp, _, _) in enumerate(speculative_output[0]['meta_info']['output_token_logprobs'])
            ]

        # print('prob_small_result:',prob_small_result)
        for item in prob_small_result:
            if item["id"] == 151643:
                valid_draft_text = valid_draft_text + unvalid_speculative_real_output_text
                break_flag = True
        if break_flag:
            break

        unvalid_checking_target_text = valid_checking_target_text+unvalid_speculative_real_output_text
        # print('\n')
        # print('------------------------------------------')
        # print('unvalid_checking_target_text:', unvalid_checking_target_text)
        valid_checking_target_text_len = target_tokenizer(valid_checking_target_text, return_tensors="pt")["input_ids"].shape[1]
        # print('valid_checking_target_text_len',valid_checking_target_text_len)
        checking_outputs = llm_big.generate([unvalid_checking_target_text],
                        sampling_params = {"temperature": 0.6,"max_new_tokens": 1},
                        return_logprob=True,
                        logprob_start_len=valid_checking_target_text_len-2,top_logprobs_num=2
            
                    )
        # print('checking_outputs',checking_outputs)
        potential_ids = extract_potential_ids(checking_outputs[0]['meta_info']['input_top_logprobs'],checking_outputs[0]['meta_info']['input_token_logprobs'], draft_len_output)
        input_top_logprobs = checking_outputs[0]['meta_info']['input_top_logprobs']

        last_top_logprobs = input_top_logprobs[-draft_len_output:]


        prob_big_result_big1 = []
        for top, (lp_tok, tid_tok, _) in zip(last_top_logprobs, checking_outputs[0]['meta_info']['input_token_logprobs'][-draft_len_output:]):
            if not top or top[0] is None:
            # top 是 None，用 token_logprobs 的 tid，prob=0.0
                prob_big_result_big1.append({"id": tid_tok, "prob": 0.0})
            else:
                lp, tid = top[0][0], top[0][1]
                prob = math.exp(lp) if lp is not None else 0.0
                prob_big_result_big1.append({"id": tid, "prob": prob})
        
        prob_big_result = [
            {"id": tid, "prob": math.exp(lp) if lp is not None else 0.0}
            for lp, tid, _ in checking_outputs[0]['meta_info']['input_token_logprobs'][-draft_len_output:]
        ]
        # print('prob_big_result', len(prob_big_result))
        # print('prob_small_result', len(prob_small_result))
        if len(prob_small_result) != len(prob_big_result):
            raise ValueError("结果列表长度不一致，无法逐项比较")
        i = 0
        valid_id = []
        for index, small in enumerate(prob_small_result):
            big = prob_big_result[index]
            big1 = prob_big_result_big1[index]


            # 只要 small 的 id 和 big / big1 的 id 有一个对得上就接受
            if small["id"] in potential_ids[index]:
                # print(f'small id in big model top2 {small["id"]} in {potential_ids[index]}')
                continue
            elif small["id"] == big["id"]:
                big_prob = big["prob"]
            else:
                raise ValueError(f"第 {index} 项 id 不匹配：small_id={small['id']} ≠ big_id={big['id']} ≠ big1_id={big1['id']}")

            # 判断是否接受该 token
            if not speculative_accept(big_prob, small["prob"]):
                #print(f"Token not accepted at index {index}: small_id={small['id']}, small_prob={small['prob']}, big_id={big['id']}, big_prob={big_prob}, big1_id={big1['id']}, big1_prob={big1['prob']}")
                valid_id = unvalid_id[:index]  # 截断
                i = index
                break
        else:

            valid_id = unvalid_id
            boundus = checking_outputs[0]['text']
            valid_checking_target_text =unvalid_checking_target_text+boundus
            valid_draft_text = valid_draft_text+unvalid_speculative_real_output_text+boundus
            # print('all good!--------------------------------------')
            continue
        # print('i--------------->',i)

        
        encoded_context_ids = speculative_tokenizer(valid_draft_text)["input_ids"]
        if len(valid_id) > 0:
            # print('replace small model output with big model output',potential_ids[i])
            encoded_context_ids = torch.cat([
                    torch.tensor(encoded_context_ids, dtype=torch.long)
                    if isinstance(encoded_context_ids, list) else encoded_context_ids,
                    torch.tensor(valid_id, dtype=torch.long)]
                        , dim=0)

            encoded_context_ids = torch.cat([
                    torch.tensor(encoded_context_ids, dtype=torch.long) if isinstance(encoded_context_ids, list) else encoded_context_ids,
                     torch.tensor([potential_ids[i][0]], dtype=torch.long)
                            ], dim=0)

            valid_draft_text = speculative_tokenizer.decode(encoded_context_ids, skip_special_tokens=True)

            valid_checking_target_text_ids = target_tokenizer(unvalid_checking_target_text)["input_ids"]
            valid_checking_target_text_ids = torch.cat([
    torch.tensor(valid_checking_target_text_ids[:-(draft_len_output - i)], dtype=torch.long)
        if isinstance(valid_checking_target_text_ids, list) else valid_checking_target_text_ids[:-(draft_len_output - i)],
    torch.tensor([potential_ids[i][0]], dtype=torch.long)
], dim=0)
            valid_checking_target_text = target_tokenizer.decode(valid_checking_target_text_ids, skip_special_tokens=True)

            
        else:
            # print('valid_id', valid_id)
            # print('potential_ids', potential_ids)
            # print('using big model', potential_ids[i])
            # print('replace small model output with big model output',potential_ids[i])
            encoded_context_ids = torch.cat([
    torch.tensor(encoded_context_ids, dtype=torch.long)
        if isinstance(encoded_context_ids, list) else encoded_context_ids,
    torch.tensor([potential_ids[i][0]], dtype=torch.long)
], dim=0)
            valid_draft_text = speculative_tokenizer.decode(encoded_context_ids, skip_special_tokens=True)
            valid_checking_target_text_id = target_tokenizer(valid_checking_target_text)["input_ids"]
            valid_checking_target_text_id = torch.cat([
    torch.tensor(valid_checking_target_text_id, dtype=torch.long)
        if isinstance(valid_checking_target_text_id, list) else valid_checking_target_text_id,
    torch.tensor([potential_ids[i][0]], dtype=torch.long)
], dim=0)

            valid_checking_target_text = target_tokenizer.decode(valid_checking_target_text_id, skip_special_tokens=True)
        


        end_time = time.time()
        

    real_answer_len = speculative_tokenizer(valid_draft_text, return_tensors="pt")["input_ids"]
    return  valid_draft_text, valid_draft_text, problem,real_answer_len.shape[1],end_time - start_time

def process_file_to_json(save_path, target_tokenizer, speculative_tokenizer,llm_big,llm_small,problem, answer):
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
    parser.add_argument("--seed", type=int, help="seed", default=3210)
    parser.add_argument("--model", type=str, help="model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--start", type=int, help="start", default=20)
    parser.add_argument("--end", type=int, help="end", default=21)
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
    llm_small = sgl.Engine(
    model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    mem_fraction_static=0.3,
    tp_size=1   
    
)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
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




    base_dir = f'../baseline/testsglang_spec_{model_name}_{args.dataset}_seed{args.seed}/'
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

 


 