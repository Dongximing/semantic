import transformers
import random
import argparse
import json
import os
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sglang as sgl
from utils import seed_everything
BEGIN_TOKEN_NUM = 500
SPECULATIVE_OUTPUT_LENGTH = 500
MATH_PROMPT = "\nPlease reason step by step, and put your final answer within \\boxed{}."
TARGET_model= 0
SPEC_model = 1
TARGET_probe = 2
SPEC_probe = 3
import requests
import math 
def speculative_accept(qi, pi, threshold_min=0.9):

    ratio = qi / pi if pi > 0 else 0
    # if ratio < threshold_min:
    #     return False
    threshold = min(1.0, ratio)
    r = random.uniform(0, 1)
    return r < threshold
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


class SemanticEntropyProbTarget(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 512)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(512, 256)  
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout1(h)
        h = F.relu(self.fc2(h))
        h = self.dropout2(h)
        h = F.relu(self.fc3(h))
        out = torch.sigmoid(self.fc4(h))
        return out.squeeze(-1)

class SemanticEntropyProbSpec(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 512)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(512, 256)   
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout1(h)
        h = F.relu(self.fc2(h))
        h = self.dropout2(h)
        h = F.relu(self.fc3(h))
        out = torch.sigmoid(self.fc4(h))
        return out.squeeze(-1)


def speculative_decoding(llm_big,llm_small,target_tokenizer,speculative_tokenizer,problem,max_new_tokens,model_target_probe,model_spec_probe):
        # add prompt before inferencing the model
        time_detial = []
        messages = [
            {"role": "user", "content": problem + MATH_PROMPT}
        ]
        # apply the pattern for speculative model and target model
        target_text = target_tokenizer.apply_chat_template( #big
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        speculative_text = speculative_tokenizer.apply_chat_template( #small
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        sampling_params = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": 500,
            # "min_new_tokens": 50,
            "stop_token_ids": [4710, 382, 1447, 271, 692, 1939, 2533, 3593],
            "no_stop_trim": True
        }

        start_target_model_inputs = target_tokenizer(target_text, return_tensors="pt")
        original_target_prompt_len = start_target_model_inputs["input_ids"].shape[1]


        start_speculative_text_inputs = speculative_tokenizer(speculative_text, return_tensors="pt")
        original_speculative_text_len = start_speculative_text_inputs["input_ids"].shape[1]

        correct_tokens, try_correct_num, correct_spe_number = [], 0, 0
        detail = []
        begin = True
        use_target = True


        def checking_is_finish(generated_ids, max_new_tokens, use_target):
            return len(target_tokenizer.encode(generated_ids)) - original_target_prompt_len < max_new_tokens

        speculative_real_output_text = ''
        prob_target = 0
        prob_spec = 0
        target_real_output = ''
        generated_text = target_text
        break_target = False
        start_time = time.time()
        # print('\n\n')
        while checking_is_finish(generated_text,max_new_tokens,use_target):
            # print('='*50)

            if break_target:
                    break
            # we start at the target model.
            if begin:
                use_target = True
            if not begin:
                if use_target:
                    detail.append({'target_model': target_real_output, 'why_is_not_gd': speculative_real_output_text,
                                   "score_target": round(prob_target, 2), "score_spec": round(prob_spec, 2)})
                    small_input = speculative_text + target_tokenizer.decode(
                        target_tokenizer(generated_text, return_tensors="pt")['input_ids'][0,
                        original_target_prompt_len:].tolist()
                    )
                else:
                    small_input  = generated_text
                # print('----------------------++----------------------')
                # print('generated_text',generated_text)
                # # print('----------------------++----------------------')
                # print('small_input',small_input)
                speculative_outputs_start = time.time()
                json_data = {
                    "text": [small_input],
                    "sampling_params": sampling_params,
                    "return_hidden_states": True,
                    "return_logprob":True
                }
                speculative_outputs = requests.post(
                                    f"http://0.0.0.0:{8006}/generate",
                                    json=json_data,
                    timeout=120
                                     )
                speculative_outputs = speculative_outputs.json()
                
                
                
                prob_small_result = [
                {"id": unvalid_id, "prob": math.exp(lp)}
                for index, (lp, unvalid_id, _) in enumerate(speculative_outputs[0]["meta_info"]['output_token_logprobs'])
            ]
     
                speculative_outputs_time = time.time()-speculative_outputs_start
                time_detial.append({'speculative_outputs_time':speculative_outputs_time})
                speculative_output = speculative_outputs[0]
                speculative_real_output_text = speculative_output['text']
                # print('-------------------------------')
                # print('speculative_real_output_text',speculative_real_output_text)
                if '</think>' in speculative_real_output_text:
                        sampling_params_end = {
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "max_new_tokens": 2000,

                        }
                        generated_text = generated_text+speculative_real_output_text
                        json_data = {
                            "text": [generated_text],
                            "sampling_params": sampling_params_end,
                            "return_hidden_states": False,
                        }

                        speculative_outputs_ending_start = time.time()
                        speculative_outputs = requests.post(
                            f"http://0.0.0.0:{8006}/generate",
                            json=json_data,
                        )
                        speculative_outputs = speculative_outputs.json()
                        speculative_outputs_ending= time.time()-speculative_outputs_ending_start
                        time_detial.append({'speculative_outputs_ending':speculative_outputs_ending})
                        detail.append({'spe_model': speculative_outputs[0]['text']})
                        generated_text = generated_text + speculative_outputs[0]['text']
                        break

                extract_time_small = time.time()
                for i in range(len(speculative_output["meta_info"]["hidden_states"])):
                    speculative_output["meta_info"]["hidden_states"][i] = torch.tensor(
                        speculative_output["meta_info"]["hidden_states"][i], dtype=torch.bfloat16
                    )
                pooling_hidden_information = torch.cat(
                    [
                        i.unsqueeze(0) if len(i.shape) == 1 else i
                        for i in speculative_output["meta_info"]["hidden_states"]
                    ]
                )
                Completion_tokens = speculative_output['meta_info']['completion_tokens']
                pooling_hidden_information = pooling_hidden_information[-Completion_tokens:, :]
                # print('pooling_hidden_information',pooling_hidden_information.shape)
                pooling_hidden_information = pooling_hidden_information.mean(dim=0, keepdim=True)
                exetract_small = time.time() - extract_time_small
                time_detial.append({'exetract_small':exetract_small})
                if len(speculative_real_output_text) ==0:
                    break

                checking_target_text =  generated_text + speculative_real_output_text

                # print('checking_target_text',checking_target_text)
                # else:
                #     checking_target_text =  target_text + target_tokenizer.decode(target_tokenizer(small_input+speculative_real_output_text,return_tensors="pt")['input_ids'][0,original_target_prompt_len:].tolist())
                valid_checking_target_text_len = len(target_tokenizer.encode(generated_text))
                # print('valid_checking_target_text_len',valid_checking_target_text_len)
                checking_start = time.time()

                json_data_check = {
                    "text": [checking_target_text],
                    "sampling_params": {"temperature": 0.1,"max_new_tokens": 1},
                    "return_hidden_states": True,
                    "return_logprob": True,
                    "logprob_start_len":valid_checking_target_text_len-2,
                    "top_logprobs_num":1
                }
                checking_outputs = requests.post(f"http://0.0.0.0:{8005}/generate",
                    json=json_data_check,
                    timeout=120
                )
                checking_outputs = checking_outputs.json()
                potential_ids = extract_potential_ids(checking_outputs[0]['meta_info']['input_top_logprobs'],checking_outputs[0]['meta_info']['input_token_logprobs'], Completion_tokens)
                potential_sets = [set(ids) for ids in potential_ids]
                result = all(small["id"] in potential_sets[i] for i, small in enumerate(prob_small_result))
                if result:
                    # print('accept directly')
                    detail.append({'spe_model':speculative_real_output_text})
                    correct_spe_number +=1
                    use_target = False
                    generated_text =  small_input + speculative_output['text']
                    if '</think>' in speculative_output['text']:
                        sampling_params_end = {
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "max_new_tokens": 2000,

                        }
                        json_data = {
                        "text": [generated_text],
                        "sampling_params": sampling_params_end,
                        "return_hidden_states": False,
                    }
                        speculative_outputs_ending_start = time.time()
                        speculative_outputs = requests.post(
                            f"http://0.0.0.0:{8006}/generate",
                            json=json_data,
                        )
                        speculative_outputs = speculative_outputs.json()
                        
                        
                    
                        speculative_outputs_ending =time.time()-speculative_outputs_ending_start
                        time_detial.append({'speculative_outputs_ending':speculative_outputs_ending})
                        detail.append({'spe_model': speculative_outputs[0]['text']})
                        generated_text = generated_text + speculative_outputs[0]['text']

                        break_target = True
                        break
                    
                    continue
            
                checking_time = time.time()-checking_start
                time_detial.append({'checking_time':checking_time,'Completion_tokens':Completion_tokens,'average_token_ckecing':checking_time/Completion_tokens})
                checking_output = checking_outputs[0]
                extract_time_big = time.time()
                for i in range(len(checking_output["meta_info"]["hidden_states"])):
                    checking_output["meta_info"]["hidden_states"][i] = torch.tensor(
                        checking_output["meta_info"]["hidden_states"][i], dtype=torch.bfloat16
                    )
                hidden_states = torch.cat(
                    [
                        i.unsqueeze(0) if len(i.shape) == 1 else i
                        for i in checking_output["meta_info"]["hidden_states"]
                    ]
                )
                

                computing_prob_start = time.time()
                target_pooling_hidden_information = hidden_states[-Completion_tokens-1:-1, :]
                exetract_big = time.time()-extract_time_big
                time_detial.append({'exetract_big':exetract_big})
                # print('target_pooling_hidden_information shape', target_pooling_hidden_information.shape)
                if target_pooling_hidden_information.shape[0] == 0:
                    break
                target_pooling_hidden_information = target_pooling_hidden_information.mean(dim=0, keepdim=True) # len *hidden
                #print('target_tokenizer_input_len',target_tokenizer_input_len)


                with torch.no_grad():
                    prob_target = model_target_probe(target_pooling_hidden_information.float().to(f"cuda:{6}"))
                    prob_spec = model_spec_probe(pooling_hidden_information.float().to(f"cuda:{6}"))

                prob_target = prob_target.item()
                prob_spec = prob_spec.item()
                # print('prob_target',prob_target)
                # print('prob_spec',prob_spec)
                computing_prob_time = time.time()-computing_prob_start
                time_detial.append({'computing_prob_time':computing_prob_time})
                if speculative_accept(prob_target,prob_spec):
                    # print('accept!!!!!!!')
                    detail.append({'spe_model':speculative_real_output_text})
                    correct_spe_number +=1
                    use_target = False
                    generated_text =  small_input + speculative_output['text'] 
                    if '</think>' in speculative_output['text']:
                        sampling_params_end = {
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "max_new_tokens": 2000,

                        }
                        json_data = {
                            "text": [generated_text],
                            "sampling_params": sampling_params_end,
                            "return_hidden_states": False,
                        }
                        speculative_outputs = requests.post(
                            f"http://0.0.0.0:{8006}/generate",
                            json=json_data,
                            timeout=120
                        )
                        speculative_outputs = speculative_outputs.json()
                        speculative_outputs_ending_start = time.time()
                        
                        speculative_outputs_ending =time.time()-speculative_outputs_ending_start
                        time_detial.append({'speculative_outputs_ending':speculative_outputs_ending})
                        detail.append({'spe_model': speculative_outputs[0]['text']})
                        generated_text = generated_text + speculative_outputs[0]['text']


                        break
                else:
                    # generated_text = checking_target_text
                    use_target = True



            # Let the target model finish the generation.
            # At the beginning of the generation, Let the target model generate the first part of completion.
            if use_target:
                # record the usage of the target model;
                begin = False
                try_correct_num = try_correct_num + 1
                sampling_params_ig = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": 500,
            "stop_token_ids": [4710, 382, 1447, 271, 692, 1939, 2533, 3593],
            "no_stop_trim": True
        }
                target_outputs_start = time.time()
                json_data = {
                    "text": [generated_text],
                    "sampling_params": sampling_params_ig,
                    "return_hidden_states": False,
                }
                target_outputs = requests.post(
                    f"http://0.0.0.0:{8005}/generate", 
                    json=json_data,
                    timeout=120
                )
                # print('generated_text',generated_text)
                target_outputs_time = time.time()-target_outputs_start
                time_detial.append({'target_outputs_time':target_outputs_time})
                target_outputs = target_outputs.json()
                target_real_output = target_outputs[0]['text']
                # print('target_real_output',target_real_output)
                if '</think>' in target_real_output:
                    sampling_params_end = {
                        "temperature": 0.6,
                        "top_p": 0.95,
                        "max_new_tokens": 2000,

                    }
                    small_input = speculative_text + target_tokenizer.decode(
                        target_tokenizer(generated_text, return_tensors="pt")['input_ids'][0,
                        original_target_prompt_len:].tolist()
                    )
                    detail.append({'target_model': target_real_output})
                    json_data = {
                        "text": [small_input+target_real_output],
                        "sampling_params": sampling_params_end,
                        "return_hidden_states": False,
                    }
                    traget_ending_start = time.time()
                    speculative_outputs = requests.post(
                        f"http://0.0.0.0:{8006}/generate",
                        json=json_data,

                    )
                    
                    speculative_outputs = speculative_outputs.json()
                    traget_ending_time = time.time()-traget_ending_start
                    time_detial.append({'traget_ending_time':traget_ending_time})
                    detail.append({'spe_model': speculative_outputs[0]['text']})


                    generated_text = small_input+ target_real_output+ speculative_outputs[0]['text']

                    break
                generated_text = generated_text + target_real_output

                # print('*'*50)


          

                if target_tokenizer.eos_token_id in target_tokenizer.encode(target_real_output):
                    print('target_tokenizer.eos_token_id 281',target_tokenizer.eos_token_id)
                    break

        

        end_time = time.time()


        length_of_output = speculative_tokenizer.encode(generated_text[original_speculative_text_len:])


        return generated_text, try_correct_num,correct_spe_number,detail,len(length_of_output),end_time-start_time,sum([i['target_outputs_time'] for i in time_detial if 'target_outputs_time' in i])+sum([i['checking_time'] for i in time_detial if 'checking_time' in i])+sum([i['computing_prob_time'] for i in time_detial if 'computing_prob_time' in i])+sum([i['speculative_outputs_time'] for i in time_detial if 'speculative_outputs_time' in i])+sum([i['speculative_outputs_ending'] for i in time_detial if 'speculative_outputs_ending' in i])+sum([i['traget_ending_time'] for i in time_detial if 'traget_ending_time' in i]),time_detial





def process_file_to_json(
    dir_path,
    llm_big,
    llm_small,
    target_tokenizer,
    speculative_tokenizer,
    problem,
    answer,
    max_new_tokens,
    model_target_probe,
    model_spec_probe,
    idx,
):
    all_generations = []
    failed_list = []

    # try:
        # start_time = time.time()

    result = speculative_decoding(
        llm_big,
        llm_small,
        target_tokenizer,
        speculative_tokenizer,
        problem,
        max_new_tokens,
        model_target_probe,
        model_spec_probe,
    )

    # end_time = time.time()

    generated_text, try_correct_num, correct_spe_number, detail, length_of_output,times,total_time,time_detial = result
    # print("real_answer\n", generated_text)

    all_generations.append({
        "input_text": problem,
        "real_answer": generated_text,
        "try_correct_num": try_correct_num,
        "standard_answer": answer,
        "execution_time": f"{times:.2f}s",
        "correct_spe_number": correct_spe_number,
        "total_time": f"{total_time:.2f}s",
        "time_detail": time_detial,
        "detail": detail,
        "length_of_output": length_of_output,
        "index": idx
    })

    os.makedirs(dir_path, exist_ok=True)
    out_path = os.path.join(dir_path, "spec_generation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_generations, f, ensure_ascii=False, indent=2)

    # except Exception as e:
    #     print(f"[Index {idx}] Failed with error: {e}")
    #     print("Sleeping 10 seconds before moving on...")
    #     time.sleep(1)
    #     failed_list.append(idx)
    return failed_list










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,  help="dataset",default='aime')#math-500
    parser.add_argument("--target_model", type=str,  help="target_model",default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--speculative_model", type=str,  help="speculative_model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--data_dir", type=str,  help="data_dir",default='../speculative/improvement1')
    parser.add_argument("--start_dataset", type=int, help="the beginning of the dataset",default=0)
    parser.add_argument("--end_dataset", type=int, help="the end of the dataset",default=30)
    parser.add_argument("--target_probe", type=str, help="target_probe",default="/home/ximing/semantic/speculative/weight/s1_valid_h100_32r1b-200data_math_output_last_hidden_list_best_probe_mse")#aime_output_last_hidden_list_best_probe_mse
    parser.add_argument("--speculative_probe", type=str, help="speculative_probe",default="/home/ximing/semantic/speculative/weight/s1_valid_h100_r1.5b_math_output_last_hidden_list_best_probe_mse")
    parser.add_argument("--target_temperature", type=float, help="target_temperature",default=0.1)
    parser.add_argument("--speculative_temperature", type=float, help="speculative_temperature",default=0.6)
    parser.add_argument("--max_new_tokens", type=int, help="max_new_tokens",default=14000)
    parser.add_argument("--top_p", type=float, help="top_p",default=0.9)
    parser.add_argument("--top_k", type=int, help="top_k",default=50)
    parser.add_argument("--seed", type=int, help="seed", default=6540)
    args = parser.parse_args()
    seed_everything(args.seed)
    # from sglang.srt.server_args import ServerArgs
    # print(ServerArgs.__init__.__annotations__)
 

    model_target_probe = SemanticEntropyProbTarget(5120, 2048)
    model_target_probe.load_state_dict(torch.load(f'{args.target_probe}.pt'))
    model_target_probe = model_target_probe.to('cuda:6')
    model_target_probe.eval()


    model_spec_probe = SemanticEntropyProbSpec(1536, 1024)
    model_spec_probe.load_state_dict(torch.load(f'{args.speculative_probe}.pt'))
    model_spec_probe = model_spec_probe.to('cuda:6')
    model_spec_probe.eval()


    llm_small = None


    llm_big = None


    target_tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.target_model,
        trust_remote_code=True
    )

    speculative_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.speculative_model,
        trust_remote_code=True
    )

    if speculative_tokenizer.pad_token_id is None:
        speculative_tokenizer.pad_token_id = speculative_tokenizer.eos_token_id
    if args.dataset == "math-500":
        ds = load_dataset("HuggingFaceH4/MATH-500")['test']
    elif args.dataset == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    elif args.dataset == "amc23":
        ds = load_dataset("zwhe99/amc23", split="test")
    elif args.dataset == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", split="train")

    else:
        raise ValueError(f"Unknown task: {args.dataset}")

    ds = ds.select(range(args.start_dataset, args.end_dataset))
    if args.dataset == "amc23":
        problems_and_answers = [{"problem": item["question"], "answer": item["answer"]} for item in ds]
    elif args.dataset == "gpqa": 
        problems_and_answers = [{"problem": item["question"], "answer": item["answer"]} for item in ds]
    else:
        problems_and_answers = [{"problem": item["problem"], "answer": item["answer"]} for item in ds]



    for idx, number in enumerate(tqdm(range(args.start_dataset, args.end_dataset))):
        dirname = f'spec_{args.dataset}_{number}'
        dir_path = os.path.join(f"{args.dataset}{args.data_dir}{args.seed}", dirname)
        problem = problems_and_answers[idx]['problem']
        answer = problems_and_answers[idx]['answer']
        failed = process_file_to_json(dir_path,llm_big,llm_small, target_tokenizer, speculative_tokenizer,problem,answer,args.max_new_tokens,model_target_probe,model_spec_probe,number)
        # failed_total.extend(failed)


        # [{'text': 'First', 'output_ids': [3491, 3019, 553, 3019, 382, 5338], 'meta_info': {'id': '80704614f4674319b5da35d79a2a2a18', 'finish_reason': {'type': 'length', 'length': 1}, 'prompt_tokens': 184, 'weight_version': 'default', 'input_token_logprobs': [[None, 382, None]], 'output_token_logprobs': [[-0.014046513475477695, 5338, None]], 'completion_tokens': 1, 'cached_tokens': 158, 'e2e_latency': 0.019438505172729492}}]