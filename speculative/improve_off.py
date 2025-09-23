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

def speculative_accept(qi, pi, threshold_min=0.7):

    ratio = qi / pi if pi > 0 else 0
    # if ratio < threshold_min:
    #     return False
    threshold = min(1.0, ratio)
    r = random.uniform(0, 1)
    return r < threshold


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
        keep_doing =False
        generated_text = target_text
        start_time = time.time()
        out_of_loop = False
        while checking_is_finish(generated_text,max_new_tokens,use_target):
            # we start at the target model.
            save_probe = [] 
            if begin:
                use_target = True
            if not begin:
                if use_target:
                 
                    small_input = speculative_text + target_tokenizer.decode(
                        target_tokenizer(generated_text, return_tensors="pt")['input_ids'][0,
                        original_target_prompt_len:].tolist()
                    )
                else:
                    small_input  = generated_text
                check_small_input = small_input
                checking_list = [] 
                if use_target:
                    checking_target_text =  generated_text
                else:
                    checking_target_text =  target_text
                for i in range(3):    
                    speculative_outputs_start = time.time()
                    speculative_outputs = llm_small.generate([small_input], sampling_params=sampling_params, return_hidden_states=True)
                    speculative_outputs_time = time.time()-speculative_outputs_start
                    time_detial.append({'speculative_outputs_time':speculative_outputs_time})
                    speculative_output = speculative_outputs[0]
                    speculative_real_output_text = speculative_output['text']
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
                    pooling_hidden_information = pooling_hidden_information.mean(dim=0, keepdim=True)

                    small_input+= speculative_real_output_text
                    if len(speculative_real_output_text) ==0:
                        out_of_loop = True
                        break
                    if '</think>' in speculative_real_output_text:
                        sampling_params_end = {
                                "temperature": 0.6,
                                "top_p": 0.95,
                                "max_new_tokens": 2000,

                            }

                        speculative_outputs_ending_start = time.time()
                        speculative_outputs = llm_small.generate(
                            [generated_text],
                            sampling_params = sampling_params_end,
                            return_hidden_states =False,     
                        )
                        speculative_outputs_ending =time.time()-speculative_outputs_ending_start
                        time_detial.append({'speculative_outputs_ending':speculative_outputs_ending})
                        detail.append({'spe_model': speculative_outputs[0]['text']})
                        generated_text = generated_text + speculative_outputs[0]['text']
                        out_of_loop = True
                        # print('in 189')
                        break
                    target_tokenizer_input = target_tokenizer(speculative_real_output_text, return_tensors="pt")['input_ids']
                    target_tokenizer_input_len = target_tokenizer_input.shape[1]
                    # print('speculative_real_output_text\n',speculative_real_output_text)
                    if use_target:
                        previous = checking_target_text
                        checking_target_text =  checking_target_text + speculative_real_output_text
                    else:
                        
                        checking_target_text =  target_text + target_tokenizer.decode(target_tokenizer(small_input,return_tensors="pt")['input_ids'][0,original_target_prompt_len:].tolist())
                        previous = checking_target_text.replace(speculative_real_output_text, '')
                        
                    # print('checking_target_text',checking_target_text)
                    checking_start = time.time()
                    checking_outputs = llm_big.generate([checking_target_text],
                        sampling_params = {"temperature": 0.1,"max_new_tokens": 1},
                        return_hidden_states = True,
            
                    )
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
                    # print('target_tokenizer_input_len---->',target_tokenizer_input_len)
                    # print("hidden_states-------->",hidden_states.shape)
                    computing_prob_start = time.time()
                    target_pooling_hidden_information = hidden_states[-target_tokenizer_input_len-1:-1, :]
                    exetract_big = time.time()-extract_time_big
                    time_detial.append({'exetract_big':exetract_big})
                    if target_pooling_hidden_information.shape[0] == 0:
                        # print("hidden_states",hidden_states.shape)
                        # print('speculative_real_output_text\n',speculative_real_output_text)
                        # print('checking_target_text',checking_target_text)
                        # print('target_tokenizer_input_len',target_tokenizer_input_len)
                        # print('use_target',use_target)
                        # print('arget_pooling_hidden_information.shape[0]',target_pooling_hidden_information.shape[0])
                        use_target = False
                        keep_doing = True
                        generated_text = small_input
                      
                    target_pooling_hidden_information = target_pooling_hidden_information.mean(dim=0, keepdim=True) # len *hidden
                    checking_list.append({
                        'speculative_real_output_text':speculative_real_output_text,
                        'pooling_hidden_information':pooling_hidden_information,
                        'target_pooling_hidden_information':target_pooling_hidden_information,
                        "checking_target_text":previous
                    }
                    
                    )
            
                if out_of_loop:
                    break
                if not keep_doing:                
                    speculative_real_output_texts = [item['speculative_real_output_text'] for item in checking_list]
                    pooling_list = [item['pooling_hidden_information'] for item in checking_list]
                    target_pooling_list = [item['target_pooling_hidden_information'] for item in checking_list]
                    checking_target_texts = [item['checking_target_text'] for item in checking_list]
                    assert len(pooling_list) == 3 and all(isinstance(t, torch.Tensor) for t in pooling_list), "pooling_list 长度或类型错误"
                    assert len(target_pooling_list) == 3 and all(isinstance(t, torch.Tensor) for t in target_pooling_list), "target_pooling_list 长度或类型错误"
                    pooling_batch = torch.stack(pooling_list).float().to('cuda:1')  # shape: [3, hidden_dim]
                    target_batch = torch.stack(target_pooling_list).float().to('cuda:1')  # 同上
                    with torch.no_grad():
                        prob_targets = model_target_probe(target_batch.float().to(f"cuda:{1}"))
                        prob_specs = model_spec_probe(pooling_batch.float().to(f"cuda:{1}"))
                    

                    prob_specs_floats = [p.item() for p in prob_specs]
                    prob_targets_floats = [p.item() for p in prob_targets]

                    # print("len of checkinglist",len())
                    
                    for i in range(3):
                        if speculative_accept(prob_targets_floats[i],prob_specs_floats[i]):
                            check_small_input += speculative_real_output_texts[i]
                            generated_text = check_small_input
                            detail.append({'spe_model':speculative_real_output_texts[i]})
                            correct_spe_number +=1
                            use_target = False
                        else:
                            # print('i------------>',i)
                            
                            generated_text =  target_text + target_tokenizer.decode(target_tokenizer(check_small_input,return_tensors="pt")['input_ids'][0,original_target_prompt_len:].tolist())
                            # if i == 1:
                                # print('generated_text',generated_text)
                            use_target = True
                            break
                    
                else:
                    keep_doing = False
               
            if use_target:
                # record the usage of the target model;
                begin = False
                try_correct_num = try_correct_num + 1
                target_outputs_start = time.time()
                target_outputs = llm_big.generate(
                    [generated_text],
                sampling_params = sampling_params,
                return_hidden_states = False,
               
                )
                target_outputs_time = time.time()-target_outputs_start
                time_detial.append({'target_outputs_time':target_outputs_time})
                target_outputs = target_outputs
                target_real_output = target_outputs[0]['text']
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
             
                    traget_ending_start = time.time()
                    speculative_outputs = llm_small.generate(

                         [small_input+target_real_output],
                        sampling_params = sampling_params_end,
                        return_hidden_states = False,

                    )
                    traget_ending_time = time.time()-traget_ending_start
                    time_detial.append({'traget_ending_time':traget_ending_time})
                    detail.append({'spe_model': speculative_outputs[0]['text']})

                    generated_text = small_input+ target_real_output+ speculative_outputs[0]['text']

                    break
                generated_text = generated_text + target_real_output


          

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
    print("real_answer\n", generated_text)
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
    parser.add_argument("--dataset", type=str,  help="dataset",default='math-500')#math-500
    parser.add_argument("--target_model", type=str,  help="target_model",default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--speculative_model", type=str,  help="speculative_model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--data_dir", type=str,  help="data_dir",default='../speculative/improve_DeepSeek-R1-Distill-32B_deepseek1.5seed_')
    parser.add_argument("--start_dataset", type=int, help="the beginning of the dataset",default=100)
    parser.add_argument("--end_dataset", type=int, help="the end of the dataset",default=500)
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
    llm_small = sgl.Engine(
    model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    enable_return_hidden_states=True,
    mem_fraction_static=0.7,
    tp_size=1   
    
)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    llm_big = sgl.Engine(
    model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    enable_return_hidden_states=True,
    mem_fraction_static=0.9,
   tp_size=1 
)

    model_target_probe = SemanticEntropyProbTarget(5120, 2048)
    model_target_probe.load_state_dict(torch.load(f'{args.target_probe}.pt'))
    model_target_probe = model_target_probe.to('cuda:1')
    model_target_probe.eval()


    model_spec_probe = SemanticEntropyProbSpec(1536, 1024)
    model_spec_probe.load_state_dict(torch.load(f'{args.speculative_probe}.pt'))
    model_spec_probe = model_spec_probe.to('cuda:1')
    model_spec_probe.eval()
    

    


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
    # # if args.seed == 6540:
    # #     wrong_list = [100, 101, 110, 118, 120, 126, 134, 136, 145, 147, 154, 163, 166, 168, 177, 198, 204, 205, 217, 219, 228, 232, 240, 242, 264, 284, 286, 295, 298, 303, 306, 308, 324, 333, 340, 352, 364, 379, 383, 392, 398, 400, 401, 403, 419, 422, 431, 444, 456, 468, 473, 478, 481, 486, 491]
    # # elif args.seed == 3210:
    # #     wrong_list = [100, 103, 104, 110, 126, 128, 136, 145, 154, 168, 198, 204, 205, 214, 217, 219, 232, 236, 239, 240, 242, 248, 264, 267, 282, 284, 286, 295, 296, 298, 301, 303, 306, 308, 320, 324, 340, 349, 351, 355, 381, 383, 392, 400, 403, 412, 419, 421, 422, 425, 444, 460, 466, 478, 481, 490, 497]
    # # else:
    # #     wrong_list = [101, 108, 109, 110, 119, 126, 138, 145, 150, 154, 158, 161, 166, 198, 204, 205, 213, 217, 219, 222, 240, 264, 284, 285, 286, 298, 303, 306, 308, 320, 324, 326, 340, 349, 369, 381, 383, 400, 412, 419, 421, 422, 423, 444, 456, 460, 461, 473, 490, 494, 497]
    # if args.seed == 3210:
    #     wrong_list =  [1, 4, 11, 13, 15, 16, 26, 27]
    # if args.seed == 9870:
    #     wrong_list = [ 13, 18,  20, 22]
    # if args.seed == 6540:
    #     wrong_list =   [1, 5, 6, 10, 13, 15, 17, 18, 19, 20, 21, 22, 25, 27]
    failed_total = []
    for idx, number in enumerate(tqdm(range(args.start_dataset, args.end_dataset))):
        dirname = f'spec_{args.dataset}_{number}'
        dir_path = os.path.join(f"{args.dataset}{args.data_dir}{args.seed}", dirname)
        problem = problems_and_answers[idx]['problem']
        answer = problems_and_answers[idx]['answer']
        failed = process_file_to_json(dir_path,llm_big,llm_small, target_tokenizer, speculative_tokenizer,problem,answer,args.max_new_tokens,model_target_probe,model_spec_probe,number)
        failed_total.extend(failed)