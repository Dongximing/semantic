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

def speculative_accept(qi, pi, threshold_min=0.8):

    ratio = qi / pi if pi > 0 else 0
    if ratio < threshold_min:
        return False   # ratio 太低，直接 reject
    threshold = min(1.0, ratio)
    r = random.uniform(0, 1)
    return r < threshold


class SemanticEntropyProbTarget(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out.squeeze(-1)

class SemanticEntropyProbSpec(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out.squeeze(-1)


# class SemanticEntropyProbSpec(nn.Module):
#     def __init__(self, input_dim, hidden_dim=512, dropout=0.3):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 256)
#         self.dropout = nn.Dropout(dropout)
#         self.fc3 = nn.Linear(256, 1)
#
#     def forward(self, x):
#         h = F.relu(self.fc1(x))
#         h = F.relu(self.fc2(h))
#         h = self.dropout(h)
#         out = torch.sigmoid(self.fc3(h))
#         return out.squeeze(-1)

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



def speculative_decoding(target_tokenizer,speculative_tokenizer,problem,max_new_tokens,model_target_probe,model_spec_probe):
        # add prompt before inferencing the model
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
            if use_target:
                if len(target_tokenizer.encode(generated_ids))- original_target_prompt_len < max_new_tokens:
                    return True
                else:
                    return False
            else:
                if len(target_tokenizer.encode(generated_ids))- original_target_prompt_len < max_new_tokens:
                    return True
                else:
                    return False

        speculative_real_output_text = ''
        prob_target = 0
        prob_spec = 0
        target_real_output = ''
        generated_text = target_text

        # print('speculative_tokenizer.eos_token_id ', speculative_tokenizer.eos_token_id)
        # print('target_tokenizer.eos_token_id ', target_tokenizer.eos_token_id)

        while checking_is_finish(generated_text,max_new_tokens,use_target):
            # we start at the target model.
            if begin:
                use_target = True
            if not begin:
                if use_target:
                    # print('target_tokenizer.decode(target_tokenizer(generated_text,return_tensors="pt")[original_target_prompt_len:]:\n',
                    #       target_tokenizer(generated_text,return_tensors="pt"))
                    detail.append({'target_model': target_real_output, 'why_is_not_good': speculative_real_output_text,
                                   "score_target": round(prob_target, 2), "score_spec": round(prob_spec, 2)})
                    small_input = speculative_text + target_tokenizer.decode(
                        target_tokenizer(generated_text, return_tensors="pt")['input_ids'][0,
                        original_target_prompt_len:].tolist()
                    )
                else:
                    small_input  = generated_text
                # print('small_input:\n',small_input)

                # if speculative_tokenizer.eos_token_id in speculative_tokenizer.encode(small_input):
                #     print('target_tokenizer.eos_token_id 285', speculative_tokenizer.eos_token_id)
                #     break

                json_data = {
                    "text": [small_input],
                    "sampling_params": sampling_params,
                    "return_hidden_states": True,
                }
                speculative_outputs = requests.post(
                                    f"http://130.179.30.7:{30000}/generate",
                                    json=json_data,
                                     )
                speculative_output = speculative_outputs.json()
                speculative_real_output_text = speculative_output[0]['text']
                speculative_output = speculative_output[0]
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
                # print('pooling_hidden_information',pooling_hidden_information.size())
                pooling_hidden_information = pooling_hidden_information.mean(dim=0, keepdim=True)



                # print('speculative_real_output_text:\n',speculative_real_output_text)
                if len(speculative_real_output_text) ==0:
                    break


                target_tokenizer_input = target_tokenizer(speculative_real_output_text, return_tensors="pt")['input_ids']
                target_tokenizer_input_len = target_tokenizer_input.shape[1]

                if use_target:
                    checking_target_text =  generated_text + speculative_real_output_text
                else:
                    checking_target_text =  target_text  + target_tokenizer.decode(target_tokenizer(small_input+speculative_real_output_text,return_tensors="pt")['input_ids'][0,original_target_prompt_len:].tolist())
                # print('checking_target_text:\n',checking_target_text)

                json_data_check = {
                    "text": [checking_target_text],
                    "sampling_params": {"temperature": 0.1,"max_new_tokens": 1},
                    "return_hidden_states": True,
                }
                checking_outputs = requests.post(
                    f"https://tqks84p4oltpp8-8800.proxy.runpod.net/generate",
                    json=json_data_check,
                )
                print('checking_outputs', checking_outputs)
                print(checking_outputs.status_code)
                print(checking_outputs.text)
                checking_outputs = checking_outputs.json()



                checking_output = checking_outputs[0]
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

                target_pooling_hidden_information = hidden_states[-target_tokenizer_input_len-1:-1, :]
                # print('target_pooling_hidden_information shape', target_pooling_hidden_information.shape)
                if target_pooling_hidden_information.shape[0] == 0:
                    break
                target_pooling_hidden_information = target_pooling_hidden_information.mean(dim=0, keepdim=True) # len *hidden
                #print('target_tokenizer_input_len',target_tokenizer_input_len)


                with torch.no_grad():
                    prob_target = model_target_probe(target_pooling_hidden_information.float().to(f"cuda:{1}"))
                with torch.no_grad():
                    prob_spec = model_spec_probe(pooling_hidden_information.float().to(f"cuda:{1}"))

                prob_target = prob_target.item()
                prob_spec = prob_spec.item()
                # print(f"prob_target.item() {prob_target} , prob_spec.item() {prob_spec}")
                if speculative_accept(prob_target, prob_spec):
                    detail.append({'spe_model':speculative_real_output_text})
                    correct_spe_number +=1
                    use_target = False
                    generated_text =  small_input + speculative_output['text']
                    # print('acceptacceptacceptacceptacceptacceptaccept!!!!!!!!!!!!!!!')
                else:
                    generated_text = target_text + speculative_tokenizer.decode(
    speculative_tokenizer(small_input, return_tensors="pt")['input_ids'][0,original_speculative_text_len :].tolist()
)
                    # print('rejectrejectrejectrejectrejectrejectrejectreject!!!!!!!!!!!!!!!')
                    use_target = True



            # Let the target model finish the generation.
            # At the beginning of the generation, Let the target model generate the first part of completion.
            if use_target:
                # record the usage of the target model;
                begin = False
                try_correct_num = try_correct_num + 1

                # print("big model input:\n",generated_text)
                json_data = {
                    "text": [generated_text],
                    "sampling_params": sampling_params,
                    "return_hidden_states": False,
                }
                target_outputs = requests.post(
                    f"https://tqks84p4oltpp8-8800.proxy.runpod.net/generate",
                    json=json_data,
                )
                print(target_outputs.status_code)
                print(target_outputs.text)
                target_outputs = target_outputs.json()


                target_real_output = target_outputs[0]['text']
                # print('big target_output:\n',target_real_output)
                generated_text = generated_text + target_real_output


                # if inferencing the model stops at the first time

                if target_tokenizer.eos_token_id in target_tokenizer.encode(target_real_output):
                    print('target_tokenizer.eos_token_id 281',target_tokenizer.eos_token_id)
                    break

            # print(speculative_tokenizer.encode(generated_text[original_speculative_text_len:]))


        length_of_output = speculative_tokenizer.encode(generated_text[original_speculative_text_len:])


        return generated_text, try_correct_num,correct_spe_number,detail,len(length_of_output)





def process_file_to_json(dir_path , target_tokenizer, speculative_tokenizer,problem, answer,max_new_tokens,model_target_probe,model_spec_probe):
    all_generations = []

    start_time = time.time()
    result = speculative_decoding( target_tokenizer, speculative_tokenizer, problem,max_new_tokens,model_target_probe,model_spec_probe)
    end_time = time.time()
    generated_text, try_correct_num,correct_spe_number,detail,length_of_output = result
    #print('real_answer\n',generated_text)

    all_generations.append({
        "input_text": problem,
        "real_answer": generated_text,
        "try_correct_num": try_correct_num,
        "standard_answer": answer,
        "execution_time": f"{end_time - start_time:.2f}s",
        "correct_spe_number":correct_spe_number,
        "detail":detail,
        "length_of_output":length_of_output

    })
    os.makedirs(dir_path, exist_ok=True)
    out_path = os.path.join(dir_path, "spec_generation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_generations, f, ensure_ascii=False, indent=2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,  help="dataset",default='math-500')#math-500
    parser.add_argument("--target_model", type=str,  help="target_model",default="Qwen/QwQ-32B")
    parser.add_argument("--speculative_model", type=str,  help="speculative_model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--data_dir", type=str,  help="data_dir",default='/data/semantic/speculative/sglang_new_spec_result_math-500_full_size_QwQ-32B_r132_deepseek1.5seed_')
    parser.add_argument("--start_dataset", type=int, help="the beginning of the dataset",default=0)
    parser.add_argument("--end_dataset", type=int, help="the end of the dataset",default=30)
    parser.add_argument("--target_probe", type=str, help="target_probe",default="/data/semantic/training/valid_new_full_size_slg_qwq-32b_math-500_output_last_hidden_list_best_probe_mse")#aime_output_last_hidden_list_best_probe_mse
    parser.add_argument("--speculative_probe", type=str, help="speculative_probe",default="/home/shaowei/new_probe/new_deepseekr11.5b_math-500_output_last_hidden_list_best_probe_mse")
    parser.add_argument("--target_temperature", type=float, help="target_temperature",default=0.1)
    parser.add_argument("--speculative_temperature", type=float, help="speculative_temperature",default=0.6)
    parser.add_argument("--max_new_tokens", type=int, help="max_new_tokens",default=14000)
    parser.add_argument("--top_p", type=float, help="top_p",default=0.9)
    parser.add_argument("--top_k", type=int, help="top_k",default=50)
    parser.add_argument("--seed", type=int, help="seed", default=729)
    args = parser.parse_args()
    seed_everything(args.seed)
    #wrong_list = [ 240, 248, 251,  282, 286, 295, 296, 299, 301, 306, 308, 309, 317, 327, 338, 341, 349,  352, 355, 369, 381, 392, 400, 403, 416, 422, 425, 432, 444, 460, 464, 469, 470, 473, 478, 481, 483, 485, 490, 493]
    #123
    #wrong_list = [100, 101, 109, 110, 119, 120,  137, 138, 145, 154, 164, 165, 166, 168, 176,  189, 197,  204,  219, 221, 228,  239, 240, 242, 246, 248, 264, 279,  286, 288, 302, 306, 308, 309, 317, 324, 332, 340, 349,  352, 359, 365, 369, 372, 380, 381, 382,  385, 392, 400, 403, 419, 421, 422, 425,444, 448, 456, 460, 466, 475, 478, 481,486, 490, 494, 497]
    # 42
    #wrong_list =  [100, 101, 103, 104, 105, 110, 119, 120, 128, 138, 145, 154, 164, 168, 176, 196, 204, 209, 217, 219, 238, 239, 240, 242, 248, 264, 282, 285, 286, 292, 295, 296, 301, 303, 308, 309, 324, 340,  352, 358, 369, 381, 392, 400, 401, 405, 409, 421, 422, 425, 432, 439, 444, 460, 466, 478, 481, 485, 489, 491, 494]
    # if args.seed  == 981:
    #     wrong_list =  [1, 2, 3, 5, 6, 10, 13, 14, 16, 17, 18, 20, 21, 22, 23, 25, 27, 28]
    # elif args.seed == 20981:
    #     wrong_list = [1, 2, 3, 4, 10, 13, 15, 16, 17, 20, 21, 22, 25, 27, 28]
    # elif args.seed == 30981:
    #     wrong_list = [1, 2, 3, 4, 5, 10, 12, 13, 14, 15, 17, 18, 20, 21, 22, 25, 27, 28]
    # if args.seed  == 7291:
    #     wrong_list =  [18]
    # elif args.seed == 12501:
    #     wrong_list = [1, 5, 6, 10, 16, 22, 25]
    # elif args.seed == 20241:
    #     wrong_list = [27]


    model_target_probe = SemanticEntropyProbTarget(5120, 512)
    model_target_probe.load_state_dict(torch.load(f'{args.target_probe}.pt'))
    model_target_probe = model_target_probe.to('cuda:1')
    model_target_probe.eval()


    model_spec_probe = SemanticEntropyProbSpec(1536, 256)
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
    else:
        raise ValueError(f"Unknown task: {args.dataset}")

    ds = ds.select(range(args.start_dataset, args.end_dataset))
    problems_and_answers = [{"problem": item["problem"], "answer": item["answer"]} for item in ds]
    # for idx, number in enumerate(tqdm(wrong_list, total=len(wrong_list))):
    #
    #     #print("doing wrong number:", number)
    #     dirname = f'spec_{args.dataset}_{number}'
    #     dir_path = os.path.join(f"{args.data_dir}{args.seed}", dirname)
    #     number = number
    #     problem = problems_and_answers[number]['problem']
    #     #print(f"{number}: {problem}")
    #     answer = problems_and_answers[number]['answer']
    #     process_file_to_json(dir_path, target_tokenizer, speculative_tokenizer, problem,answer,args.max_new_tokens,model_target_probe,model_spec_probe)

    for idx, number in enumerate(tqdm(range(args.start_dataset, args.end_dataset))):
        dirname = f'spec_{args.dataset}_{number}'
        dir_path = os.path.join(f"{args.data_dir}{args.seed}", dirname)
        problem = problems_and_answers[idx]['problem']
        answer = problems_and_answers[idx]['answer']
        process_file_to_json(dir_path,  target_tokenizer, speculative_tokenizer, problem,answer,args.max_new_tokens,model_target_probe,model_spec_probe)