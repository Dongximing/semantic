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
    if ratio < threshold_min:
        return False
    threshold = min(1.0, ratio)
    r = random.uniform(0, 1)
    return r < threshold


# class SemanticEntropyProbTarget(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)
#     def forward(self, x):
#         h = F.relu(self.fc1(x))
#         out = torch.sigmoid(self.fc2(h))
#         return out.squeeze(-1)
class SemanticEntropyProbTarget(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        out = torch.sigmoid(self.fc3(h))
        return out.squeeze(-1)
# class SemanticEntropyProbSpec(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)
#     def forward(self, x):
#         h = F.relu(self.fc1(x))
#         out = torch.sigmoid(self.fc2(h))
#         return out.squeeze(-1)


class SemanticEntropyProbSpec(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        out = torch.sigmoid(self.fc3(h))
        return out.squeeze(-1)

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
        start_time = time.time()
        while checking_is_finish(generated_text,max_new_tokens,use_target):
            # we start at the target model.
            if begin:
                use_target = True
            if not begin:
                if use_target:
                    # print('target_tokenizer.decode(target_tokenizer(generated_text,return_tensors="pt")[original_target_prompt_len:]:\n',
                    #       target_tokenizer(generated_text,return_tensors="pt"))
                    detail.append({'target_model': target_real_output, 'why_is_not_gd': speculative_real_output_text,
                                   "score_target": round(prob_target, 2), "score_spec": round(prob_spec, 2)})
                    small_input = speculative_text + target_tokenizer.decode(
                        target_tokenizer(generated_text, return_tensors="pt")['input_ids'][0,
                        original_target_prompt_len:].tolist()
                    )
                else:
                    small_input  = generated_text

                json_data = {
                    "text": [small_input],
                    "sampling_params": sampling_params,
                    "return_hidden_states": True,
                }
                speculative_outputs = requests.post(
                                    f"http://130.179.30.7:{8801}/generate",
                                    json=json_data,
                    timeout=120
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
                pooling_hidden_information = pooling_hidden_information.mean(dim=0, keepdim=True)

                if len(speculative_real_output_text) ==0:
                    break


                target_tokenizer_input = target_tokenizer(speculative_real_output_text, return_tensors="pt")['input_ids']
                target_tokenizer_input_len = target_tokenizer_input.shape[1]

                if use_target:
                    checking_target_text =  generated_text + speculative_real_output_text
                else:
                    checking_target_text =  target_text  + target_tokenizer.decode(target_tokenizer(small_input+speculative_real_output_text,return_tensors="pt")['input_ids'][0,original_target_prompt_len:].tolist())


                json_data_check = {
                    "text": [checking_target_text],
                    "sampling_params": {"temperature": 0.1,"max_new_tokens": 0},
                    "return_hidden_states": True,
                }

                checking_outputs = requests.post(
                    f"http://130.179.30.15:{8080}/generate",
                    json=json_data_check,
                    timeout=120
                )
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
                            f"http://130.179.30.7:{8801}/generate",
                            json=json_data,
                            timeout=120
                        )
                        detail.append({'spe_model': speculative_outputs.json()[0]['text']})
                        generated_text = generated_text + speculative_outputs.json()[0]['text']

                        break
                else:
                    generated_text = target_text + speculative_tokenizer.decode(
    speculative_tokenizer(small_input, return_tensors="pt")['input_ids'][0,original_speculative_text_len :].tolist()
)
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
                # print(json_data)
                target_outputs = requests.post(
                    f"http://130.179.30.15:{8080}/generate",
                    json=json_data,
                    timeout=120
                )
                # print(target_outputs.status_code)
                # print(target_outputs.text)
                target_outputs = target_outputs.json()


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

                    json_data = {
                        "text": [small_input+target_real_output],
                        "sampling_params": sampling_params_end,
                        "return_hidden_states": False,
                    }
                    speculative_outputs = requests.post(
                        f"http://130.179.30.7:{8801}/generate",
                        json=json_data,

                    )
                    detail.append({'spe_model': speculative_outputs.json()[0]['text']})

                    generated_text = generated_text+speculative_outputs.json()[0]['text']

                    break
                generated_text = generated_text + target_real_output


                # if inferencing the model stops at the first time

                if target_tokenizer.eos_token_id in target_tokenizer.encode(target_real_output):
                    print('target_tokenizer.eos_token_id 281',target_tokenizer.eos_token_id)
                    break

            # print(speculative_tokenizer.encode(generated_text[original_speculative_text_len:]))

        end_time = time.time()


        length_of_output = speculative_tokenizer.encode(generated_text[original_speculative_text_len:])

        print('end_time-start_time',end_time-start_time)
        return generated_text, try_correct_num,correct_spe_number,detail,len(length_of_output),end_time-start_time





def process_file_to_json(
    dir_path,
    target_tokenizer,
    speculative_tokenizer,
    problem,
    answer,
    max_new_tokens,
    model_target_probe,
    model_spec_probe,
    idx
):
    all_generations = []
    failed_list = []

    try:
        # start_time = time.time()

        result = speculative_decoding(
            target_tokenizer,
            speculative_tokenizer,
            problem,
            max_new_tokens,
            model_target_probe,
            model_spec_probe
        )

        # end_time = time.time()

        generated_text, try_correct_num, correct_spe_number, detail, length_of_output,times = result
        print("real_answer\n", generated_text)

        all_generations.append({
            "input_text": problem,
            "real_answer": generated_text,
            "try_correct_num": try_correct_num,
            "standard_answer": answer,
            "execution_time": f"{times:.2f}s",
            "correct_spe_number": correct_spe_number,
            "detail": detail,
            "length_of_output": length_of_output,
            "index": idx
        })

        os.makedirs(dir_path, exist_ok=True)
        out_path = os.path.join(dir_path, "spec_generation.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_generations, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"[Index {idx}] Failed with error: {e}")
        print("Sleeping 10 seconds before moving on...")
        time.sleep(1)
        failed_list.append(idx)
    return failed_list










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,  help="dataset",default='math-500')#math-500
    parser.add_argument("--target_model", type=str,  help="target_model",default="Qwen/QwQ-32B")
    parser.add_argument("--speculative_model", type=str,  help="speculative_model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--data_dir", type=str,  help="data_dir",default='/data/semantic/speculative/sglang_new_spec_result_math-500_full_size_QwQ-32B_r132_deepseek1.5seed_')
    parser.add_argument("--start_dataset", type=int, help="the beginning of the dataset",default=0)
    parser.add_argument("--end_dataset", type=int, help="the end of the dataset",default=30)
    parser.add_argument("--target_probe", type=str, help="target_probe",default="/data/semantic/training/valid_new_2048_full_size_slg_qwq-32b_math-500_output_last_hidden_list_best_probe_mse")#aime_output_last_hidden_list_best_probe_mse
    parser.add_argument("--speculative_probe", type=str, help="speculative_probe",default="/home/shaowei/new_probe/s1_valid_new_deepseekr11.5b_s1_output_last_hidden_list_best_probe_mse")
    parser.add_argument("--target_temperature", type=float, help="target_temperature",default=0.1)
    parser.add_argument("--speculative_temperature", type=float, help="speculative_temperature",default=0.6)
    parser.add_argument("--max_new_tokens", type=int, help="max_new_tokens",default=14000)
    parser.add_argument("--top_p", type=float, help="top_p",default=0.9)
    parser.add_argument("--top_k", type=int, help="top_k",default=50)
    parser.add_argument("--seed", type=int, help="seed", default=301)
    args = parser.parse_args()
    seed_everything(args.seed)



    model_target_probe = SemanticEntropyProbTarget(5120, 2048)
    model_target_probe.load_state_dict(torch.load(f'{args.target_probe}.pt'))
    model_target_probe = model_target_probe.to('cuda:1')
    model_target_probe.eval()


    model_spec_probe = SemanticEntropyProbSpec(1536, 512)
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
    if args.seed == 3001:
        wrong_list = [267, 271, 274, 276, 278, 279, 281, 282, 284, 285, 286, 287, 288, 289, 291, 292, 294, 295, 296, 298, 299, 301, 302, 303, 306, 308, 309, 311, 315, 316, 317, 320, 323, 324, 326, 327, 328, 331, 332, 333, 335, 337, 340, 342, 348, 349, 351, 352, 353, 355, 356, 360, 361, 365, 366, 369, 371, 372, 375, 377, 379, 380, 381, 382, 383,387, 390, 391, 392, 394, 395, 396, 397, 399, 400, 401, 403, 408, 409, 412, 415, 416, 420, 421, 422, 423, 425, 428, 429, 431, 432, 435,439, 443, 444, 445, 446, 448, 454, 456, 457, 459, 460, 461, 462, 465, 466, 467, 468,470, 473, 475, 477, 478, 480, 481, 482, 483, 484, 485, 486, 489, 490, 491, 493, 494, 495, 497, 498, 499]
    elif args.seed == 3002:
        wrong_list = [138, 139, 142, 145, 147, 150, 151, 152, 153, 154, 155, 156, 158, 163, 164, 166, 168, 169, 170, 173, 175, 176, 177, 178, 179, 180, 183, 184, 188, 189, 194, 197, 198,202, 204, 205, 209, 210, 211, 213, 217, 219, 221, 222, 223, 224, 226, 228, 232, 235,236, 237, 239, 240, 241, 242, 244, 245, 246, 248, 250, 252, 256, 264, 265, 267, 268, 271, 273, 274, 275, 276, 279, 281, 284,285, 286, 287, 288, 291, 292, 294, 295, 296, 298, 299, 300, 301, 302, 303, 306, 308, 309, 311, 316, 317, 320, 323, 324, 325, 326, 327, 328, 332, 333, 335, 336, 337, 338,340, 341, 345, 348, 349, 350, 351, 352, 355, 356, 357, 358, 360, 361, 365, 366, 368, 369, 371, 372, 377, 378, 379, 380, 381, 382, 383,385, 390, 391, 392, 394, 395, 396, 397, 399, 400, 401, 403, 406, 408, 409, 412, 415, 416, 419, 420, 421, 422, 423, 425, 428, 429, 432, 433, 439, 444, 445, 449, 454, 460, 462, 466, 467, 469, 470, 473, 474, 475, 477, 478, 480, 481, 482, 483, 484, 485, 486, 489, 490, 491, 493, 494, 496, 497, 498, 499]
    elif args.seed == 998:
        wrong_list = [101, 103, 104, 108, 109, 110, 112, 114, 115, 117, 118, 119, 120, 123, 124, 126, 128, 129, 135, 136, 137, 138, 139, 142, 145, 147, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 163, 164, 165, 166, 168, 170, 173, 175, 176, 177, 181, 183, 184, 188, 189, 193, 194, 195, 196, 200, 202, 203, 204,205, 208, 209, 210, 211, 216, 217, 219, 221,222, 223, 224, 230, 232, 234, 235, 236, 239, 240, 242, 244, 245, 247, 251, 256, 260, 264, 267, 271, 274, 276, 279, 281, 282, 284, 285, 286, 287, 288, 289, 291, 292, 294, 295, 296, 298, 300, 301, 302, 303, 306, 308, 309, 310, 311, 314, 315, 317, 320, 323, 324, 325, 326, 327, 328, 330, 331, 332, 333, 335, 337, 338, 339, 340, 342, 349, 350, 351, 352, 353, 355, 356, 359, 361, 365, 369, 371, 372, 375, 377, 378, 379, 380, 381, 382, 383,385, 390, 392, 394, 395, 396, 397, 398, 400, 401, 405, 408, 409, 412, 413, 414, 415, 416, 419, 420, 421, 422, 423, 425, 428, 429, 432, 434, 436, 439, 444, 445, 446, 448, 449, 450, 451, 454, 459, 460, 461, 466, 467, 468, 469, 470, 473, 475, 477, 478, 481, 482, 483, 484, 485, 486, 487, 489, 490, 491, 493, 494, 495, 497, 498, 499]
    for idx, number in enumerate(tqdm(wrong_list, total=len(wrong_list))):

        print("doing wrong number:", number)
        dirname = f'spec_{args.dataset}_{number}'
        dir_path = os.path.join(f"{args.data_dir}{args.seed}", dirname)
        number = number-100
        problem = problems_and_answers[number]['problem']
        #print(f"{number}: {problem}")
        answer = problems_and_answers[number]['answer']
        process_file_to_json(dir_path, target_tokenizer, speculative_tokenizer, problem,answer,args.max_new_tokens,model_target_probe,model_spec_probe,number)

    # common_errors_minus_100 = [
    #     10, 28, 54, 104, 140, 164, 208,
    #     224, 322, 344
    # ]
    #
    #
    # [198, 383, 435, 468]


    # failed_total = []
    # for idx, number in enumerate(tqdm(range(args.start_dataset, args.end_dataset))):
    #     if idx in common_errors_minus_100:
    #         continue
    #     dirname = f'spec_{args.dataset}_{number}'
    #     dir_path = os.path.join(f"{args.data_dir}{args.seed}", dirname)
    #     problem = problems_and_answers[idx]['problem']
    #     answer = problems_and_answers[idx]['answer']
    #
    #     failed = process_file_to_json(dir_path,  target_tokenizer, speculative_tokenizer, problem,answer,args.max_new_tokens,model_target_probe,model_spec_probe,number)
    #     failed_total.extend(failed)