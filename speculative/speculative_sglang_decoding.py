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

def speculative_accept(qi, pi, threshold_min=0.7):

    ratio = qi / pi if pi > 0 else 0
    if ratio < threshold_min:
        return False
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
STOP_TOKENS = [
    ' \n\n', '.\n\n', ':\n\n', '\n\n',
    ')\n\n', '?\n\n', ']\n\n', ').\n\n'
]



def speculative_decoding(target_model, target_tokenizer, speculative_model,speculative_tokenizer,problem,max_new_tokens,model_target_probe,model_spec_probe):
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
            "stop_token_ids": [4710, 382, 1447, 271, 692, 1939, 2533, 3593, 13824, 14190],
            "no_stop_trim": True
        }
        # since we first feed the input to the target model, we need to add record the length of the input text in the target model and speculative model both;
        # this one is for the edge case, We need to check that the target model generates the answer within 200 words, then we need to stop directly.
        start_target_model_inputs = target_tokenizer(target_text, return_tensors="pt")
        generated_ids = start_target_model_inputs['input_ids']
        target_prompt_len = start_target_model_inputs["input_ids"].shape[1]
        original_len = target_prompt_len
        start_speculative_text_inputs = target_tokenizer(speculative_text, return_tensors="pt")['input_ids']
        original_target_text_len = start_target_model_inputs["input_ids"].shape[1]
        # there are kv caches in both the target model and speculative model.
        correct_tokens, try_correct_num, correct_spe_number = [], 0, 0
        detail = []
        begin = True
        use_target = True
        previous_original_target_text_len = original_target_text_len

        def checking_is_finish(generated_ids, max_new_tokens, use_target):

            if use_target:
                if generated_ids.shape[1] - original_target_text_len < max_new_tokens:
                    return True
                else:
                    return False
            else:
                if generated_ids.shape[1] - original_target_text_len < max_new_tokens:
                    return True
                else:
                    return False

        speculative_real_output = ''
        prob_target = 0
        prob_spec = 0
        target_real_output = ''

        while checking_is_finish(generated_ids,max_new_tokens,use_target):
            # we start at the target model.
            if begin:
                change_tokens = BEGIN_TOKEN_NUM
                use_target = True
            if not begin:
                # generating the text and check by probe
                # if it uses the target model, we need to covert the input text to the speculative model.
                if use_target:
                    speculative_text = speculative_text+target_real_output
                    real_target_output = target_real_output
                    detail.append({'target_model':real_target_output,'why_is_not_good':speculative_real_output,"score_target":round(prob_target, 2),"score_spec":round(prob_spec, 2)})

                    print('small model input\n',speculative_tokenizer.decode(speculative_text[0]))


               ## small model generation
                small_input = speculative_text

                speculative_outputs = speculative_model.generate(
                        [small_input], sampling_params=sampling_params, return_hidden_states=True)
                speculative_real_output = speculative_tokenizer.decode(speculative_outputs[0]['text'])
                #print("checking_generated_ids[0,small_input_ids.shape[1]:]\n",checking_generated_ids[0,small_input_ids.shape[1]:])
                special_token_id = 151646
                target_tokenizer_input = target_tokenizer(speculative_real_output, return_tensors="pt")['input_ids']
                if target_tokenizer_input[0, 0].item() == special_token_id:
                    target_tokenizer_input = target_tokenizer_input[:, 1:]

                target_tokenizer_input = target_tokenizer_input.to(
                    target_model.device)


                if use_target:
                    checking_target_ids =torch.cat([target_output_id,target_tokenizer_input], dim=-1)
                else:
                    previous_checking_target_ids = copy.deepcopy(checking_target_ids)
                    # print('previous_checking_target_ids',previous_checking_target_ids.shape)
                    checking_target_ids =  torch.cat([checking_target_ids.to(target_model.device),target_tokenizer_input.to(target_model.device)], dim=-1)

                checking_outputs = target_model.generate([small_input], sampling_params={"temperature": 0.1,"max_new_tokens": 1}, return_hidden_states=True)

                checking_output = checking_outputs[0]
                for i in range(len(checking_output["meta_info"]["hidden_states"])):
                    checking_output["meta_info"]["hidden_states"][i] = torch.tensor(
                        checking_output["meta_info"]["hidden_states"][i], dtype=torch.bfloat16
                    )
                Completion_tokens = checking_output['meta_info']['completion_tokens']
                hidden_states = torch.cat(
                    [
                        i.unsqueeze(0) if len(i.shape) == 1 else i
                        for i in checking_output["meta_info"]["hidden_states"]
                    ]
                )
                real_answer = checking_output['text']
                hidden_states = checking_output[-5:-1, :]  # len *hidden





                with torch.no_grad():
                    prob_target = model_target_probe(target_pooling_hidden_information.float().to(f"cuda:{1}"))
                with torch.no_grad():
                    prob_spec = model_spec_probe(pooling_hidden_information.float().to(f"cuda:{1}"))

                prob_target = prob_target.item()
                prob_spec = prob_spec.item()
                #print(f"prob_target.item() {prob_target} , prob_spec.item() {prob_spec}")
                if speculative_accept(prob_target, prob_spec):
                    detail.append({'spe_model':speculative_real_output})
                    correct_spe_number +=1
                    use_target = False
                    generated_ids = checking_generated_ids
                    target_output_id = checking_target_ids
                else:

                    # valid_tgt_kv  not change
                    if use_target:
                        generated_ids = target_output_id
                    else:
                        generated_ids = previous_checking_target_ids


                    use_target = True
                    start_speculative_text_inputs = small_input_ids
                    #spec_kv = spec_kv # not change


            # Let the target model finish the generation.
            # At the beginning of the generation, Let the target model generate the first part of completion.
            if use_target:
                # record the usage of the target model;
                begin = False
                try_correct_num = try_correct_num + 1

                previous_original_target_text_len = generated_ids.shape[1]

                target_model_input = target_tokenizer.decode(generated_ids[0],skip_special_tokens=True)
                target_outputs = target_model.generate(
                    [target_model_input], sampling_params=sampling_params)
                target_real_output = target_outputs[0]['text']

                print('speculative_outputs\n',target_outputs[0])


                print('\n\n\n\n')
                print('speculative_real_output\n',target_real_output)

                # if inferencing the model stops at the first time
                if target_tokenizer.eos_token_id in target_tokenizer.encode(target_real_output):
                    generated_text = target_tokenizer.decode(generated_ids[0, :], skip_special_tokens=True)
                    #print('target_tokenizer.eos_token_id in the generated_text',target_tokenizer.eos_token_id)
                    break

            if speculative_tokenizer.eos_token_id in generated_ids[0, target_prompt_len:]:
                break
            generated_text = speculative_tokenizer.decode(generated_ids[0, :], skip_special_tokens=True)
            length_of_output = generated_ids.shape[1]


        return generated_text, try_correct_num,correct_spe_number,detail,length_of_output-original_len





def process_file_to_json(dir_path, target_model, target_tokenizer,speculative_model, speculative_tokenizer,problem, answer,max_new_tokens,model_target_probe,model_spec_probe):
    all_generations = []

    start_time = time.time()
    result = speculative_decoding(target_model, target_tokenizer, speculative_model, speculative_tokenizer, problem,max_new_tokens,model_target_probe,model_spec_probe)
    end_time = time.time()
    generated_text, try_correct_num,correct_spe_number,detail,length_of_output = result
    print('real_answer\n',generated_text)

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
    parser.add_argument("--dataset", type=str,  help="dataset",default='aime')#math-500
    parser.add_argument("--target_model", type=str,  help="target_model",default="Qwen/QwQ-32B")
    parser.add_argument("--speculative_model", type=str,  help="speculative_model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--data_dir", type=str,  help="data_dir",default='/data/semantic/speculative/new_spec_result_aime_full_size_QwQ-32B_r132_deepseek1.5seed_')
    parser.add_argument("--start_dataset", type=int, help="the beginning of the dataset",default=0)
    parser.add_argument("--end_dataset", type=int, help="the end of the dataset",default=30)
    parser.add_argument("--target_probe", type=str, help="target_probe",default="/data/semantic/training/valid_deepseek32b_aime_output_last_hidden_list_best_probe_mse")#aime_output_last_hidden_list_best_probe_mse
    parser.add_argument("--speculative_probe", type=str, help="speculative_probe",default="/home/shaowei/new_probe/new_deepseekr11.5b_math-500_output_last_hidden_list_best_probe_mse")
    parser.add_argument("--target_temperature", type=float, help="target_temperature",default=0.1)
    parser.add_argument("--speculative_temperature", type=float, help="speculative_temperature",default=0.6)
    parser.add_argument("--max_new_tokens", type=int, help="max_new_tokens",default=14000)
    parser.add_argument("--top_p", type=float, help="top_p",default=0.9)
    parser.add_argument("--top_k", type=int, help="top_k",default=50)
    parser.add_argument("--seed", type=int, help="seed", default=456)
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

    model_target_probe = SemanticEntropyProbTarget(5120, 256)
    model_target_probe.load_state_dict(torch.load(f'{args.target_probe}.pt'))
    model_target_probe = model_target_probe.to('cuda:1')
    model_target_probe.eval()


    model_spec_probe = SemanticEntropyProbSpec(1536, 256)
    model_spec_probe.load_state_dict(torch.load(f'{args.speculative_probe}.pt'))
    model_spec_probe = model_spec_probe.to('cuda:1')
    model_spec_probe.eval()

    target_model = sgl.Engine(
        model_path=args.target_model,
        tp_size=4,
        enable_return_hidden_states=True,
        mem_fraction_static=0.7
    )
    target_tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.target_model,
        trust_remote_code=True
    )

    speculative_model = sgl.Engine(
        model_path=args.speculative_model,
        tp_size=4,
        enable_return_hidden_states=True,
        mem_fraction_static=0.2
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
    #     print("doing wrong number:", number)
    #     dirname = f'spec_{args.dataset}_{number}'
    #     dir_path = os.path.join(f"{args.data_dir}{args.seed}", dirname)
    #     number = number
    #     problem = problems_and_answers[number]['problem']
    #     print(f"{number}: {problem}")
    #     answer = problems_and_answers[number]['answer']
    #     process_file_to_json(dir_path, target_model, target_tokenizer,speculative_model, speculative_tokenizer, problem,answer,args.target_temperature,args.speculative_temperature,args.max_new_tokens,model_target_probe,model_spec_probe)

    for idx, number in enumerate(tqdm(range(args.start_dataset, args.end_dataset))):
        dirname = f'spec_{args.dataset}_{number}'
        dir_path = os.path.join(f"{args.data_dir}{args.seed}", dirname)
        problem = problems_and_answers[idx]['problem']
        answer = problems_and_answers[idx]['answer']
        process_file_to_json(dir_path, target_model, target_tokenizer,speculative_model, speculative_tokenizer, problem,answer,args.max_new_tokens,model_target_probe,model_spec_probe)