import transformers
import torch
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

BEGIN_TOKEN_NUM = 200
SPECULATIVE_OUTPUT_LENGTH = 200
MATH_PROMPT = "\nPlease reason step by step, and put your final answer within \\boxed{}."
TARGET_model= 1
SPEC_model = 0
TARGET_probe = 2
SPEC_probe = 3

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
        temperature=1.0, top_k=50, top_p=0.95,checking = False
):


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
    print('-------------------------------')
    print('model name hidden_size',model.config.hidden_size)
    print('-------------------------------')


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
            output_scores=True,
            output_hidden_states=True,
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
    print("generated_ids:\n",tokenizer.decode(generated_ids[0, :], skip_special_tokens=True))
    past_key_values = output.past_key_values

    hidden = output.hidden_states
    if checking:
        output_last_hidden_list = torch.stack([layer[-1][:, -1, :] for layer in hidden[:-1]]).cpu()
    else:
        output_last_hidden_list = torch.stack([layer[-1][:, -1, :] for layer in hidden]).cpu()
    output_last_hidden_list = output_last_hidden_list.squeeze(1)  # [len ,D]
    output_last_hidden_list = output_last_hidden_list.mean(dim=0, keepdim=True)  # [1,D]
    return generated_ids, past_key_values,output_last_hidden_list


def speculative_decoding(target_model, target_tokenizer, speculative_model,speculative_tokenizer,problem, target_temperature,speculative_temperature,max_new_tokens,model_target_probe,model_spec_probe):
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
        # since we first feed the input to the target model, we need to add record the length of the input text in the target model and speculative model both;
        # this one is for the edge case, We need to check that the target model generates the answer within 200 words, then we need to stop directly.
        start_target_model_inputs = target_tokenizer(target_text, return_tensors="pt").to(target_model.device)
        generated_ids = start_target_model_inputs['input_ids']
        target_prompt_len = start_target_model_inputs["input_ids"].shape[1]
        start_speculative_text_inputs = target_tokenizer(speculative_text, return_tensors="pt")['input_ids'].to(speculative_model.device)
        original_target_text_len = start_target_model_inputs["input_ids"].shape[1]
        # there are kv caches in both the target model and speculative model.
        spec_kv, tgt_kv = None, None
        correct_tokens, try_correct_num = [], 0
        # change flag is used to check whether the target models need to guide the speculative model.
        token_num, change_tokens, change_flag = 0, 0, False
        # 'begin' implicate the first time we start the speculative model.
        begin = True
        use_target = True

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



        while checking_is_finish(generated_ids,max_new_tokens,use_target):
            # we start at the target model.
            if begin:
                change_tokens = BEGIN_TOKEN_NUM
                valid_tgt_kv = None
                spe_decoded_text = '',
                use_target = True
            if not begin:
                # generating the text and check by probe
                # if it uses the target model, we need to covert the input text to the speculative model.
                if use_target:
                    target_output_id = generated_ids
                    real_target_output = target_tokenizer.decode(generated_ids[:-(generated_ids.shape[1] - original_target_text_len)],skip_special_tokens=True)
                    speculative_tokenizer_input = speculative_tokenizer(real_target_output, return_tensors="pt")['input_ids'].to(speculative_model.device)
                    generated_ids = torch.cat([start_speculative_text_inputs,speculative_tokenizer_input], dim=-1)
                    print('speculative_tokenizer_input',target_tokenizer.decode(generated_ids[0,:],skip_special_tokens=True))

                small_input_ids = generated_ids

               ## small model generation
                checking_generated_ids, checking_spec_kv,pooling_hidden_information = generate_with_partial_kv(
                    speculative_model, speculative_tokenizer, small_input_ids , spec_kv,
                    max_new_tokens=SPECULATIVE_OUTPUT_LENGTH, temperature=0.6, top_k=50, top_p=0.95,checking=False
                )
                speculative_real_output = speculative_tokenizer.decode(checking_generated_ids[:-(checking_generated_ids.shape[1] - small_input_ids.shape[1])])
                target_tokenizer_input = target_tokenizer(speculative_real_output, return_tensors="pt")['input_ids'].to(
                    target_model.device)
                # big model checking
                # if we use the target model at last generation, we directly use 'target_output_id' and 'target_tokenizer_input'
                # if not, we use last the checking_target_ids and 'target_tokenizer_input'
                if use_target:
                    checking_target_ids =torch.cat([target_output_id,target_tokenizer_input], dim=-1)
                else:
                    checking_target_ids =  torch.cat([checking_target_ids,target_tokenizer_input], dim=-1)
                ## TODO: need to optimize the checking generation
                check_output, tgt_kv, target_pooling_hidden_information = generate_with_partial_kv(
                target_model, target_tokenizer, checking_target_ids , valid_tgt_kv,
                    max_new_tokens=1, temperature=0.1, top_k=50, top_p=0.95, checking=True
                )
                # check the entropy of the target model and speculative model.
                with torch.no_grad():
                    prob_target = model_spec_probe(pooling_hidden_information)
                with torch.no_grad():
                    prob_spec = model_target_probe(target_pooling_hidden_information)
                # if the prob of the target model is higher than the prob of the speculative model, we use the speculative model to keep going.
                # if the prob of the target model is lower than the prob of the speculative model, we use the target model to generate the current part.
                if prob_target >= prob_spec:
                    use_target = False
                    valid_tgt_kv = tgt_kv[:-1] # we just want to real generation KV cache,
                    spec_kv = checking_spec_kv
                    generated_ids = checking_generated_ids
                    target_output_id = checking_target_ids
                else:
                    use_target = True
                    # valid_tgt_kv  not change
                    generated_ids = checking_target_ids - target_tokenizer_input
                    start_speculative_text_inputs = small_input_ids
                    #spec_kv = spec_kv # not change


            # Let the target model finish the generation.
            # At the beginning of the generation, Let the target model generate the first part of completion.
            if use_target:
                # record the usage of the target model;
                begin = False
                try_correct_num = try_correct_num + 1
                print('try_correct_num',try_correct_num)
                generated_ids, valid_tgt_kv,output_last_hidden_list = generate_with_partial_kv(
                target_model, target_tokenizer, generated_ids, valid_tgt_kv,
                    max_new_tokens=change_tokens, temperature=0.6, top_k=50, top_p=0.95,checking=False
                )

                # if inferencing the model stops at the first time
                if target_tokenizer.eos_token_id in generated_ids[0, target_prompt_len:]:
                    generated_text = target_tokenizer.decode(generated_ids[0, :], skip_special_tokens=True)
                    break

            if speculative_tokenizer.eos_token_id in generated_ids[0, target_prompt_len:]:
                break
            generated_text = speculative_tokenizer.decode(generated_ids[0, :], skip_special_tokens=True)



        return generated_text, try_correct_num





def process_file_to_json(dir_path, target_model, target_tokenizer,speculative_model, speculative_tokenizer,problem, answer,target_temperature,speculative_temperature,max_new_tokens,model_target_probe,model_spec_probe):
    all_generations = []
    # try:
    start_time = time.time()
    print(f'problem: {problem}')
    result = speculative_decoding(target_model, target_tokenizer, speculative_model, speculative_tokenizer, problem, target_temperature, speculative_temperature,max_new_tokens,model_target_probe,model_spec_probe)
    end_time = time.time()
    generated_text, try_correct_num = result

    print('generated_text',generated_text)
    all_generations.append({
        "input_text": problem,
        "real_answer": generated_text,
        "try_correct_num": try_correct_num,
        "standard_answer": answer,
        "execution_time": f"{end_time - start_time:.2f}s"
    })
    # except Exception as e:
    #     all_generations.append({
    #         "input_text": problem,
    #         "real_answer": None,
    #         "full_answer": None,
    #         "answer": answer,
    #     })

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
    parser.add_argument("--start_dataset", type=int, help="the beginning of the dataset",default=0)
    parser.add_argument("--end_dataset", type=int, help="the end of the dataset",default=1)
    parser.add_argument("--target_probe", type=str, help="target_probe",default="/data/semantic/training/math-500_output_last_hidden_list_best_probe_mse")
    parser.add_argument("--speculative_probe", type=str, help="speculative_probe",default="/home/shaowei/training_probe/math-500_output_last_hidden_list_best_probe_mse")
    parser.add_argument("--target_temperature", type=float, help="target_temperature",default=0.1)
    parser.add_argument("--speculative_temperature", type=float, help="speculative_temperature",default=0.6)
    parser.add_argument("--max_new_tokens", type=int, help="max_new_tokens",default=32000)
    parser.add_argument("--top_p", type=float, help="top_p",default=0.9)
    parser.add_argument("--top_k", type=int, help="top_k",default=50)
    args = parser.parse_args()

    model_target_probe = SemanticEntropyProbTarget(5120, 256)
    model_target_probe.load_state_dict(torch.load(f'{args.target_probe}.pt'))
    model_target_probe = model_target_probe.to(f'cuda:{TARGET_probe}')

    model_spec_probe = SemanticEntropyProbSpec(1536, 256)
    model_spec_probe.load_state_dict(torch.load(f'{args.speculative_probe}.pt'))
    model_spec_probe = model_spec_probe.to(f'cuda:{TARGET_probe}')


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
        process_file_to_json(dir_path, target_model, target_tokenizer,speculative_model, speculative_tokenizer, problem,answer,args.target_temperature,args.speculative_temperature,args.max_new_tokens,model_target_probe,model_spec_probe)