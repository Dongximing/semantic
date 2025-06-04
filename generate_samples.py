import datetime
import torch
import json
import os
import pickle
import numpy as np
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys 
STOP_TOKENS = [
    ' \n\n', '.\n\n', ':\n\n', '\n\n', ' Wait', 'Alternatively', 'Wait', ' But',
    ')\n\n', '?\n\n', ']\n\n', ').\n\n'
]
AIME_STOP_TOKENS = [
    ' \n\n', '.\n\n', ':\n\n', '\n\n', ' Wait', 'Alternatively', 'Wait', ' But',
    ')\n\n', '?\n\n', ']\n\n', ').\n\n', ' Alternatively','Hmm',' Hmm'
]
NUMBER = 0
def predict(tokenizer, model, input_data, temperature, return_full=False, return_latent=False):
    max_new_tokens = 150
    inputs = tokenizer(input_data, return_tensors="pt").to(f"cuda:{NUMBER}")
    initial_length = len(inputs['input_ids'][0])
    STOP = None
    stopping_criteria = None
    if STOP_TOKENS is not None:
        from transformers import StoppingCriteria, StoppingCriteriaList
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

        stopping_criteria_obj = StoppingCriteriaSub(
                stops=AIME_STOP_TOKENS,
                initial_length=initial_length,
                tokenizer=tokenizer
            )
        stopping_criteria = StoppingCriteriaList([
            stopping_criteria_obj
        ])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            stopping_criteria=stopping_criteria,
        )
    full_answer = tokenizer.decode(outputs.sequences[0],skip_special_tokens=True)
    real_answer = tokenizer.decode(outputs.sequences[0][initial_length:], skip_special_tokens=True)
    triggered_stop = stopping_criteria_obj.triggered_stop
    answer = full_answer
    if full_answer.startswith(input_data):
        answer = full_answer[len(input_data):]

    # hidden states
    if 'decoder_hidden_states' in outputs.keys():
        hidden = outputs.decoder_hidden_states
    else:
        hidden = outputs.hidden_states

    last_input = hidden[-1]
    last_token_embedding = torch.stack([layer[:, -1, :] for layer in last_input]).cpu()
    sec_last_input = hidden[-2]
    sec_last_token_embedding = torch.stack([layer[:, -1, :] for layer in sec_last_input]).cpu()
    last_tok_bef_gen_input = hidden[0]
    last_tok_bef_gen_embedding = torch.stack([layer[:, -1, :] for layer in last_tok_bef_gen_input]).cpu()
    output_last_hidden_list = torch.stack([layer[-1][:, -1, :] for layer in hidden]).cpu()


    last_output = hidden[-1]
    last_output = last_output[-1]
    last_hidden_state = last_output[:, -1, :].cpu()

    sec_last_input = hidden[-2]
    sec_last_input = sec_last_input[-1]
    sec_last_hidden_state = sec_last_input[:, -1, :].cpu()

    last_input_token = hidden[0]
    last_input_token = last_input_token[-1]
    last_input_token_state = last_input_token[:, -1, :].cpu()

    # token log likelihood, probs, ppl
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True)
    log_likelihoods = [score.item() for score in transition_scores[0]]
    probs = [np.exp(x) for x in transition_scores[0].tolist()]
    mean_neg_logprob = -np.mean(transition_scores[0].cpu().numpy())
    ppl = np.exp(mean_neg_logprob)

    hidden_states = (
        last_hidden_state, sec_last_hidden_state, last_input_token_state,
        last_token_embedding, sec_last_token_embedding, last_tok_bef_gen_embedding,output_last_hidden_list
    )

    return (real_answer,answer, log_likelihoods, probs, ppl, triggered_stop ,hidden_states)



def process_file_to_pickle(json_path, out_pkl_path, tokenizer, model, num_generations):
    with open(json_path, 'r', encoding='utf-8') as f:
        alldata = json.load(f)
    all_generations = []
  
    log_file = out_pkl_path.replace('.pkl', '.log')
    n = int(len(alldata))
    data = alldata[:n]
    print(len(data))

    for index, element in enumerate(data):
        input_text = element['text']
        for i in range(num_generations+1):
            log_entry = {
                "time": datetime.datetime.now().isoformat(),
                "sample_index": index,
                "generation_index": i,
                "input_text_preview": input_text[:80] + ("..." if len(input_text) > 80 else ""),
            }
            try:
                if i == 0:
                    (
                        real_answer, predicted_answer, log_likelihoods, probs, ppl,triggered_stop,
                        (last_hidden_state, sec_last_hidden_state, last_input_token_state,
                         embedding, emb_last_before_gen, emb_before_eos, output_last_hidden_list)
                    ) = predict(tokenizer, model, input_text, temperature=0.1, return_full=False, return_latent=True)
                else:
                    (
                        real_answer,predicted_answer, log_likelihoods, probs, ppl,triggered_stop,
                        (last_hidden_state, sec_last_hidden_state, last_input_token_state,
                         embedding, emb_last_before_gen, emb_before_eos, output_last_hidden_list)
                    ) = predict(tokenizer, model, input_text, temperature=0.7, return_full=False, return_latent=True)


                log_entry.update({
                    "status": "success",
                    "predicted_answer_preview": str(predicted_answer)[:80] + ("..." if predicted_answer and len(predicted_answer) > 80 else ""),
                    "ppl": float(ppl) if ppl is not None else None,
                    "log_likelihoods_len": len(log_likelihoods) if log_likelihoods is not None else 0,
                })
                print('triggered_stop',triggered_stop)
                if i == 0:
                    all_generations.append({
                        "most_input_text": input_text,
                        'most_real_answer': real_answer,
                        "most_predicted_answer": predicted_answer,
                        "most_last_hidden_state": last_hidden_state.cpu(),
                        "most_sec_last_hidden_state": sec_last_hidden_state.cpu(),
                        "most_last_input_token_state": last_input_token_state.cpu(),
                        "most_output_last_hidden_list": output_last_hidden_list.cpu(),
                        "most_ppl": ppl,
                        "most_log_likelihoods": log_likelihoods,
                        "most_probs": probs,
                        "most_embedding": embedding.cpu(),
                        "most_emb_last_before_gen": emb_last_before_gen.cpu(),
                        "most_emb_before_eos": emb_before_eos,
                        "most_generation_index": -1,
                        "most_sample_index": index,
                        'triggered_stop':triggered_stop
                    })
                else:

                    all_generations.append({
                        "input_text": input_text,
                        'real_answer':real_answer,
                        "predicted_answer": predicted_answer,
                        "last_hidden_state": last_hidden_state.cpu(),
                        "sec_last_hidden_state": sec_last_hidden_state.cpu(),
                        "last_input_token_state": last_input_token_state.cpu(),
                        "output_last_hidden_list": output_last_hidden_list.cpu(),
                        "ppl": ppl,
                        "log_likelihoods": log_likelihoods,
                        "probs": probs,
                        "embedding": embedding.cpu(),
                        "emb_last_before_gen": emb_last_before_gen.cpu(),
                        "emb_before_eos": emb_before_eos,
                        "generation_index": i-1,
                        "sample_index": index,
                        'triggered_stop': triggered_stop
                    })
            except Exception as e:
                log_entry.update({
                    "status": "error",
                    "error_msg": str(e),
                    "traceback": traceback.format_exc()
                })
                if i == 0:
                    all_generations.append({
                        "most_input_text": input_text,
                        "most_real_answer": None,
                        "most_predicted_answer": None,
                        "most_output_last_hidden_list": None,
                        "most_ppl": None,
                        "most_log_likelihoods": None,
                        "most_embedding": None,
                        "most_last_hidden_state": None,
                        "most_sec_last_hidden_state": None,
                        "most_last_input_token_state": None,
                        "most_generation_index": -1,
                        "most_sample_index": index,
                        'triggered_stop': None
                    })
                else:
                    all_generations.append({
                        "input_text": input_text,
                        "real_answer": None,
                        "predicted_answer": None,
                        "output_last_hidden_list":None,
                        "ppl": None,
                        "log_likelihoods": None,
                        "embedding": None,
                        "last_hidden_state": None,
                        "sec_last_hidden_state": None,
                        "last_input_token_state": None,
                        "generation_index": i-1,
                        "sample_index": index,
                        'triggered_stop': None
                    })

            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    with open(out_pkl_path, "wb") as f:
        pickle.dump(all_generations, f)
    print(f"Saved {out_pkl_path} with {len(all_generations)} generations.")



# wail /home/cs/staff/shaowei/hf/math-result_left
#quail /data/ximing/math-result_left
def inference_model_pickle(task_name: str, model, tokenizer, base_dir,
                           start=20, end=250, num_generations=20):

    for number in range(start, end):
        if task_name == 'math-500':
            dirname = f'data-500-temp0_{number}'
            dir_path = os.path.join(base_dir, dirname)
            json_path = os.path.join(dir_path, f'seg_by_stop_{number}.json')
            out_pkl_path = os.path.join(dir_path, f'new_generations_{number}.pkl')
        elif task_name == 'aime':
            dirname = f'data-500-temp0_{number}'
            dir_path = os.path.join(base_dir, dirname)
            json_path = os.path.join(dir_path, f'seg_by_stop_{number}.json')
            out_pkl_path = os.path.join(dir_path, f'new_generations_{number}.pkl')


        if not os.path.isfile(json_path):
            print(f"[Warning] {json_path} does not exist! Skipping...")
            continue

        print(f"[Info] Processing file: {json_path}")
        process_file_to_pickle(json_path, out_pkl_path, tokenizer, model, num_generations)

    print("[Info] Processing completed.")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/QwQ-32B-AWQ",
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/QwQ-32B-AWQ",
        torch_dtype=torch.float16,
        device_map=f"cuda:{NUMBER}"
    )
    base_dir= '/home/cs/staff/shaowei/semantic/aime'
    inference_model_pickle(task_name="aime", model=model,base_dir=base_dir, tokenizer=tokenizer,start=0, end=60)
    print("done")
