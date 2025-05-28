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

def predict(tokenizer, model, input_data, temperature, return_full=False, return_latent=False):
    max_new_tokens = 200
    inputs = tokenizer(input_data, return_tensors="pt").to("cuda")
    initial_length = len(inputs['input_ids'][0])
    stopping_criteria = None
    if STOP_TOKENS is not None:
        from transformers import StoppingCriteria, StoppingCriteriaList
        class StoppingCriteriaSub(StoppingCriteria):
            def __init__(self, stops, tokenizer, initial_length=None):
                super().__init__()
                self.stops = stops
                self.initial_length = initial_length
                self.tokenizer = tokenizer
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                for stop in self.stops:
                    if stop in generation:
                        print(f"[StoppingCriteria] Matched stop: {repr(stop)}")
                        return True
                return False
        stopping_criteria = StoppingCriteriaList([
            StoppingCriteriaSub(
                stops=STOP_TOKENS,
                initial_length=initial_length,
                tokenizer=tokenizer
            )
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

    full_answer = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
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
        last_token_embedding, sec_last_token_embedding, last_tok_bef_gen_embedding
    )
    return (answer, log_likelihoods, probs, ppl, hidden_states)



def process_file_to_pickle(json_path, out_pkl_path, tokenizer, model, num_generations):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_generations = []
  
    log_file = out_pkl_path.replace('.pkl', '.log')
    
    for index, element in enumerate(data):
        input_text = element['text']
        for i in range(num_generations):
            log_entry = {
                "time": datetime.datetime.now().isoformat(),
                "sample_index": index,
                "generation_index": i,
                "input_text_preview": input_text[:80] + ("..." if len(input_text) > 80 else ""),
            }
            try:
                (
                    predicted_answer, log_likelihoods, probs, ppl,
                    (last_hidden_state, sec_last_hidden_state, last_input_token_state,
                     embedding, emb_last_before_gen, emb_before_eos)
                ) = predict(tokenizer, model, input_text, temperature=0.7, return_full=False, return_latent=True)

                log_entry.update({
                    "status": "success",
                    "predicted_answer_preview": str(predicted_answer)[:80] + ("..." if predicted_answer and len(predicted_answer) > 80 else ""),
                    "ppl": float(ppl) if ppl is not None else None,
                    "log_likelihoods_len": len(log_likelihoods) if log_likelihoods is not None else 0,
                })

                all_generations.append({
                    "input_text": input_text,
                    "predicted_answer": predicted_answer,
                    "last_hidden_state": last_hidden_state.cpu(),
                    "sec_last_hidden_state": sec_last_hidden_state.cpu(),
                    "last_input_token_state": last_input_token_state.cpu(),
                    "ppl": ppl,
                    "log_likelihoods": log_likelihoods,
                    "probs": probs,
                    "embedding": embedding.cpu(),
                    "emb_last_before_gen": emb_last_before_gen.cpu(),
                    "emb_before_eos": emb_before_eos,
                    "generation_index": i,
                    "sample_index": index
                })
            except Exception as e:
                log_entry.update({
                    "status": "error",
                    "error_msg": str(e),
                    "traceback": traceback.format_exc()
                })
                all_generations.append({
                    "input_text": input_text,
                    "predicted_answer": None,
                    "ppl": None,
                    "log_likelihoods": None,
                    "embedding": None,
                    "last_hidden_state": None,
                    "sec_last_hidden_state": None,
                    "last_input_token_state": None,
                    "generation_index": i,
                    "sample_index": index
                })
            # 日志每轮 append 到 log_file
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    with open(out_pkl_path, "wb") as f:
        pickle.dump(all_generations, f)
    print(f"Saved {out_pkl_path} with {len(all_generations)} generations.")




def inference_model_pickle(task_name: str, model, tokenizer, base_dir='/home/cs/staff/shaowei/hf/math-result_left',
                           start=0, end=250, num_generations=20):

    for number in range(start, end):
        dirname = f'data-500-temp0_{number}'
        dir_path = os.path.join(base_dir, dirname)
        json_path = os.path.join(dir_path, f'seg_by_stop_{number}.json')
        out_pkl_path = os.path.join(dir_path, f'generations_{number}.pkl')

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
        device_map={'': 'cuda:0'}
    )
    inference_model_pickle(task_name="math-500", model=model, tokenizer=tokenizer,start=250, end=500)
    print("done")
