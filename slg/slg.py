
"""
Usage:
python hidden_states.py

Note that each time you change the `return_hidden_states` parameter,
the cuda graph will be recaptured, which might lead to a performance hit.
So avoid getting hidden states and completions alternately.
"""

import torch

import sglang as sgl


def main():
    prompts = [
        "How many positive whole-number divisors does 196 have?\nPlease reason step by step, and put your final answer within \\boxed{}.<think>",
    ]
    # Create an LLM.
    llm = sgl.Engine(
        model_path="Qwen/QwQ-32B",
        tp_size=4,

    )

    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 10000,
        "stop_token_ids":[4710,382,1447,271,692,1939,2533,3593,13824,14190],
        "no_stop_trim": True
    }

    outputs = llm.generate(
        prompts, sampling_params=sampling_params, enable_return_hidden_states=True,
    )

    llm.shutdown()
    # print(outputs)

    for prompt, output in zip(prompts, outputs):
        for i in range(len(output["meta_info"]["hidden_states"])):
            output["meta_info"]["hidden_states"][i] = torch.tensor(
                output["meta_info"]["hidden_states"][i], dtype=torch.bfloat16
            )
        print("===============================")
        print(
            f"Prompt: {prompt}\n"
            f"Generated text: {output['text']}\n"
            f"Prompt_Tokens: {output['meta_info']['prompt_tokens']}\t"
            f"Completion_tokens: {output['meta_info']['completion_tokens']}"
        )
        print("Hidden states: ")
        hidden_states = torch.cat(
            [
                i.unsqueeze(0) if len(i.shape) == 1 else i
                for i in output["meta_info"]["hidden_states"]
            ]
        )
        print(hidden_states)
        print(hidden_states.size())
        # print()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()
