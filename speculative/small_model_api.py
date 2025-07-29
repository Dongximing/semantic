"""
Usage:

python hidden_states_server.py

Note that each time you change the `return_hidden_states` parameter,
the cuda graph will be recaptured, which might lead to a performance hit.
So avoid getting hidden states and completions alternately.
"""

import requests
import torch

from sglang.test.test_utils import is_in_ci
from sglang.utils import terminate_process, wait_for_server

if is_in_ci():
    from docs.backend.patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd


def main():
    # Launch the server

    port = 30000
    wait_for_server(f"http://130.179.30.7:{port}")

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 10,
    }

    json_data = {
        "text": prompts,
        "sampling_params": sampling_params,
        "return_hidden_states": True,
    }

    response = requests.post(
        f"http://130.179.30.7:{port}/generate",
        json=json_data,
    )


    outputs = response.json()
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
        print()


if __name__ == "__main__":
    main()

    [151644, 872, 198, 11510, 6556, 362, 7755, 5780, 369, 264, 400, 24, 3, 12, 85526, 20408, 23791, 4227, 323, 17933,
     518, 264, 10799, 8061, 26807, 13, 3197, 1340, 22479, 518, 264, 6783, 4628, 315, 400, 82, 3, 40568, 817, 6460, 11,
     279, 4227, 4990, 1059, 220, 19, 4115, 11, 2670, 400, 83, 3, 4420, 7391, 304, 279, 10799, 8061, 13, 3197, 1340,
     22479, 400, 82, 10, 17, 3, 40568, 817, 6460, 11, 279, 4227, 4990, 1059, 220, 17, 4115, 323, 220, 17, 19, 4420, 11,
     2670, 400, 83, 3, 4420, 7391, 304, 279, 10799, 8061, 13, 82610, 362, 7755, 22479, 518, 400, 82, 41715, 37018, 90,
     16, 15170, 17, 31716, 40568, 817, 6460, 13, 7379, 279, 1372, 315, 4420, 279, 4227, 4990, 1059, 11, 2670, 279, 400,
     83, 3, 4420, 7391, 304, 279, 10799, 8061, 624, 5501, 2874, 3019, 553, 3019, 11, 323, 2182, 697, 1590, 4226, 2878,
     1124, 79075, 46391, 151645, 198, 151644, 77091, 198, 151667, 198, 32313, 11, 773, 358, 1184, 311, 11625, 419, 3491,
     911, 362, 7755, 594, 11435, 323, 10799, 8061, 17933, 13, 6771, 752, 1349, 432, 1549, 323, 1430, 311, 3535, 1128,
     594, 2087, 389, 1588, 382]