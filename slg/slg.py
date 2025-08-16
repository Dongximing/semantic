import requests
from openai import timeout

token = "HuaweiDockerSquadAssemble1105.VeryComplicatedTokenThatNoOneCanGuessHaHa!"
small_input  = 'How many positive whole-number divisors does 196 have?'
sampling_params = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens":  8,

    "no_stop_trim": True
}
json_data = {
    "text": [small_input],
    "sampling_params": sampling_params,
}
for i in range(1000):
    speculative_outputs = requests.post(
        "https://lux-2-cyber-09.tailb940e6.ts.net/sglang2/generate",
        json=json_data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout =60

    )

    print(speculative_outputs.json())



