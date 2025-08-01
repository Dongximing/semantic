import requests

small_input  = 'How many positive whole-number divisors does 196 have?'
sampling_params = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens":  1,
    "stop_token_ids": [4710, 382, 1447, 271, 692, 1939, 2533, 3593],
    "no_stop_trim": True
}
json_data = {
    "text": [small_input],
    "sampling_params": sampling_params,
}
speculative_outputs = requests.post(
    "https://m2d1q0gzo61thi-8800.proxy.runpod.net/generate",
    json=json_data,

)

print(speculative_outputs)
