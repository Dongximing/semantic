import requests

url = "http://194.68.245.149:{22071}/generate"
small_input  = 'hell0'
sampling_params = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens": 5,
    "stop_token_ids": [4710, 382, 1447, 271, 692, 1939, 2533, 3593],
    "no_stop_trim": True
}
json_data = {
    "text": [small_input],
    "sampling_params": sampling_params,
    "return_hidden_states": True,
}
speculative_outputs = requests.post(
    f"http://194.68.245.149:{22071}/generate",
    json=json_data,
)

print(speculative_outputs.json())
