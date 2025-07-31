import requests
from requests.exceptions import RequestException

small_input = 'hell0'
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
}
try:
    speculative_outputs = requests.post(
        "http://194.68.245.149:8888/generate",  # 使用 8888
        json=json_data,
        timeout=10
    )
    speculative_outputs.raise_for_status()
    print(speculative_outputs.json())
except RequestException as e:
    print(f"Request failed: {e}")