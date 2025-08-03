import requests

small_input  = 'How many positive whole-number divisors does 196 have?'
sampling_params = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens":  8000,

    "no_stop_trim": True
}
json_data = {
    "text": [small_input],
    "sampling_params": sampling_params,
}
speculative_outputs = requests.post(
    "https://0jlklcgw6wsdau-8800.proxy.runpod.net/generate",
    json=json_data,

)

print(speculative_outputs.json())



