import requests

url = "http://194.68.245.149:22096/generate"

data = {
    "text": "Hello",
    "max_new_tokens": 100,
    "temperature": 0.7
}

response = requests.post(url, json=data)
print(response.json())
