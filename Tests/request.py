import requests

r = requests.post("http://127.0.0.1:8080/predictions/DialoGPT-medium", "How are you?")
print(r.text)