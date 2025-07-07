import requests
import os
from dotenv import load_dotenv

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

def ask_ollama(prompt, history=None):
    if history is None:
        history = []
    full_prompt = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history])
    full_prompt += f"\nUser: {prompt}\nAssistant:"

    response = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": "mistral",
        "prompt": full_prompt,
        "stream": False
    })

    return response.json().get("response", "").strip()