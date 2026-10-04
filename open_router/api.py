import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["OPENROUTER_API_KEY"]

MODEL = "openai/gpt-5.6-luna"
TEMPERATURE = 1.0

def send_request(query: str, model: str = MODEL, temperature: float = TEMPERATURE) -> dict:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": temperature,
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    query = "What is the meaning of life? Answer in 10 words or less."
    response = send_request(query)
    print(response)