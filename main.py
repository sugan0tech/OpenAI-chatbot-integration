import os
import requests
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_api_key : str = os.getenv("OPENAI_API_KEY")
print(openai_api_key)



def get_openai_completion(input_text: str, api_key: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": input_text}],
        "temperature": 0.7
    }

    url = "https://api.openai.com/v1/chat/completions"

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        result = response.json()
        completion = result['choices'][0]['message']['content']
        return completion
    else:
        print("Request failed with status code:", response.status_code)
        print("Error message:", response.text)
        return None 
    print(response.status_code)


if __name__ == "__main__":
    usr_query : str = input("Enter the contect")
    get_openai_completion(usr_query, openai_api_key)