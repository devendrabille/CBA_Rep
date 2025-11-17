import os
import json
import requests

class AzureOpenAI:
    def __init__(self, api_key, endpoint, deployment_name, api_version="2024-10-21"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

    def chat(self, messages, max_completion_tokens=16384):
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
        payload = {
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }
        response = requests.post(url, headers=self._headers(), json=payload)
        response.raise_for_status()
        return response.json()

def create_client():
    api_key = os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("OPENAI_ENDPOINT")
    deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME")

    if not api_key or not endpoint or not deployment_name:
        raise ValueError("Missing environment variables for Azure OpenAI client.")

    return AzureOpenAI(api_key, endpoint, deployment_name)