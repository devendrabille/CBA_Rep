
import os
from openai import AzureOpenAI

# Load credentials from environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Validate credentials
if not all([api_key, endpoint, deployment_id]):
    raise ValueError("Missing one or more required environment variables: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version="2023-07-01-preview",
    azure_endpoint=endpoint
)

# Prepare a simple test message
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, can you summarize Azure OpenAI deployment types?"}
]

# Send request to Azure OpenAI
response = client.chat.completions.create(
    deployment_id=deployment_id,
    messages=messages,
    max_tokens=200
)

# Print the model's response
print("Model Response:\n")
print(response.choices[0].message.content)
