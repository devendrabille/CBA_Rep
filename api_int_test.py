import streamlit as st
from openai import AzureOpenAI

deployment = "simplegptnano"

# --- Azure OpenAI Client Setup ---
client = AzureOpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    api_version="2024-12-01-preview",
    azure_endpoint=st.secrets["OPENAI_ENDPOINT"]
)

st.title("Azure OpenAI GPT.nano Test")

prompt = st.text_input("Enter a prompt for GPT.nano")

if prompt:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            messages=messages,
            max_tokens=16384,
            model=deployment
        )
        st.write("### Response")
        st.write(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error calling Azure OpenAI API: {e}")
