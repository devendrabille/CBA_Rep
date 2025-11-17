import os

class Config:
    """Configuration settings for the application."""
    
    # Streamlit settings
    PAGE_TITLE = "Agentic EDA Tool"
    PAGE_LAYOUT = "wide"
    
    # Azure OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")  # e.g., https://<resource>.openai.azure.com
    OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")  # Azure deployment name
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-10-21")  # Default API version

    # Other configurations can be added here as needed
