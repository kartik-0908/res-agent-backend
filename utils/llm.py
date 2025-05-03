import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv() 

api_key = os.getenv("AZURE_OPENAI_API_KEY")
llm_o1 = AzureChatOpenAI(
    azure_deployment="o1",  # or your deployment
    api_key=api_key,
    api_version="2024-12-01-preview"
)