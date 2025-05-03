import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI

load_dotenv() 

api_key = os.getenv("AZURE_OPENAI_API_KEY")

class State(TypedDict):
    # messages is a list; add_messages will append new entries
    messages: Annotated[list, add_messages]

# 2) Build the graph
graph_builder = StateGraph(State)

# 3) Initialize an LLM (here using OpenAI's chat endpoint)
#    Make sure OPENAI_API_KEY is set in your environment
llm_o1 = AzureChatOpenAI(
    azure_deployment="o1",  # or your deployment
    api_key=api_key,
    api_version="2024-12-01-preview"
)

# 4) Define the chatbot node
def chatbot_node(state: State):
    # Invoke the model with the conversation so far
    return {"messages": [llm_o1.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot_node)
# tell the graph where to start and finish
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 5) Compile into a runnable graph
graph = graph_builder.compile()


class ChatRequest(BaseModel):
    message: str

# Response schema
class ChatResponse(BaseModel):
    response: str