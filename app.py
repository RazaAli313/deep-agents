from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from typing import Literal
from dotenv import load_dotenv
import os


load_dotenv()
MODEL=os.getenv("MODEL",'openai/gpt-oss-20b')
HF_TOKEN=os.getenv("HF_TOKEN")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

client=ChatOpenAI(
    model=MODEL,
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
    
)

tavily_client=TavilyClient(
    api_key=TAVILY_API_KEY
)
def internet_search_tool(
    
  query:str,
  max_result:int=5,
  include_raw_content:bool=False,
  topic:Literal["general","financial"]="general"):
    """
    This is a tool function that searches internet for the given user query
    """

    return tavily_client.search(
    query=query,max_result=max_result,include_raw_content=include_raw_content,topic=topic

)
def get_name()->str:
    """
    This  is a tool function  that returns my name
    """
    return "Syed Muhammad Raza Ali Zaidi"
# agent=create_deep_agent(
#     model=client,
#     tools=[get_name,internet_search_tool],
#     system_prompt="You are a helpful assistant that tells my name using get_name tool and searches internet on the given user query using internet_search tool",
  
# )

# result=agent.invoke({"messages":[{"role":"user","content":"What is langgrap???"}]})
# print(result["messages"][-1].content)


research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

"""

agent = create_deep_agent(
    model=client,
    tools=[internet_search_tool],
    system_prompt=research_instructions
)
result = agent.invoke({"messages": [{"role": "user", "content": "what is latest python version?"}]})

print(result["messages"][-1].content)