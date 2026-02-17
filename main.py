from deepagents import create_deep_agent
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
from tavily import TavilyClient
from typing import  Literal




load_dotenv()

HF_TOKEN=os.getenv("HF_TOKEN")
BASE_URL=os.getenv("BASE_URL")
MODEL=os.getenv("MODEL")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

tavily_client=TavilyClient(
    api_key=TAVILY_API_KEY
)

client=ChatOpenAI(
    api_key=HF_TOKEN,
    base_url=BASE_URL,
    model=MODEL
)

count_tool_call=[0]
@wrap_tool_call
def log_tool_call(request,handler):
    count_tool_call[0]+=1
    print("Tool call number: ",count_tool_call)
    print("Middleware tool name: ",request.name if hasattr(request,"name") else str(request))
    print("Middleware tool arguments: ",request.args if hasattr(request,'args') else "N/A")
    result=handler(request)
    print("Tool call number: ",count_tool_call[0]," ended")

    return result

@tool
def Web_Search(
    query:str,
    max_results:int=5,
    include_raw_content:bool=False,
    topic: Literal["financial","General","History"]="History"):

    """
    This is a tool function for web searching as per the user topic and returns max_results as passed parameters
    """
    return tavily_client.search(
        query=query, max_results=max_results, include_raw_content=include_raw_content, topic=topic
    )


agent=create_deep_agent(
    model=client,
    system_prompt=
    """
    You are a helpul assistant that returns the web search result using your tool tavily about the given topic by the user,remember only use the tavily tool for task of general knowledge don;t use your internal knowledge
    """,
    tools=[Web_Search],
    middleware=[log_tool_call]

)

result=agent.invoke({"messages":[{"role":"user","content":"Hi what is the capital of Pakistan???"}]})

# print(result)
print(result["messages"][-1].content)


