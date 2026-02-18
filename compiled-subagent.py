from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents.middleware import wrap_tool_call
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
MODEL=os.getenv("MODEL")

HF_TOKEN=os.getenv("HF_TOKEN")
BASE_URL=os.getenv("BASE_URL")
MODEL=os.getenv("MODEL")

client=ChatOpenAI(
    api_key=HF_TOKEN,
    base_url=BASE_URL,
    model=MODEL
)

agent=create_agent(
    # tools=[],
    model=client,
    system_prompt="You are a agent that returns the question in reverse order asked by user",
)

reverse_subagent=CompiledSubAgent(
    name="Reverse agent",
    description="Reverse agent that returns the question in reverse order asked by user",
    runnable=agent
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

agent=create_deep_agent(
    system_prompt=
    """
    You are a agent that uses subagent reverse_subagent to answer the question of user starting with ! and answers itself when not statts with !
    
    """,
    model=client,
    subagents=[reverse_subagent],
    middleware=[log_tool_call]
)

result=agent.invoke({"messages":[{"role":"user","content":"What is capital of pakistan???"}]})
print(result["messages"][-1].content)