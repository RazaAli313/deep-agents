import modal
from langchain_modal import ModalSandbox
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

MODEL=os.getenv("MODEL")
BASE_URL=os.getenv("BASE_URL")
HF_TOKEN=os.getenv("HF_TOKEN")

client=ChatOpenAI(
    model=MODEL,
    api_key=HF_TOKEN,
    base_url=BASE_URL

)

# app=modal.App("Sandbox")
app=modal.App.lookup("Sandbox",create_if_missing=True)
image=modal.Image.debian_slim(python_version="3.12")
sandbox=modal.Sandbox.create(app=app,image=image)
backend=ModalSandbox(sandbox=sandbox)

agent=create_deep_agent(
 model=client,
 system_prompt=
 """
 You are a react agent that writes and run Python code in sandbox
 """,
 backend=backend
)

result=agent.invoke({"messages":[{"role":"user","content":"Write unit tests in pytest for sum function and run"}]})

print(result)
# print(result["messages"][-1].content)
modal.sandbox.terminate()