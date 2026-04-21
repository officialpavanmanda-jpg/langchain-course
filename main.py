import os

from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

@tool
def search(query: str) -> str:
    """
    Tools that searches over internet
    Args: 
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return "Tokyo's weather is sunny"

ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
llm = ChatOllama(ollama_model)
tools = [search]
agent = create_agent(model=llm,tools=tools)

def main():
    print("Hello from langchain-courses search-agent")
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tokyo ?")]}
    )
    print(result)

if __name__ == "__main__":
    main()
