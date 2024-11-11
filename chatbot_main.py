from dotenv import load_dotenv

from langchain import hub
from langchain.agents import create_structured_chat_agent, create_openai_functions_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.prebuilt.tool_node import ToolNode
from langgraph.graph import END, StateGraph, MessagesState

from utils_function.chatbot_utils import *

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

workflow = StateGraph(
    state_schema=MessagesState
)

workflow.add_node("agent", run_chatbot(llm=llm))

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    source="agent",
    path=should_continue,
    path_map={
        "continue": "agent",
        "end": END
    }
)

app = workflow.compile()

inputs = {
    "messages": ["what is the weather in sf"]
}

for s in app.stream(inputs):
    print(list(s.values())[0])
    print("--------------------------------------------")