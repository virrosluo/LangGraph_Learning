from dotenv import load_dotenv

from langchain import hub
from langchain.agents import create_structured_chat_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph import END, StateGraph

from state import AgentState
from utils import *

load_dotenv()

tools = [TavilySearchResults(max_results=1)]

prompt = hub.pull("hwchase17/structured-chat-agent")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

agent_runnable = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Helper class which is useful for running tools
tool_executor = ToolExecutor(tools=tools)

workflow = StateGraph(
    state_schema=AgentState
)

workflow.add_node("agent", run_agent(agent_runnable=agent_runnable))
workflow.add_node("action", execute_tools(tool_executor=tool_executor))

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    source="agent",
    path=should_continue,
    path_map={
        "continue": "action",
        "end": END
    }
)

workflow.add_edge(
    start_key="action",
    end_key="agent"
)

app = workflow.compile()

inputs = {
    "input": "what is the weather in sf",
    "chat_history": []
}

for s in app.stream(inputs):
    print(list(s.values())[0])
    print("--------------------------------------------")