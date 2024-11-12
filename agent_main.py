from dotenv import load_dotenv
import functools 

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import create_react_agent

from state import AgentState, SearchTool
from utils_function.agent_utils import *
from utils import create_agent
from tools.tool_descriptions import *

load_dotenv()

search_tool = TavilySearchResults(max_results=1, args_schema=SearchTool)
python_tool = PythonREPLTool()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)

members = ["Researcher", "Coder"]
options = ["FINISH"] + members
supervisor_prompt = (
    "You are a supervisor tasked with managing a conversation between the following workers: {members}. " 
    "Given the following user request, response with the worker to act next. "
    "Each worker will perform a task and respond with their results and status. "
    "Given the conversation below, who should act next? Or should we FINISH? Select one of {options}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", supervisor_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_agent = prompt | llm.with_structured_output(schema=routeResponse)
supervisor_node = functools.partial(
    run_supervisor,
    agent_runnable=supervisor_agent
)

research_agent = create_react_agent(
    model=llm,
    tools=[search_tool]
)
research_node = functools.partial(
    run_agent,
    agent_runnable=research_agent,
    name="Researcher"
)

code_agent = create_agent(
    model=llm,
)
code_node = functools.partial(
    run_agent,
    agent_runnable=code_agent,
    name="Coder"
)

workflow = StateGraph(state_schema=AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("Supervisor", supervisor_node)

for member in members:
    workflow.add_edge(member, "Supervisor")
    
workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["agent_outcome"],
    {
        "Researcher": "Researcher",
        "Coder": "Coder",
        "FINISH": END
    }
)

workflow.add_edge(START, "Supervisor")
graph = workflow.compile()

input = {
    # "messages": [HumanMessage(content="Code hello world and print it to the terminal", name="Human")],
    "messages": [HumanMessage(content="Research and write a code to crawl the research study on pikas.", name="Human")],
}

for s in graph.stream(input):
    print(s)
    print("----------------------------------------")