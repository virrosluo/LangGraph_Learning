from IPython import display
from playwright.async_api import async_playwright
from dotenv import load_dotenv

from langchain import hub 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langgraph.graph import END, START, StateGraph

from utils_function.agent_utils import *
from tools.tool_descriptions import *
from utils import update_scratchpad

load_dotenv()

prompt = hub.pull("wfh/web-voyager")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)

agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

workflow = StateGraph(state_schema=AgentState)
workflow.add_node("agent", agent)
workflow.add_edge(start_key=START, end_key="agent")

workflow.add_node("update_scratchpad", update_scratchpad)
workflow.add_edge("update_scratchpad", end_key="agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
}

for node_name, tool in tools.items():
    workflow.add_node(
        node_name,
        RunnableLambda(tool) | (lambda observation: {"observation": observation})
    )
    
    workflow.add_edge(node_name, "update_scratchpad")
    
def select_tool(state: AgentState):
    action = state["prediction"]["action"]
    if "ANSWER" in action:
        return END
    if "retry" in action:
        return "agent"
    return action

workflow.add_conditional_edges("agent", select_tool)

graph = workflow.compile()

async def call_agent(question: str, page, max_step: int = 150):
    input_state = {
        "page": page,
        "input": question,
        "scratchpad": [],
        "observation": "",
    }
    event_stream = graph.astream(input_state)
    
    final_answer = None
    steps = []
    async for event in event_stream:
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")
        display.clear_output(wait=False)
        steps.append(f"{len(steps) + 1}. {action}: {action_input}")
        print("\n".join(steps))
        if "ANSWER" in action:
            final_answer = action_input
            break
    return final_answer

# Start the neccesary components
async def main():
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=True, args=None)
    page = await browser.new_page()
    await page.goto("https://www.google.com")

    result = await call_agent("Could you explain the WebVoyager paper (on arxiv)?", page)
    print(result)