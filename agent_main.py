from IPython import display
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

from langchain import hub 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langgraph.graph import END, START, StateGraph

from utils_function.agent_utils import *
from tools.tool_descriptions import *
from utils import update_scratchpad

from PIL import Image
from io import BytesIO

load_dotenv()

prompt = hub.pull("wfh/web-voyager")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
)

agent = annotate | RunnablePassthrough.assign(prediction=format_descriptions | prompt | llm | StrOutputParser() | parse) | RunnablePassthrough.assign(scratchpad=lambda x: [])

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

from playwright.sync_api import sync_playwright
import base64
from PIL import Image
from io import BytesIO
    
def call_agent(question: str, page, max_step: int = 150):
    input_state = {
        "page": page,
        "input": question,
        "scratchpad": [],
        "observation": "",
    }
    event_stream = graph.stream(input_state)
    
    step = 0
    for event in event_stream:
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")

        img_data = None
        if "img" in event["agent"]:
            img_data = Image.open(BytesIO(base64.b64decode(event["agent"]["img"])))
        
        yield (f"{step + 1}. {action}: {action_input}", img_data)
        
        step += 1
        if "ANSWER" in action or step >= max_step:
            break
    return

# # Start the necessary components
# def search(question, max_step):
#     p = sync_playwright().start()
#     browser = p.chromium.launch(headless=True, args=None)
#     page = browser.new_page()
#     page.goto("https://www.google.com")

#     for step_info, img in call_agent(question, page, max_step=max_step):
#         yield step_info, img
        
#     browser.close()
#     p.stop()
    
# # Start the neccesary components
# def main():
#     browser = sync_playwright().start()
#     browser = browser.chromium.launch(headless=True, args=None)
#     page = browser.new_page()
#     page.goto("https://www.google.com")

#     result = call_agent("Give me a summary about cabibara?", page, max_step=3)
#     print(result)
    
# main()