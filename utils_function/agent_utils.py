import asyncio
import platform
import base64
import time

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import chain as chain_decorator

from playwright.sync_api import Page

from state import AgentState

def run_supervisor(data, agent_runnable):
    agent_outcome = agent_runnable.invoke(data)
    return dict(
        agent_outcome=agent_outcome.next
    )

def run_agent(data, agent_runnable, name):
    agent_outcome = agent_runnable.invoke(data) # Output after passed through JSONAgentOutputParser
    message: AIMessage = agent_outcome["messages"][-1]
    message = HumanMessage(content=message.content, name=name)
    return dict(
        agent_outcome=message,
        messages=[message]
    )
    
def click(state: AgentState):
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = click_args[0]
    bbox_id = int(bbox_id)
    try:
        bbox = state["bboxes"][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox["x"], bbox["y"]
    page.mouse.click(x, y)
    return {"observation": f"Clicked {bbox_id}"}

def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if type_args is None or len(type_args) != 2:
        return f"Failed to type text into bounding box labeled as number {type_args}"
    bbox_id = type_args[0]
    bbox_id = int(bbox_id)
    bbox = state["bboxes"][bbox_id]
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    page.mouse.click(x, y)
    
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    page.keyboard.press(select_all)
    page.keyboard.press("Backspace")
    page.keyboard.type(text_content)
    page.keyboard.press("Enter")
    return {
        "observation": f"Typed {text_content} and submitted"
    }

def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return f"Failed to scroll the page"

    target, direction = scroll_args
    
    if target.upper() == "WINDOW":
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        page.mouse.move(x, y)
        page.mouse.wheel(0, scroll_direction)
        
    return {"observation": f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"}

def wait(state: AgentState):
    sleep_time = 5
    time.sleep(sleep_time)
    return {"observation": f"Waited for {sleep_time}s."}

def go_back(state: AgentState):
    page = state["page"]
    page.go_back()
    return {"observation": f"Navigated back a page to {page.url}."}

def to_google(state: AgentState):
    page = state["page"]
    page.goto("https://www.google.com")
    return {"observation": f"Navigated to google.com."}

with open("mark_page.js") as f:
    mark_page_script = f.read()

@chain_decorator
def mark_page(page: Page):
    page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = page.evaluate("markPage()")
            break
        except:
            time.sleep(3)
    screenshot = page.screenshot()
    page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes
    }

def annotate(state):
    marked_page = mark_page.with_retry().invoke(state["page"])
    return {**state, **marked_page}

def format_descriptions(state: AgentState):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}

def parse(text: dict) -> dict:
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]

    action_str = action_block[len(action_prefix) :]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}