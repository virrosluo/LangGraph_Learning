from langchain_core.language_models import BaseChatModel
from langchain_core.agents import AgentFinish
from langgraph.prebuilt import ToolInvocation

def run_chatbot(llm: BaseChatModel):
    def _run(state):
        llm_outcome = llm.invoke(state['messages'])
        return dict(
            messages = [llm_outcome]
        )
    return _run

def should_continue(state):
    message = state["messages"][-1]
    if "function_call" not in message.additional_kwargs:
        return "end"
    else:
        return "continue"