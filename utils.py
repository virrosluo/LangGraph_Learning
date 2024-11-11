from langchain_core.runnables import Runnable
from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor

def run_agent(agent_runnable: Runnable):
    def _run(data):
        agent_outcome = agent_runnable.invoke(data)
        return dict(
            agent_outcome=agent_outcome
        )
    return _run

def execute_tools(tool_executor: ToolExecutor):
    def _execute(data):
        agent_action = data["agent_outcome"]
        agent_action.tool_input["query"] = agent_action.tool_input["query"]["description"]
        output = tool_executor.invoke(agent_action)
        return dict(
            intermediate_steps=[(agent_action, str(output))]
        )
    return _execute

def should_continue(data):
    if isinstance(data["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"