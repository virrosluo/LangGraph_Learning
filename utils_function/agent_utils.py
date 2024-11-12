from langchain_core.messages import HumanMessage, AIMessage

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