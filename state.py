from typing import TypedDict, Annotated, List, Union, Tuple
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    input: str

    chat_history: List[BaseMessage]

    agent_outcome: Union[AgentAction, AgentFinish, None]

    intermediate_steps: Annotated[List[Tuple[AgentAction, str]], operator.add]