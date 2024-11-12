from typing import TypedDict, Annotated, List, Union, Tuple, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
import operator

class AgentState(TypedDict):
    '''
    The LangGraph ecosystem will automatically casting the return value of 'each node' into this type structure.
    '''    
    messages: Annotated[Sequence[BaseMessage], operator.add]

    agent_outcome: Union[BaseMessage, None]
    
class SearchTool(BaseModel):
    '''
    Let the model understand what structure that the tool input should be.
    '''
    query: str = Field(description="Query to look up online")
    
    return_direct: bool = Field(
        description="Whether or the result of this should be returned directly to the user without you seeing what it is",
        default=False
    )