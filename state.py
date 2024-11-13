import operator

from typing import TypedDict, Optional, List, Annotated
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from playwright.sync_api import Page

class BBox(TypedDict):
    x: float
    y: float
    
class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]

class AgentState(TypedDict):
    '''
    The LangGraph ecosystem will automatically casting the return value of 'each node' into this type structure.
    '''
    page: Page
    input: str
    img: str
    bboxes: List[BBox]
    prediction: Prediction
    
    scratchpad: Annotated[List[BaseMessage], operator.add]
    observation: str
    
class SearchTool(BaseModel):
    '''
    Let the model understand what structure that the tool input should be.
    '''
    query: str = Field(description="Query to look up online")
    
    return_direct: bool = Field(
        description="Whether or the result of this should be returned directly to the user without you seeing what it is",
        default=False
    )