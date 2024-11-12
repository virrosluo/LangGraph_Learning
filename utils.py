from typing import Optional

from langchain_core.language_models import LanguageModelLike
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import StateGraph
from langgraph.types import Checkpointer
from langgraph.store.base import BaseStore
from langgraph.graph.graph import CompiledGraph
from langgraph.utils.runnable import RunnableCallable, RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState, StateSchemaType, StateModifier, MessagesModifier

def create_agent(
    model: LanguageModelLike,
    *,
    state_schema: Optional[StateSchemaType] = None,
    messages_modifier: Optional[MessagesModifier] = None,
    state_modifier: Optional[StateModifier] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
) -> CompiledGraph:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful Coder AI assistant, Writing code is your specialty. You cannot response with natural language."
                " Write a python code or a bash script to progress towards answering the question. "
                " If you are unable to fully answer, that's OK, another assistant will help where you left off."
                " Execute what you can to make progress."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    model_runnable = prompt | model
    
    
    def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
        response = model_runnable.invoke(state, config)
        return {"messages": [response]}
    
    async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
        response = await model_runnable.ainvoke(state, config)
        return {"messages": [response]}
    
    workflow = StateGraph(state_schema=AgentState)
    workflow.add_node("agent", RunnableCallable(call_model, acall_model))
    workflow.add_edge("agent", end_key="__end__")
    
    workflow.set_entry_point("agent")
    return workflow.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )
