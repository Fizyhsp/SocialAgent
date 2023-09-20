from datetime import time
from typing import List, Callable, Any

from jinja2 import Template
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import BaseMessage, SystemMessage, HumanMessage
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import BaseModel



class AgentPrompt(BaseChatPromptTemplate, BaseModel):
    """Prompt for AutoGPT."""

    agent_info: dict
    actions_info: List[dict]
    token_counter: Callable[[str], int]
    send_token_limit: int = 4196
    base_prompt: str


    def construct_full_prompt(self, kwargs) -> str:
        # Construct full prompt
        base_template = Template(self.base_prompt)
        full_prompt = base_template.render(self.agent_info, actions=self.actions_info, **kwargs)
        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        base_prompt = SystemMessage(content=self.construct_full_prompt(kwargs))
        used_tokens = self.token_counter(base_prompt.content)

        # Memory
        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]
        relevant_docs = memory.get_relevant_documents(str(previous_messages[-10:]))
        relevant_memory = [d.page_content for d in relevant_docs]
        relevant_memory_tokens = sum(
            [self.token_counter(doc) for doc in relevant_memory]
        )
        while used_tokens + relevant_memory_tokens > 2500:
            relevant_memory = relevant_memory[:-1]
            relevant_memory_tokens = sum(
                [self.token_counter(doc) for doc in relevant_memory]
            )
        content_format = (
            f"This reminds you of these events "
            f"from your past:\n{relevant_memory}\n\n"
        )
        memory_message = SystemMessage(content=content_format)
        used_tokens += self.token_counter(memory_message.content)
        historical_messages: List[BaseMessage] = []
        for message in previous_messages[-10:][::-1]:
            message_tokens = self.token_counter(message.content)
            if used_tokens + message_tokens > self.send_token_limit - 1000:
                break
            historical_messages = [message] + historical_messages
            used_tokens += message_tokens
        input_message = HumanMessage(content=kwargs["user_input"])
        messages: List[BaseMessage] = [base_prompt, memory_message]
        messages += historical_messages
        messages.append(input_message)
        return messages
