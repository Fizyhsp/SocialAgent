import json
from typing import Any, List

from langchain import LLMChain, PromptTemplate
from langchain.memory import ChatMessageHistory
from langchain.schema import Document
from langchain_experimental.autonomous_agents.autogpt.output_parser import AutoGPTOutputParser

from prompt import AgentPrompt
import utils
from actions import Action
from envs import BasicEnvironment
import random
from llm import *
from templates.templates import LLM_agent_template
class BasicAgent:
    """Base class for the agent"""

    def __init__(self, name: str, env: BasicEnvironment,  **kwargs: Any) -> None:
        self.name = name
        self.env = env
        for key, value in kwargs.items():
            setattr(self, key, value)

    def next_action(
            self, *args: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def plan(
            self, *args: Any
    ) -> List[Action]:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
            self, *args
    ) -> None:
        raise NotImplementedError


class RandomAgent(BasicAgent):
    def __init__(self, name: str, env: BasicEnvironment, **kwargs: Any) -> None:
        super().__init__(name, env, **kwargs)

    def next_action(
            self
    ) -> Action:
        action = random.choice(list(self.env.action_space.values()))
        return action

    def plan(
            self, steps=3
    ) -> List[Action]:
        actions = [self.next_action() for _ in range(steps)]
        return actions

    def reset(
            self, *args
    ) -> None:
        pass


class LLMAgent(BasicAgent):

    def __init__(self, name: str, env: BasicEnvironment, memory, base_prompt="",
                 api_base="https://api.chatanywhere.cn/v1",
                 api_key="sk-kJdjn5gp7aCpofbFQJdqL4GKE153kc13URPePANT0OU1Ukft", model_name='gpt-3.5-turbo-0613', temperature=0.0,
                 tasks=("Do the action plan based on your personalty and observation", ), profile=None,
                 short_memory_window_size=2,
                 **kwargs: Any) -> None:
        super().__init__(name, env, **kwargs)
        self.base_prompt = base_prompt
        self.llm = ChatOpenAI(temperature=temperature, openai_api_base=api_base,
                              openai_api_key=api_key, model_name=model_name)
        self.memory = memory
        self.tasks = tasks
        self.profile = profile or dict()

        self.agent_info = {
            "Name": name,
            "Profiles": self.profile
        }
        self.short_memory_window_size = short_memory_window_size
        self.actions_info = env.actions_info

        self.chat_history_memory = ChatMessageHistory()
        self.user_input = (
            "Thinking and determine which action to use, "
            "and respond using the format specified above:"
        )
        self.prompt = AgentPrompt(agent_info=self.agent_info, base_prompt=base_prompt,
                                  actions_info=self.actions_info,
                                  input_variables=["messages", "tasks", "user_input", "memory", "reflection"],
                                  token_counter=self.llm.get_num_tokens,)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.output_parser = AutoGPTOutputParser()

    def next_action(
            self, observation, reflection=None, *args: Any
    ) -> Action:

        if len(self.chat_history_memory.messages) > 0:
            memory_to_add = (
                f"Assistant Reply: {self.chat_history_memory.messages[-1].content} " f"\nResult: {observation} "
            )
            self.memory.add_documents([Document(page_content=memory_to_add)])
        self.chat_history_memory.add_message(SystemMessage(content='{}'.format(observation)))

        assistant_reply = self.chain.run(
            memory=self.memory,
            tasks=self.tasks,
            messages=self.chat_history_memory.messages,
            user_input=self.user_input,
            reflection=reflection,
        )
        self.chat_history_memory.add_message(HumanMessage(content=self.user_input))
        self.chat_history_memory.add_message(AIMessage(content=assistant_reply))

        try:
            assistant_reply = assistant_reply.replace("\r", "")
            assistant_reply = assistant_reply.replace("\n", "")
            action_return = json.loads(assistant_reply)
            action = self.env.action_space.get(action_return['action']['name'])
            action.action_args = action_return['action']['args']
        except Exception as e:
            print(e)
            action = self.env.action_space.get('stop')
        if action is None:
            action = self.env.action_space.get('stop')
        action.act_time = utils.create_now_time()
        return action

    def plan(
            self, observation, steps=3
    ) -> List[Action]:
        actions = [self.next_action(observation) for _ in range(steps)]
        return actions

    def reset(
            self, *args
    ) -> None:
        self.chat_history_memory.clear()

    @property
    def agent_trajectory(self):
        trajectory = [message for message in self.chat_history_memory.messages if message.type != 'human']
        return trajectory


class RedditAgent(LLMAgent):
    def __init__(self, name: str, env: BasicEnvironment, memory, base_prompt="",
                 api_base="https://api.chatanywhere.cn/v1",
                 api_key="sk-kJdjn5gp7aCpofbFQJdqL4GKE153kc13URPePANT0OU1Ukft", model_name='gpt-3.5-turbo-0613', temperature=0.0,
                 tasks=("Do the action plan based on your personalty and observation",), profile="",
                 short_memory_window_size=2,
                 client_id=None, client_secret=None, user_agent=None, password=None):
        super().__init__(name, env, memory, base_prompt, api_base, api_key, model_name, temperature, tasks, profile,
                         short_memory_window_size)
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.password = password