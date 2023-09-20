import json

from langchain import PromptTemplate
from langchain.evaluation import load_evaluator
import os
import subprocess

from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentType, initialize_agent

from pydantic import HttpUrl
from urllib.parse import urlparse

from typing import Any, Optional, Sequence, Tuple
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema import AgentAction
from langchain.evaluation import AgentTrajectoryEvaluator
from tqdm import trange


class LLMAgentEvaluator(AgentTrajectoryEvaluator):
    """Evaluate the perplexity of a predicted string."""

    def __init__(self, model_name, temperature, api_base="https://api.chatanywhere.cn/v1",
                 api_key="sk-C566fFClzgn8aWYbaqhCQgu1oJzFlvM90ftVNFtnjnrXq1VT") -> None:

        self.llm = ChatOpenAI(openai_api_base=api_base, temperature=temperature,
                              openai_api_key=api_key,
                              model_name=model_name)
        with open('templates/evaluate.txt', 'r') as f:
            template = f.read()
        prompt = PromptTemplate(template=template, input_variables=["task", "trajectory", "profile", "past_trajectory"],
                                template_format="jinja2")
        self.evaluation_chain = LLMChain(prompt=prompt, llm=self.llm)

        with open("templates/persona_reflection.txt", 'r') as f:
            reflection_template = f.read()
        reflection_prompt = PromptTemplate(template=reflection_template, input_variables=["task", "actions",
                                                                                          "trajectory", "personality"],
                                           template_format="jinja2")
        self.reflection_chain = LLMChain(prompt=reflection_prompt, llm=self.llm)

    def _evaluate_agent_trajectory(
        self,
        task,
        trajectory,
        profile,
        past_trajectory,
        **kwargs: Any,
    ) -> dict:
        try:
            response = self.evaluation_chain.run(dict(trajectory=trajectory, task=task,
                                                 profile=profile, past_trajectory=past_trajectory), **kwargs)
        except Exception as e:
            try:
                print(e)
                trajectory = trajectory[:int(len(trajectory) * 0.5)]
                past_trajectory = past_trajectory[:int(len(past_trajectory) * 0.5)]
                response = self.evaluation_chain.run(dict(trajectory=trajectory, task=task,
                                                          profile=profile, past_trajectory=past_trajectory), **kwargs)
            except Exception as e:
                print(e)
                trajectory = trajectory[:int(len(trajectory) * 0.2)]
                past_trajectory = past_trajectory[:int(len(past_trajectory) * 0.2)]
                response = self.evaluation_chain.run(dict(trajectory=trajectory, task=task,
                                                          profile=profile, past_trajectory=past_trajectory), **kwargs)
        try:
            res = json.loads(response)
        except Exception as e:
            print(e)
            res = self.reformat(response)
        return res

    def reflection(self, task, actions, trajectory, personality=None):
        response = self.reflection_chain.predict(task=task, actions=actions,
                                                 trajectory=trajectory, personality=personality)
        try:
            res = json.loads(response)
        except Exception as e:
            print(e)
            res = self.reformat(response)
        return res

    def multi_reflection(self, task, actions, trajectory, reflection_nums=2, reflection_combine=False):
        reflections = [self.reflection(task, actions, trajectory) for _ in trange(reflection_nums)]
        if not reflection_combine:
            return reflections
        else:
            return reflections

    def reformat(self, input):
        template = """
            Please reformat the [Input] to follow json [Format]:

            [Input]
            {{ input }}

            [Format]
            {
            "task_achieving":
            {
                "i": {"score": "score", "reason": "reason},
                "ii": {"score": "score", "reason": "reason},
                ...
            },
            "simulation": {
                "i": {"score": "score", "reason": "reason},
                "ii": {"score": "score", "reason": "reason},
                ...
            }

            [Reformat]
        }
        """
        prompt = PromptTemplate(template=template, input_variables=["input"], template_format="jinja2")
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        response = llm_chain.predict(input=input)
        json_res = json.loads(response)
        return json_res