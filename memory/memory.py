import os
import json
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel
from copy import deepcopy

class AgentMemory(BaseModel):
    character: str
    level: str
    llm: ChatOpenAI
    history_length: int = 10
    authorization: str = None

    reflection_questions: List[str] = ["What content do you most disagree with? Why?",
                                       "What content do you most agree with? Why?",
                                       "What are the latest information that interest you?",
                                       "What do you think of Taiwan's Tsai Ing-wen administration"
                                       "What do you think of the Chinese government's Taiwan policy"]

    @property
    def path(self):
        return 'templates/characters/level{}/{}'.format(self.level, self.character)

    @property
    def history(self):
        with open(os.path.join(self.path, 'history.json'), 'r', encoding='utf-8') as f:
            history = json.load(f)
            return history


    @property
    def profile(self):
        with open(os.path.join(self.path, 'backstory.txt'), 'r', encoding='utf-8') as f:
            character_description = f.read()
        return character_description

    @property
    def tone(self):
        with open(os.path.join(self.path, 'tone.txt'), 'r', encoding='utf-8') as f:
            tones = f.read()
        return tones

    @property
    def posts(self):
        with open(os.path.join(self.path, 'posts.txt'), 'r', encoding='utf-8') as f:
            history_posts = f.read()
        return history_posts

    @property
    def opinions(self):
        with open(os.path.join(self.path, 'opinions.txt'), 'r', encoding='utf-8') as f:
            opinion_content = f.read().splitlines()
        return opinion_content

    def record(self, content):
        history = deepcopy(self.history)
        history.append(content)
        with open(os.path.join(self.path, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f)

    def rewrite(self, text):
        # message = [
        #     SystemMessage(content='I want to act the character {}, Overall, his writing style is {}.'
        #                           ' There are some post he made before: {}'.format(character_name, tone, posts)),
        #     HumanMessage(content="Now please rewrite the tweet in {}'s style: ".format(character_name) + text),
        # ]

        message = [
            HumanMessage(content="Instruct: Rewrite a tweet in the given writing style and language:"
                                 "\n {} \n"
                                 "Tweet: {}"
                                 "Rewrite Tweet:".format(self.tone, text)),
        ]

        res = self.llm(message).content
        return res

    def reflection(self, observation):
        prompt = "You named {},  There are some information about yourself: {}" \
                 "You have some opinions before {}.".format(self.character,
                  self.profile, self.opinions[-self.history_length // 2:])
        message = [
            HumanMessage(content=prompt + "Now please play the role above to discuss your views on the following content according to your background and opinions before, don't answer with 'As a language model', answer should be less than 50 words: {}".format(observation)),
        ]
        res = self.llm(message).content
        opinions = deepcopy(self.opinions)
        opinions.append(res)

        with open(os.path.join(self.path, 'opinions.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(opinions))

    def interview(self, question):
        prompt = "You named {}, there are some information about yourself: {}\n you have done these in twitter recently:" \
                 " {} \n You have some opinions before {}.\n Now please play the role above to answer the question.".format(self.character, self.profile, self.history[-self.history_length:], self.opinions[-self.history_length:])
        message = [
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ]
        res = self.llm(message)
        return res.content



if __name__ == '__main__':
    llm = ChatOpenAI(temperature=0, openai_api_key='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')
    agent = AgentMemory(character='hawke2210', level=2, llm=llm)
    res = agent.interview("Who are you, introduce yourself")
    print(res)
