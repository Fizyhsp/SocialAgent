import openai
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
class Chatgpt:
    def __init__(self, api_base, api_key, model_name='gpt-3.5-turbo', temperature=0):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        openai.api_base = api_base
        openai.api_key = api_key
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name, openai_api_base=api_base, openai_api_key=api_key)

    def generate(self, user_prompt, system_prompt=""):
        message = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        result = self.llm(message).content
        # completion = openai.ChatCompletion.create(model=self.model_name,
        #                                           messages=[{"role": "system", "content": system_prompt},
        #                                                     {"role": "user", "content": user_prompt}])
        # result = completion.choices[0].message.content
        return result