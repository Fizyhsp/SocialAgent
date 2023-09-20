import os
from datetime import datetime

from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

time_str_format = "%Y-%m-%d %H:%M:%S"




def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def create_now_time():
    current_time = datetime.now()
    current_time_str = current_time.strftime(time_str_format)
    return current_time_str

def build_short_memory(observation, action):
    memory = {
        # "ActionTime": action.act_time,
        "Thingking": action.thinking,
        "Observation": observation,
        "ActionName": action.name,
        "ActionArgs": action.action_args
    }
    # memory_string = "{} Observation: {}, Thought: {}, Did Action: {} with args: {}".format(action.act_time,
    #                                                                                        observation, action.thinking,
    #                                                                                        action.name, action.action_args)
    return memory


def format_llm_response(content):
    if content[0] != '{':
        content = '{' + content
    if content[-2:] != '}}':
        if content[-1] == '}':
            content = content + '}'
        else:
            content = content + '}}'
    return content


### Data Process Utils

