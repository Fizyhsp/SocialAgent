import os
import time

from langchain import FAISS, InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from tqdm import trange

from envs import RedditEnvironment
import faiss
import tqdm
from agents import RedditAgent
import json
import random
import evaluate


setting_name = "reflection_gpt3516k"
openai_api_key = "sk-C566fFClzgn8aWYbaqhCQgu1oJzFlvM90ftVNFtnjnrXq1VT"
platform = 'reddit'
env = RedditEnvironment()
env.read_only = False
embeddings_model = OpenAIEmbeddings(openai_api_base="https://api.chatanywhere.cn/v1",
                                    openai_api_key=openai_api_key)

embedding_size = 1536
with open('templates/reflection_agent.txt', 'r') as f:
    base_prompt = f.read()
with open('tasks/{}_tasks.json.bak'.format(platform)) as f:
    tasks = json.load(f)
task = tasks["Task 7"]
user_name = "2Years2Go"
max_iter = 10
with open("toy_dir/reddit/characters/{}/profile.json".format(user_name), 'r', encoding='utf-8') as f:
    profile = json.load(f)
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
# Baseline Setting
agent = RedditAgent(user_name, env, memory=vectorstore.as_retriever(), base_prompt=base_prompt,
                    tasks=(task,), api_key=openai_api_key,
                    profile=profile, short_memory_window_size=4,
                    model_name="gpt-4", temperature=0.2,
                    client_id="mIG3nmMon4AXT7Q2zowbkw", client_secret="vq7SHgD-INoYsyoACRhgI-61g8NX_Q",
                    user_agent="linkseed007-Python", password="lk12114957583")
reflection = None
observation = env.reset(agent)
res = dict()
for _ in trange(max_iter):
    action = agent.next_action(observation, reflection)
    print(action)

    observation = env.step(action, agent)
res['agent_trajectory'] = ["Observation: {}".format(
            message.content) if message.type == 'system' else "Agent Thought and Action: {}".format(message.content) for
                                message in agent.agent_trajectory]
with open("toy_dir/reddit/characters/{}/action_logs.json".format(user_name), 'w') as f:
    json.dump(res, f)