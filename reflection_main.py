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


setting_name = "personality_reflection_gpt3516k"
openai_api_key = "sk-kdF0OEQjOwTyve2HI9g4JGCp2jpFZaud21aJxLWYHpFUxcbS"
platform = 'reddit'
env = RedditEnvironment()
embeddings_model = OpenAIEmbeddings(openai_api_base="https://api.chatanywhere.cn/v1",
                                    openai_api_key=openai_api_key)
embedding_size = 1536
with open('templates/reflection_agent.txt', 'r') as f:
    base_prompt = f.read()
with open('tasks/{}_tasks.json'.format(platform)) as f:
    tasks = json.load(f)
with open('data/personality_info.json'.format(platform)) as f:
    personality_infos = json.load(f)
# user_names = os.listdir("toy_dir/{}/characters".format(platform))
user_names = ["2Years2Go", "Accomplished-Heat-59"]
max_iter = 10
# print(profile)
# print(user_name + ' ' + profile['personality'] + ' ' + str(len(past_trajectory)))
personality_info = 'INTP: {} \n ENFP: {}'.format(personality_infos['INTP'], personality_infos['ENFP'])
if os.path.exists("data/agent_result/{}/{}.json".format(platform, setting_name)):
    with open("data/agent_result/{}/{}.json".format(platform, setting_name), 'r', encoding='utf-8') as f:
        res = json.load(f)
# if os.path.exists("data/agent_result/{}/{}.json".format(platform, "baseline_gpt3516k")):
#     with open("data/agent_result/{}/{}.json".format(platform, "baseline_gpt3516k"), 'r') as f:
#         res = json.load(f)
else:
    res = dict()
    for task_name in tasks.keys():
        res[task_name] = {user_name: {} for user_name in user_names}

with open("data/agent_result/{}/{}.json".format(platform, "reflection_baseline"), 'r', encoding='utf-8') as f:
    res_e = json.load(f)

temperature = 0.2
evaluator = evaluate.LLMAgentEvaluator("gpt-4", temperature, api_key=openai_api_key)
for task_name, task in tqdm.tqdm(tasks.items()):
    for user_name in tqdm.tqdm(user_names):
        another_username = user_names[0] if user_name != user_names[0] else user_names[0]
        with open("toy_dir/reddit/characters/{}/profile.json".format(user_name), 'r', encoding='utf-8') as f:
            profile = json.load(f)
        with open("toy_dir/reddit/characters/{}/profile.json".format(another_username), 'r', encoding='utf-8') as f:
            anti_profile = json.load(f)
        with open("toy_dir/reddit/characters/{}/actions.json".format(user_name), 'r', encoding='utf-8') as f:
            past_trajectory = json.load(f)
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        agent = RedditAgent(user_name, env, memory=vectorstore.as_retriever(), base_prompt=base_prompt,
                            tasks=(task,), api_key=openai_api_key,
                            profile=profile, short_memory_window_size=4,
                            model_name="gpt-3.5-turbo-0613", temperature=temperature,
                            client_id="mIG3nmMon4AXT7Q2zowbkw", client_secret="vq7SHgD-INoYsyoACRhgI-61g8NX_Q",
                            user_agent="linkseed007-Python", password="lk12114957583")
        # if True:
        if 'agent_trajectory_after_reflection' not in res[task_name][user_name]:
            trajectory = res_e[task_name][user_name]['agent_trajectory']
            reflection = evaluator.reflection(task, env.actions_info, trajectory=trajectory, personality=personality_info)
            # reflection = evaluator.multi_reflection(task, env.actions_info, trajectory=trajectory)
            observation = env.reset(agent)
            for _ in trange(max_iter):
                action = agent.next_action(observation, reflection)
                if action.name == 'stop':
                    break
                observation = env.step(action, agent)

            res[task_name][user_name]['reflection'] = reflection
            res[task_name][user_name]['agent_trajectory_after_reflection'] = ["Observation: {}".format(
                message.content) if message.type == 'system' else "Agent Thought and Action: {}".format(message.content) for
                                                             message in agent.agent_trajectory]
            with open("data/agent_result/{}/{}.json".format(platform, setting_name), 'w') as f:
                json.dump(res, f)

        # if True:
        if 'evaluation_after_reflection' not in res[task_name][user_name].keys():
            # model_name = 'gpt-4'
            # model_name = 'gpt-3.5-turbo-0613'
            # model_name = 'gpt-3.5-turbo-16k'
            past_trajectory = ["Action Name: {}; Action Content: {}".format(action['name'],
                                                                            action['action_args']['content']) for action in past_trajectory]
            past_trajectory = past_trajectory[-5:]

            # trajectory = res_e[task_name][user_name]['agent_trajectory']
            # evaluation_result = evaluator._evaluate_agent_trajectory(
            #     task=task,
            #     trajectory=trajectory,
            #     profile=profile,
            #     past_trajectory=past_trajectory
            # )
            # res_e[task_name][user_name]['evaluation'] = evaluation_result
            # with open("data/agent_result/{}/{}.json".format(platform, "reflection_baseline"), 'w') as f:
            #     json.dump(res_e, f)

            trajectory = res[task_name][user_name]['agent_trajectory_after_reflection']
            evaluation_result = evaluator._evaluate_agent_trajectory(
                task=task,
                trajectory=trajectory,
                profile=profile,
                past_trajectory=past_trajectory
            )
            res[task_name][user_name]['evaluation'] = evaluation_result
            res[task_name][user_name]['evaluation_after_reflection'] = evaluation_result
            with open("data/agent_result/{}/{}.json".format(platform, setting_name), 'w') as f:
                json.dump(res, f)
