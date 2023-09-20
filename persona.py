import json
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
import os
import tqdm

template = """
Please predict the MBTI personality of the following roles.
You should analyse the character according to the  [Profile] and [Recent Post Content], then give the personality predict
 [Profile]
{{profile}}

[Recent Post Content]
{{contents}}
 
 [FORMAT]
 {"personality:" "personality", "character_analyse": "character_analyse"}
 """
prompt = PromptTemplate(template=template, input_variables=["profile", "contents"], template_format="jinja2")
llm = ChatOpenAI(openai_api_base="https://api.chatanywhere.cn/v1", temperature=0.1,
                 openai_api_key="sk-kJdjn5gp7aCpofbFQJdqL4GKE153kc13URPePANT0OU1Ukft",
                 model_name='gpt-3.5-turbo-0613')
llm_chain = LLMChain(prompt=prompt, llm=llm)

base_path = 'toy_dir/reddit/characters'
names = os.listdir('toy_dir/reddit/characters')

count = 0
for name in tqdm.tqdm(names):
    with open(os.path.join(base_path, name, 'profile.json'), encoding='utf-8') as f:
        profile = json.load(f)
    # if 'personality' in profile.keys():
    #     continue
    #
    # with open(os.path.join(base_path, name, 'actions.json'), encoding='utf-8') as f:
    #     actions = json.load(f)
    # try:
    #     contents = ';'.join([action['action_args']['content'] for action in actions][-30:])[:2000]
    #     response = llm_chain.predict(profile=profile, contents=contents)
    #     res = json.loads(response)
    # except Exception as e:
    #     print(name, e)
    #     contents = ';'.join([action['action_args']['content'] for action in actions][-5:])[:1000]
    #     response = llm_chain.predict(profile=profile, contents=contents)
    #     res = json.loads(response)
    # profile.update(res)
    with open(os.path.join(base_path, name, 'profile.json'), 'w') as f:
        json.dump(profile, f)
