from jinja2 import Template

LLM_agent_prompt = """
[Task]:
    /** Task **/

[Thoughts Rules]
    1. You should first analyse the [Observation]
    2. You should then analyse the [Observation] and [Task] and [Latest Action] to give a plan

[Action Rules]
    1. You must choose at only one [ActionName] From following [Available Actions].
    2. You must return the [ActionArgs] for the responding [ActionName].
    3. The format of [Thinking] & [ActionPlan] must be a json format such as showed in [Format].



[Character]
    [Your Name]
        /** Name **/
    [Your Profiles]
        /** Profiles **/

[Available Actions]
    {% for action in actions %}
    [ActionName]: /** action['ActionName'] **/, [ActionArgs]: /** action['ActionArgs'] **/
    [ActionDescription]: /** action['Description'] **/
    {% endfor %}
    
[Format]
    {
        "thoughts":
        {
            "observation_analyse": "analyse",
            "connection_of_observation_and_latest_action": "analyse",
            "next_plan": "- short bulleted\n- list that conveys\n- long-term plan",
            "criticism": "constructive self-criticism",
        },
        "action": {
            "name": "action name",
            "args":{
                "arg name": "value"
            }
        }
    }

[Response]
System: This reminds you of these events from your past:
"""

LLM_agent_thinking_prompt = """
[Task]:
    /** Task **/

[Overall Rules to follow]
    1. Act as your are the [Character], you should consider both the [Task] and current [Observation], then give the [Thinking]
    2. It's import to consider the coherence of your new [Thinking] and [Thinking and Action History]

[Character]
    [Your Name]
        /** Name **/
    [Your Profiles]
        /** Profiles **/
        
[Short Memory]
    {% for memory in short_memory %}
    [Thinking and Action history] /** memory **/
    {% endfor %}
    
[Observation]
    /** observation **/

[Thinking]
"""

LLM_agent_action_prompt = """
[Overall Rules to follow]
    1. You must choose at only one [ActionName] From following [Available Actions].
    2. You must return the [ActionArgs] for the responding [ActionName].
    3. The format of [Action Plan] must be a json format such as showed in [Action Plan Example].
    4. Based one the [Thinking] to give the [Action Plan]
    5. Make sure your [ActionName] and content in the [ActionArgs] are diverse from the [Short Memory]

[Action Plan Example]
    {"ActionName": "post", "ActionArgs": {"content": "Nice Day!"}}

[Available Actions]
    {% for action in actions %}
    [ActionName]: /** action['ActionName'] **/, [ActionArgs]: /** action['ActionArgs'] **/
    [ActionDescription]: /** action['Description'] **/
    {% endfor %}

[Thinking]
    /** thinking **/

[Action Plan]
"""

autoGPT_prompt = """
[Task]:
    /** Task **/
    
[Role]
    [Your Name]
        /** Name **/
    [Your Profiles]
        /** Profiles **/

[CONSTRAINTS]
1. You should thinking based on the [Observation] now
2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
3. Exclusively use the tools listed in [Available Actions]

[Available Actions]
    {% for action in actions %}
    [ActionName]: /** action['ActionName'] **/, [ActionArgs]: /** action['ActionArgs'] **/
    [ActionDescription]: /** action['Description'] **/
    {% endfor %}

[PERFORMANCE EVALUATION]
1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

[Observation]
/** observation **/

You should only respond in JSON format as described below
[RESPONSE FORMAT]:
{
    "observation" : "latest observation showed in [Observation]"
    "thoughts":
    {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\n- list that conveys\n- long-term plan",
        "criticism": "constructive self-criticism",
    },
    "action": {
        "name": "action name",
        "args":{
            "arg name": "value"
        }
    }
}


Ensure the response can be parsed by Python json.loads
[RESPONSE]
    {% for memory in short_memory %}
    /** memory **/
    {% endfor %}
"""

"""
Use the following format, the output must contains Action and Action Input:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Args: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)

Question: {input}
{agent_scratchpad}
"""


LLM_agent_template = Template(LLM_agent_prompt, variable_start_string='/**', variable_end_string='**/')
LLM_agent_thinking_template = Template(LLM_agent_thinking_prompt, variable_start_string='/**', variable_end_string='**/')
LLM_agent_action_template = Template(LLM_agent_action_prompt, variable_start_string='/**', variable_end_string='**/')
AutoGPT_template = Template(autoGPT_prompt, variable_start_string='/**', variable_end_string='**/')
# """
# [Short Memory]
#     {% for sm in short_memory %}
#     /** sm **/
#     {% endfor %}
# """
# m = ["2023-08-10 17:46:39 Observation: Google Search Content:\nBeing physically active can improve your brain health, help manage weight, reduce the risk of disease, strengthen bones and muscles, and improve your ability to do everyday activities. Adults who sit less and do any amount of moderate-to-vigorous physical activity gain some health benefits., Thought: I want to search for information about the benefits of exercise, Did Action: search with args: {'search_query': 'benefits of exercise'}", "2023-08-10 17:46:43 Observation: Google Search Content:\nBeing physically active can improve your brain health, help manage weight, reduce the risk of disease, strengthen bones and muscles, and improve your ability to do everyday activities. Adults who sit less and do any amount of moderate-to-vigorous physical activity gain some health benefits., Thought: I want to search for information about the benefits of exercise, Did Action: search with args: {'search_query': 'benefits of exercise'}"]
# st = LLM_agent_template.render(observation="Post the", short_memory=m)
# print(st)