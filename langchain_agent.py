from langchain import GoogleSerperAPIWrapper
from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI
import openai
from envs import RedditEnvironment

# def create_tool_from_action(action):
#     tool = Tool(
#         name=action.name,
#         description=action.description,
#         func=action.func
#     )
# RedditEnvironment.BrowseHots.description


openai.api_base = "https://api.chatanywhere.cn/v1"
openai.api_key = "sk-bvFurWdnhTmrbCsGY26FdAmEmRc2LqiWXuEzwBEqYu7jv7Jz"

search = GoogleSerperAPIWrapper(serper_api_key='dbc9cde263f7b6dd95a53eb184e4d46fc0834c41')
tools = [
    Tool(
        name="search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    WriteFileTool(),
    ReadFileTool(),
]

# Initialize the vectorstore as empty
import faiss
embeddings_model = OpenAIEmbeddings(openai_api_base="https://api.chatanywhere.cn/v1",
                                    openai_api_key="sk-bvFurWdnhTmrbCsGY26FdAmEmRc2LqiWXuEzwBEqYu7jv7Jz")
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0, openai_api_base="https://api.chatanywhere.cn/v1",
                   openai_api_key="sk-bvFurWdnhTmrbCsGY26FdAmEmRc2LqiWXuEzwBEqYu7jv7Jz"),
    memory=vectorstore.as_retriever(),
)
# Set verbose to be true
agent.chain.verbose = True

agent.run(["write a weather report for SF today"])