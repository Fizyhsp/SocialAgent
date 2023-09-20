from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper, GoogleSerperAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

ACTION_TYPES = ["COMMON", "WEB", "SOCIAL_NETWORK"]


class Action:
    action_type: str
    name: str
    description: str
    action_args: dict
    enable: bool = True
    thinking: str
    act_time: str

    @classmethod
    def func(cls, action_args, env, agent):
        raise NotImplementedError


""" COMMON ACTIONS """


class StopAction(Action):
    action_type = "COMMON"
    name = "stop"
    description = """When you think the task is completed, use this action"""
    action_args = {}

class CalculateAction(Action):
    action_type = "COMMON"
    name = "calculate"
    description = """Perform mathematical computations accurately using the "Question" parameter, providing precise results for various calculations, from basic operations to complex equations."""
    action_args = {"question": "2 * 3 + 4"}

    wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid='LH9954-7RTAHAYRHX')

    @classmethod
    def func(cls, action_args, env, agent):
        question = action_args.get('question')
        if question is None or len(question) <= 0:
            return "No Result"
        result = cls.wolfram.run(question)
        return "Calculate Result:\n" + result


""" WEB ACTIONS """
class WikiAction(Action):
    action_type = "WEB"
    name = "wiki"
    description = """Swiftly retrieve comprehensive information from wiki-based platforms using the "Item" parameter, streamlining access to reliable details about diverse topics for efficient knowledge acquisition."""
    action_args = {"item": "Bob Dylan"}

    enable = False

    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    @classmethod
    def func(cls, action_args, env, agent):
        item = action_args.get('item')
        if item is None or len(item) <= 0:
            return "No Result"
        result = cls.wikipedia_tool.run(item)
        return "Wikipedia Page Content:\n" + result


class SearchAction(Action):
    action_type = "WEB"
    name = "search"
    description = """Initiate searches on Google using the "Search Query" parameter, swiftly retrieving information, articles, and resources from the broader internet based on indicated keywords, expanding access to diverse information sources."""
    action_args = {"search_query": "Bob Dylan"}
    search_tool = GoogleSerperAPIWrapper(serper_api_key='dbc9cde263f7b6dd95a53eb184e4d46fc0834c41')

    @classmethod
    def func(cls, action_args, env, agent):
        search_query = action_args.get('search_query')
        if search_query is None or len(search_query) <= 0:
            return "No Result"
        result = cls.search_tool.run(search_query)
        return "Google Search Content:\n" + result


class SearchImageAction(Action):
    action_type = "WEB"
    name = "search_image"
    description = "Useful for when "
    action_args = {"search_query": str}

    enable = False


class SearchGifAction(Action):
    action_type = "WEB"
    name = "search_gif"
    description = "Useful for when "
    action_args = {"search_query": str}

    enable = False


""" SOCIAL_NETWORK ACTIONS """
class BrowseRecommendFeed(Action):
    action_type = "SOCIAL_NETWORK"
    name = "browse_recommend_feed"
    description = "Explore personalized feed with recommended content which you will interested in for engaging forum experience."
    action_args = {}


class BrowseHots(Action):
    action_type = "SOCIAL_NETWORK"
    name = "browse_hots"
    description = "Explore most hot post in the forum."
    action_args = {}


class BrowseRising(Action):
    action_type = "SOCIAL_NETWORK"
    name = "browse_hots"
    description = "Explore most quickly rising post in the forum."
    action_args = {}


class BrowseControversial(Action):
    action_type = "SOCIAL_NETWORK"
    name = "browse_hots"
    description = "Explore the most controversial post in the forum."
    action_args = {}


class BrowseHotCommentOfPost(Action):
    action_type = "SOCIAL_NETWORK"
    name = "browse_hot_comment_of_post"
    description = "Enables your to discover and engage with the most popular and engaging comments on a specific post, enhancing your forum experience by facilitating participation in high-quality discussions."
    action_args = {}


class SearchUser(Action):
    action_type = "SOCIAL_NETWORK"
    name = "search_user"
    description = "Find and access member profiles by inputting desired username, fostering connections."
    action_args = {"username": "Elon"}


class SearchPost(Action):
    action_type = "SOCIAL_NETWORK"
    name = "search_post"
    description = "Discover posts aligned with interests through efficient content retrieval."
    action_args = {"keyword": "Elon Musk"}


class PostAction(Action):
    action_type = "SOCIAL_NETWORK"
    name = "post"
    description = "Initiate conversations by sharing content like thoughts, images, and links."
    action_args = {"content": "Nice to see you!"}


class CommentAction(Action):
    action_type = "SOCIAL_NETWORK"
    name = "comment"
    description = "After browse a post or comment, you engage in discussions by adding comments to it, fostering interactions."
    action_args = {"content": "Nice to see you!"}


class ForwardAction(Action):
    action_type = "SOCIAL_NETWORK"
    name = "forward"
    description = "After browse a post or comment, redistribute content to enhance visibility across diverse sections."
    action_args = {}


class LikeAction(Action):
    action_type = "SOCIAL_NETWORK"
    name = "like"
    description = "After browse a post or comment, Show approval for content, contributing to engagement metrics."
    action_args = {}


class DisLikeAction(Action):
    action_type = "SOCIAL_NETWORK"
    name = "dislike"
    description = "After browse a post or comment, Show disapproval for content"
    action_args = {}


class FollowAction(Action):
    action_type = "SOCIAL_NETWORK"
    name = "follow"
    description = "After search a user, follow him and receive updates for ongoing engagement."
    action_args = {"username": "Elon Musk"}


class FollowTopic(Action):
    action_type = "SOCIAL_NETWORK"
    name = "follow_topic"
    description = "Useful when you want to follow a topic, Add personalize experience by following topic keywords, staying informed."
    action_args = {"topic": "Science"}


if __name__ == '__main__':
    Action()