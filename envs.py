import random
import string
import time

import praw
import requests
from gymnasium import Env
from typing import Any
import os
import utils
import pandas as pd
from actions import *
from itertools import chain
import warnings

# 在这里放置你的代码
warnings.filterwarnings("ignore", message="The frame.append method is deprecated and will be removed")
os.environ['WOLFRAM_ALPHA_APPID'] = 'LH9954-7RTAHAYRHX'
os.environ['SERPER_API_KEY'] = "dbc9cde263f7b6dd95a53eb184e4d46fc0834c41"

StringObservation = str


class BasicEnvironment:
    action_space = {action.name: action for action in chain([])}
    actions_info = [{"ActionName": n, "ActionArgs": a.action_args.keys(), "Description": a.description} for n, a in
                    action_space.items()]
    def step(
            self, action: Action, agent=None
    ) -> tuple[StringObservation, float, bool, bool, dict[str, Any]]:
        return action.func(action.action_args, self, agent)
    reset_action = None


class WebAPIEnvironment(BasicEnvironment):
    action_space = {action.name: action for action in chain(BasicEnvironment.action_space.values(), [])}
    actions_info = [{"ActionName": n, "ActionArgs": a.action_args.keys(), "Description": a.description} for n, a in action_space.items()]
    reset_action = None


class ToySocialNetworkEnvironment(WebAPIEnvironment):

    def __init__(self, toy_dir):
        self.feed_posts = {}
        self.feed_num = 1
        self.search_num = 1
        self.toy_dir = toy_dir

        self.env_posts = pd.read_csv(os.path.join(toy_dir, 'env_posts.csv'))
        self.env_users = pd.read_csv(os.path.join(toy_dir, 'env_users.csv'))

    def reset(
            self, agent
    ):
        utils.create_directory_if_not_exists(os.path.join(self.toy_dir, 'agent_logs', agent.name))
        assert agent.name in self.env_posts['from_user'].tolist()
        self.feed_posts = {agent.name: self.env_posts[self.env_posts['from_user'] == agent.name]}
        init_observation = self.BrowseRecommendFeed.func({}, self, agent)
        return init_observation

    class BrowseRecommendFeed(BrowseRecommendFeed):
        description = """Meaning: Going through personalized content suggestions based on the your history and interests.
Execution: Engage with this when wanting to discover content tailored to the your preferences.
        """

        @classmethod
        def func(cls, action_args, env, agent):
            feed_posts = env.feed_posts.get(agent.name)
            if feed_posts is None:
                return ""
            read_rows = feed_posts.sample(n=env.feed_num)
            feed_posts.drop(read_rows.index, inplace=True)

            result = []
            for i, row in read_rows.iterrows():
                result.append("Feed Article id {}: {}".format(row['id'], row['text'][:600]))
            return '\n\n'.join(result)

    class SearchPost(SearchPost):
        description = """Meaning: Looking up specific topics, posts, or comments using keywords.
Execution: Use when you're seeking specific information or topics not readily available in your feed.
                """

        @classmethod
        def func(cls, action_args, env, agent):
            search_res = env.env_posts[
                env.env_posts['text'].str.contains(action_args.get("keyword"), case=False)].head(env.search_num)
            result = []
            for i, row in search_res.iterrows():
                result.append("Find Article {}: {}".format(row['id'], row['text']))
            return '\n\n'.join(result)

    class SearchUser(SearchUser):

        @classmethod
        def func(cls, action_args, env, agent):
            search_res = env.env_users[
                env.env_users['username'].str.contains(action_args.get("username"), case=False)].head(env.search_num)
            result = []
            for i, row in search_res.iterrows():
                result.append("Find User {}".format(row['username']))
            return '\n\n'.join(result)

    class PostAction(PostAction):
        description = """Meaning: Sharing original content or links to the Reddit platform in the form of text, image, video, or a link.
Execution: Ensure content is relevant, original, and adheres to the guidelines of the specific subreddit.
                        """

        @classmethod
        def func(cls, action_args, env, agent):
            post_text_path = os.path.join(env.toy_dir, 'agent_logs', agent.name, 'posts.csv')
            if os.path.exists(post_text_path):
                df = pd.read_csv(post_text_path)
            else:
                df = pd.DataFrame({"content": [], "act_time": []})
            action_args['act_time'] = utils.create_now_time()
            df = df.append(action_args, ignore_index=True)
            df.to_csv(post_text_path, index=False)
            return "Post Success!"

    class CommentAction(CommentAction):

        description = """Meaning: Responding or adding input to a post or another comment.
Execution: Comment when you have valuable insights, questions, or responses to contribute. """

        @classmethod
        def func(cls, action_args, env, agent):

            comment_text_path = os.path.join(env.toy_dir, 'agent_logs', agent.name, 'comments.csv')
            if os.path.exists(comment_text_path):
                df = pd.read_csv(comment_text_path)
            else:
                df = pd.DataFrame({"content": [], "root": [], "act_time": []})
            action_args['act_time'] = utils.create_now_time()
            df = df.append(action_args, ignore_index=True)
            df.to_csv(comment_text_path, index=False)
            return "Comment Article-{} Success !".format(action_args.get("root"))

    class FollowAction(FollowAction):
        @classmethod
        def func(cls, action_args, env, agent):
            follow_text_path = os.path.join(env.toy_dir, 'agent_logs', agent.name, 'follows.csv')
            if os.path.exists(follow_text_path):
                df = pd.read_csv(follow_text_path)
            else:
                df = pd.DataFrame({"username": [], "act_time": []})
            action_args['act_time'] = utils.create_now_time()
            df = df.append(action_args, ignore_index=True)
            df.to_csv(follow_text_path, index=False)
            return "Follow User-{} Success !".format(action_args.get("username"))

    class LikeAction(LikeAction):
        description = """Meaning: Expressing approval or appreciation for a particular post or comment.
Execution: Upvote when you find content insightful, valuable, or agreeable."""

        @classmethod
        def func(cls, action_args, env, agent):
            like_text_path = os.path.join(env.toy_dir, 'agent_logs', agent.name, 'likes.csv')
            if os.path.exists(like_text_path):
                df = pd.read_csv(like_text_path)
            else:
                df = pd.DataFrame({"root": [], "act_time": []})
            action_args['act_time'] = utils.create_now_time()
            df = df.append(action_args, ignore_index=True)
            df.to_csv(like_text_path, index=False)
            return "Like Article-{} Success !".format(action_args.get("root"))

    class ForwardAction(ForwardAction):
        @classmethod
        def func(cls, action_args, env, agent):
            forward_text_path = os.path.join(env.toy_dir, 'agent_logs', agent.name, 'forward.csv')
            if os.path.exists(forward_text_path):
                df = pd.read_csv(forward_text_path)
            else:
                df = pd.DataFrame({"root": [], "act_time": []})
            action_args['act_time'] = utils.create_now_time()
            df = df.append(action_args, ignore_index=True)
            df.to_csv(forward_text_path, index=False)
            return "Forward Article-{} Success!".format(action_args.get("root"))

    action_space = {action.name: action for action in
                    chain(WebAPIEnvironment.action_space.values(), [
                        PostAction, CommentAction, ForwardAction, SearchPost,
                        BrowseRecommendFeed, SearchUser, FollowAction, LikeAction
    ])}
    actions_info = [{"ActionName": n, "ActionArgs": a.action_args.keys(), "Description": a.description} for n, a in
                    action_space.items()]
    reset_action = BrowseRecommendFeed

    def step(self, action: Action, agent=None):
        assert agent is not None
        if not action.enable:
            action_res = ''
            observation = "Can't do this action now, Please choose another one."
        else:
            action_res = action.func(action.action_args, self, agent)
            observation = action_res
        return observation


class RedditEnvironment(WebAPIEnvironment):

    reddit: praw.Reddit
    feed_num: int = 1
    search_num: int = 1
    read_only: bool = True
    recent_read_ids: list[str] = []
    read_subreddit: str = 'shitposting'
    records: set[str] = set()

    def reset(
            self, agent, read_only=True
    ):
        self.reddit = praw.Reddit(
            client_id=agent.client_id,
            client_secret=agent.client_secret,
            user_agent=agent.user_agent,
            username="Ok-Opening4548",
            password=agent.password,
        )
        self.read_only = read_only

        # init_observation = self.BrowseRecommendFeed.func({}, self, agent)
        init_observation = "Start Your Task!"
        return init_observation

    def get_root(self, id):
        try:
            root = praw.models.Submission(self.reddit, id)
        except Exception as e:
            root = praw.models.Comment(self.reddit, id)
        return root

    @staticmethod
    def generate_random_id(length):
        characters = string.ascii_letters + string.digits
        random_id = ''.join(random.choice(characters) for _ in range(length))
        return random_id

    class BrowseRecommendFeed(BrowseRecommendFeed):
        description = """Meaning: Going through personalized content suggestions based on the your history and interests.
        Execution: Engage with this when wanting to discover content tailored to the your preferences."""
        @classmethod
        def func(cls, action_args, env, agent):
            result = []
            hots = env.reddit.front.hot(limit=env.feed_num * 5)
            news = env.reddit.front.new(limit=env.feed_num * 5)
            risings = env.reddit.front.rising(limit=env.feed_num * 5)

            feed_posts = list(hots) + list(news) + list(risings)
            read_submissions = random.choices(feed_posts, k=env.feed_num)
            for submission in read_submissions:
                if submission.id in env.records:
                    continue
                res = {
                    "id": submission.id,
                    "title": submission.title,
                    "content": submission.selftext
                }
                result.append(res)
                env.recent_read_ids.append(submission.id)
                env.records.add(submission.id)
            return {"NewFeedArticle": result}

    class BrowseHots(BrowseHots):
        description = """Meaning: Viewing posts that are currently trending or popular in a specific subreddit or Reddit as a whole.
Execution: Browse these posts when aiming to stay updated with the latest popular content."""
        @classmethod
        def func(cls, action_args, env, agent):
            result = []

            for submission in env.reddit.subreddit("all").hot():
                if submission.id in env.records:
                    continue
                res = {
                    "id": submission.id,
                    "title": submission.title,
                    "content": submission.selftext
                }
                result.append(res)
                env.recent_read_ids.append(submission.id)
                env.records.add(submission.id)
                if len(result) >= env.search_num:
                    break
            return {"Hot Article": result}

    class BrowseRisings(BrowseRising):
        description = """Meaning: Checking out posts that are quickly gaining traction and popularity.
Execution: Engage with this to identify potentially viral content early on."""
        @classmethod
        def func(cls, action_args, env, agent):
            result = []

            for submission in env.reddit.subreddit("all").rising():
                if submission.id in env.records:
                    continue
                res = {
                    "id": submission.id,
                    "title": submission.title,
                    "content": submission.selftext
                }
                result.append(res)
                env.recent_read_ids.append(submission.id)
                env.records.add(submission.id)
                if len(result) >= env.search_num:
                    break
            return {"Rising Article": result}

    class BrowseControversial(BrowseControversial):
        description = """Meaning: Exploring posts that have received a mix of upvotes and downvotes, indicating varying opinions.
Execution: Engage with these to understand divisive topics or discussions."""

        @classmethod
        def func(cls, action_args, env, agent):
            result = []

            for submission in env.reddit.subreddit("all").controversial():
                if submission.id in env.records:
                    continue
                res = {
                    "id": submission.id,
                    "title": submission.title,
                    "content": submission.selftext
                }
                result.append(res)
                env.recent_read_ids.append(submission.id)
                env.records.add(submission.id)
                if len(result) >= env.search_num:
                    break
            return {"Controversial Article": result}

    class BrowseSubRedditOfSomeTopic(BrowseHots):

        description = "Useful when you want to use it to browse a specific subreddit and gather the information you interested, action args must have topic"
        action_args = {"topic": "topic"}
        @classmethod
        def func(cls, action_args, env, agent):
            result = []

            for submission in env.reddit.subreddit(action_args["topic"]).hot():
                if submission.id in env.records:
                    continue
                res = {
                    "id": submission.id,
                    "title": submission.title,
                    "content": submission.selftext
                }
                result.append(res)
                env.recent_read_ids.append(submission.id)
                env.records.add(submission.id)
                if len(result) >= env.search_num:
                    break
            return {"Hot Article": result}

    class SearchPost(SearchPost):
        description = """Meaning: Looking up specific posts using keywords.
Execution: Use when you're seeking specific information or topics not readily available in your feed."""
        @classmethod
        def func(cls, action_args, env, agent):
            result = []
            try:
                for submission in env.reddit.subreddit("all").search(action_args['keyword']):
                    if submission.id in env.records:
                        continue
                    res = {
                        "id": submission.id,
                        "title": submission.title,
                        "content": submission.selftext
                    }
                    env.read_subreddit = submission.subreddit
                    result.append(res)
                    env.recent_read_ids.append(submission.id)
                    env.records.add(submission.id)
                    if len(result) >= env.search_num:
                        break
            except Exception as e:
                print(e)
            return {"SearchArticleResult": result}

    class BrowseHotCommentOfPost(BrowseHotCommentOfPost):
        description = """Meaning: Reading the most upvoted comments beneath a particular post.
Execution: Do this when wanting to grasp the community's primary reactions or opinions on a post."""

        @classmethod
        def func(cls, action_args, env, agent):
            if len(env.recent_read_ids) == 0:
                return "No more comments for recent read!"
            root = env.get_root(env.recent_read_ids[-1])
            root.comment_sort = "top"
            try:
                comments = root.comments.list()[:10]
                comments = [c for c in comments if c not in env.records]
            except Exception as e:
                nmc = env.recent_read_ids.pop()
                res = "No more comments for {}! ".format(nmc)
                return res

            if len(comments) <= 0:
                nmc = env.recent_read_ids.pop()
                res = "No more comments for {}! ".format(nmc)
            else:
                comment = random.choice(comments)
                res = {
                    "BrowseComment": {
                        "root": env.recent_read_ids[-1],
                        "comment_id": comment.id,
                        "content": comment.body
                    }
                }
                env.recent_read_ids.append(comment.id)
                env.records.add(comment.id)
            return res

    class SearchUser(SearchUser):

        @classmethod
        def func(cls, action_args, env, agent):
            result = []
            for redditor in env.reddit.redditors.search(action_args['keyword']):
                res = {
                    "userid": redditor.id,
                    "username": redditor.name
                }
                result.append(res)
                if len(result) >= env.search_num:
                    break
            return {"SearchUserResult": result}

    class PostAction(PostAction):
        description = """Meaning: Sharing original content or links to the Reddit platform in the form of text, image, video, or a link.
Execution: Ensure content is relevant, original, and adheres to the guidelines of the specific subreddit."""

        action_args = {"subreddit": "your subreddit name", "title": "article title", "content": "article content"}

        @classmethod
        def func(cls, action_args, env, agent):
            # if not env.read_only:
            subreddit = action_args.get('subreddit') or env.read_subreddit
            # submission = env.reddit.subreddit(subreddit).submit(title=action_args['title'], selftext=action_args['content'], url='')
            submission = env.reddit.subreddit(subreddit).submit(title=action_args['title'], url='', selftext=action_args['content'])
            submission_id = submission.id
            # else:
            #     submission_id = env.generate_random_id(7)
            res = {
                "You Just Post Success": {
                    "id": submission_id,
                    "title": action_args['title'],
                    "content": action_args['content']
                }
            }
            res = "You Just Post Success"
            return res

    class CommentAction(CommentAction):
        description = """Meaning: Responding or adding input to a post or another comment.
Execution: Comment when you have valuable insights, questions, or responses to contribute.
        """
        @classmethod
        def func(cls, action_args, env, agent):
            if len(env.recent_read_ids) == 0:
                action_args['title'] = ""
                return env.PostAction.func(action_args, env, agent)
            # if not env.read_only:
            root = env.get_root(env.recent_read_ids[-1])
            comment = root.reply(action_args["content"])
            time.sleep(10)
            comment_id = comment.id
            # else:
            #     comment_id = env.generate_random_id(7)
            res = {
                "You Just Comment Success": {
                    "id": comment_id,
                    "root": env.recent_read_ids[-1],
                    "content": action_args["content"]
                }
            }
            res = "You Just Comment Success"
            return res

    class FollowTopic(FollowTopic):
        description = """Meaning: Exploring individual communities or subreddits focused on particular topics or interests.
Execution: Do this when you're interested in a niche topic or wish to interact with a specific community."""
        @classmethod
        def func(cls, action_args, env, agent):
            if not env.read_only:
                subreddit = env.reddit.subreddit(action_args['topic'])
                subreddit.subscribe()
            res = {
                "You Just Follow the Topic Success": action_args['topic']
            }
            res = "You Just Follow the Topic Success"
            return res

    class LikeAction(LikeAction):
        description = """Meaning: Expressing approval or appreciation for a particular post or comment.
Execution: Upvote when you find content insightful, valuable, or agreeable."""
        @classmethod
        def func(cls, action_args, env, agent):
            if not env.read_only:
                root = env.get_root(env.recent_read_ids[-1])
                root.upvote()
            # res = {"You Just Like the Post Success": env.recent_read_id}
            res = "You Just Like the Post Success"
            return res

    class DisLikeAction(DisLikeAction):
        description = """Meaning: Expressing disapproval or disagreement with content.
Execution: Downvote only when content is not relevant, misleading, or violates community norms. """
        @classmethod
        def func(cls, action_args, env, agent):
            if not env.read_only:
                root = env.get_root(env.recent_read_ids[-1])
                root.upvote()
            res = {"You Just DislikePost": env.recent_read_ids[-1]}
            return res

    action_space = {action.name: action for action in
                    chain(WebAPIEnvironment.action_space.values(), [
                        PostAction, CommentAction, SearchPost, BrowseRising, BrowseControversial, BrowseHots,
                        BrowseRecommendFeed, FollowTopic,
                        LikeAction, DisLikeAction, BrowseHotCommentOfPost])}
    actions_info = [{"ActionName": n, "ActionArgs": a.action_args.keys(), "Description": a.description} for n, a in
                    action_space.items()]

    def step(self, action: Action, agent=None):
        assert agent is not None
        if not action.enable:
            action_res = ''
            observation = "Can't do this action now, Please choose another one."
        else:
            action_res = action.func(action.action_args, self, agent)
            observation = action_res
        return observation


if __name__ == '__main__':
    ts_env = ToySocialNetworkEnvironment(toy_dir='./toy_dir/twitter')
    print([action.name for action in ts_env.action_space])
