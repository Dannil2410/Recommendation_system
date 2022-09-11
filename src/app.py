import os
from typing import List
from fastapi import FastAPI
from .schema import PostGet, Response
from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger
import pandas as pd
import hashlib
import joblib

app = FastAPI()

database = ''
user = ''
password = ''
host = ''
port = 0

CONNECTION = f"postgresql://{user}:{password}@{host}:{port}/{database}"


def batch_load_sql(query: str):
    """Loading Models"""
    engine = create_engine(CONNECTION)
    conn = engine.connect().execution_options(
        stream_results=True
    )

    chunks = []

    for chunk_dataframe in pd.read_sql(query, conn, chunksize=200000):
        chunks.append(chunk_dataframe)
        logger.info(f"Got chuck:{len(chunk_dataframe)}")
    conn.close()

    return pd.concat(chunks, ignore_index=True)


def load_features():
    # Distinct notes for user_id and post_id
    # where like has been done
    logger.info("Loading liked posts")
    liked_post_query = """
        SELECT DISTINCT post_id, user_id
        FROM public.feed_data
        WHERE action='like'"""
    liked_posts = batch_load_sql(liked_post_query)

    # Features for posts on basis of TF-IDF
    logger.info("Loading posts features")
    posts_features = pd.read_sql(
        """SELECT * FROM public.post_processed_features""",

        con=CONNECTION
    )

    # Features for users
    logger.info("Loading users features")
    users_features = pd.read_sql(
        """SELECT * FROM public.user_data""",

        con=CONNECTION
    )

    return [liked_posts, posts_features, users_features]


# Download models

def get_model_path(path: str, exp_group: str) -> str:
    """Check where code is being performed: LMS or local"""
    if os.environ.get("IS_LMS") == "1":
        model_path = f'/workdir/user_input/model_{exp_group}'
    else:
        model_path = path
    return model_path


def load_models(exp_group: str):
    model_path = get_model_path(f"C:\PATH\model_{exp_group}.pkl", exp_group)
    model = joblib.load(model_path)
    return model


logger.info("Loading control model")
model_control = load_models('control')

logger.info("Loading test model")
model_test = load_models('test')

logger.info("Loading features")
features = load_features()

logger.info("Service is up and running")


def get_exp_group(user_id):
    logger.info('Select a group for user')
    salt = 'salt'
    value_str = str(user_id) + salt
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num % 100
    if percent < 50:
        return 'control'
    elif percent < 100:
        return 'test'
    return 'unknown'


def get_recommended_feed(id: int, time: datetime, limit: int, exp_group: str):
    # Download features for users
    logger.info(f"user_{id}")
    logger.info("reading features")
    users_features = features[2].loc[features[2].user_id == id]
    users_features = users_features.drop('user_id', axis=1)

    # Download features for posts
    logger.info("dropping columns")
    posts_features = features[1].drop('text', axis=1)
    content = features[1][['post_id', 'text', 'topic']]

    # Unite features for users and posts
    logger.info("zipping everything")
    add_user_features = dict(zip(users_features.columns, users_features.values[0]))
    user_posts_features = posts_features.assign(**add_user_features)
    user_posts_features = user_posts_features.set_index('post_id')

    # Add information about recommendation datetime
    logger.info("Adding date information")
    user_posts_features['hour'] = time.hour
    user_posts_features['dayofweek'] = time.weekday()
    user_posts_features['month'] = time.month

    # Form predict probability "post is liked" for all posts
    logger.info("Predicting that post will be liked")
    if exp_group == 'control':
        prediction_liked = model_control.predict_proba(user_posts_features)[:, 1]
    elif exp_group == 'test':
        prediction_liked = model_test.predict_proba(user_posts_features)[:, 1]
    user_posts_features['predict'] = prediction_liked

    # Select posts which haven't been liked
    logger.info("Deleting liked posts")
    posts_already_liked = features[0][features[0]['user_id'] == id]
    posts_already_liked_ids = list(posts_already_liked['post_id'].values)
    user_posts_features_unliked = user_posts_features[~user_posts_features.index.isin(posts_already_liked_ids)]
    posts_for_recommendation_id = list(user_posts_features_unliked.sort_values('predict')[-limit:].index)

    return Response(
        **{
            "exp_group": exp_group,
            "recommendations": [
                PostGet(**{
                    "id": id,
                    "text": content[content.post_id == id].text.values[0],
                    "topic": content[content.post_id == id].topic.values[0]
                }) for id in posts_for_recommendation_id
            ]
        }
    )


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 10) -> List[PostGet]:

    group = get_exp_group(id)

    if not group:
        logger.info("Control group!!!")
        return get_recommended_feed(id, time, limit, 'control')
    else:
        logger.info("Test group!!!")
        return get_recommended_feed(id, time, limit, 'test')
