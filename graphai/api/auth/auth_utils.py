import json
from typing import Union, List

from db_cache_manager.db import DB
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from graphai.core.common.config import config

import string
import random
import cachetools.func

AUTH_SCHEMA = config['auth']['schema']
ALL_SCOPES = ['user', 'voice', 'video', 'translation', 'text', 'scraping', 'ontology', 'image', 'completion']
DEFAULT_RATE_LIMIT_SCHEMA = 'unlimited'
DEFAULT_RATE_LIMITS = {
    'base': {
        'global': {
            'max_requests': 100,
            'window': 1
        },
        'video': {
            'max_requests': 20,
            'window': 10
        },
        'image': {
            'max_requests': 30,
            'window': 10
        },
        'voice': {
            'max_requests': 20,
            'window': 10
        },
        'translation': {
            'max_requests': 10,
            'window': 1
        },
        'scraping': {
            'max_requests': 10,
            'window': 10
        },
        'rag': {
            'max_requests': 20,
            'window': 1
        }
    },
    DEFAULT_RATE_LIMIT_SCHEMA: {
        'global': {
            'max_requests': None,
            'window': None
        },
        'video': {
            'max_requests': None,
            'window': None
        },
        'image': {
            'max_requests': None,
            'window': None
        },
        'voice': {
            'max_requests': None,
            'window': None
        },
        'translation': {
            'max_requests': None,
            'window': None
        },
        'scraping': {
            'max_requests': None,
            'window': None
        },
        'rag': {
            'max_requests': None,
            'window': None
        }
    },
}


@cachetools.func.lru_cache(maxsize=512)
def get_ratelimit_values():
    # Load rate-limit dictionary
    try:
        with open(config.get('ratelimiting', {'custom_limits': ''}).get('custom_limits', ''), 'r') as f:
            rate_limit_values = json.load(f)
            rate_limit_values.update(DEFAULT_RATE_LIMITS)
    except (FileNotFoundError, json.JSONDecodeError):
        rate_limit_values = DEFAULT_RATE_LIMITS
    # Fill in the blanks of the rate-limit dictionary
    for key in DEFAULT_RATE_LIMITS[DEFAULT_RATE_LIMIT_SCHEMA].keys():
        if key not in rate_limit_values:
            rate_limit_values[key] = DEFAULT_RATE_LIMITS[DEFAULT_RATE_LIMIT_SCHEMA][key]
    # Load selected rate-limiting schema
    limit_schema = config.get(
        'ratelimiting', {'limit': DEFAULT_RATE_LIMIT_SCHEMA}
    ).get('limit', DEFAULT_RATE_LIMIT_SCHEMA)
    if limit_schema not in rate_limit_values.keys():
        limit_schema = DEFAULT_RATE_LIMIT_SCHEMA \
            if DEFAULT_RATE_LIMIT_SCHEMA in rate_limit_values.keys() else list(rate_limit_values.keys())[0]
    return rate_limit_values[limit_schema]


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None
    scopes: List[str] = []


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    scopes: Union[List[str]] = []
    disabled: Union[bool, None] = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def generate_random_password_string(length=32):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
                   for _ in range(length))


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


@cachetools.func.ttl_cache(maxsize=1024, ttl=12 * 3600)
def get_user(username: str):
    db_manager = DB(config['database'])
    columns = ['username', 'full_name', 'email', 'hashed_password', 'disabled', 'scopes']
    query = f"SELECT {', '.join(columns)} FROM {AUTH_SCHEMA}.Users WHERE username=%s"
    results = db_manager.execute_query(query, (username, ))
    if len(results) > 0:
        user_list = results[0]
        user_dict = {columns[i]: user_list[i] for i in range(len(columns))}
        if user_dict['scopes'] is not None:
            user_dict['scopes'] = user_dict['scopes'].strip().split(',')
        else:
            user_dict['scopes'] = list()
        return UserInDB(**user_dict)


@cachetools.func.ttl_cache(maxsize=1024, ttl=12 * 3600)
def get_user_ratelimit_overrides(username: str, path: str):
    db_manager = DB(config['database'])
    columns = ['username', 'max_requests', 'window_size']
    try:
        query = (f"SELECT {', '.join(columns)} FROM {AUTH_SCHEMA}.User_Rate_Limits "
                 f"WHERE username=%s AND api_path=%s")
        results = db_manager.execute_query(query, (username, path, ))
        if len(results) > 0:
            return {columns[i]: results[0][i] for i in range(len(columns))}
        else:
            return None
    except Exception:
        return None


def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


SECRET_KEY = config['auth']['secret_key']
ALGORITHM = "HS256"


async def extract_username_and_scopes(token):
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    username: str = payload.get("sub")
    token_scopes = payload.get("scopes", [])
    return username, token_scopes


def extract_username_sync(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        return username
    except Exception:
        return '__UNAUTHENTICATED___USER__'


@cachetools.func.ttl_cache(maxsize=1024, ttl=12 * 3600)
def has_rag_access_rights(username, index_name):
    """
    Checks to see if a given user has access to a given index for the /rag/retreive endpoint.
    Args:
        username: Username
        index_name: Name of the index

    Returns:
        True if the user is granted access or if the access management table has not been set up, False otherwise.
    """
    db_manager = DB(config['database'])
    # First, we try without aliases
    try:
        query = (f"SELECT index_name FROM {AUTH_SCHEMA}.User_Retrieve_Access "
                 f"WHERE username=%s;")
        permitted_rags = db_manager.execute_query(query, (username, ))
        permitted_rags = [row[0] for row in permitted_rags]
        # Either the user has access to this particular index, or to any and all indexes.
        if index_name in permitted_rags or 'ANY/ALL' in permitted_rags:
            return True
    except Exception as e:
        # If the base table doesn't exist, anyone is permitted to access any index
        print(e)
        return True
    # If we're here, the User_Retrieve_Access table exists, and this user seems not to have access.
    # We now check the aliases to see if that yields something.
    try:
        query = (f"SELECT b.alias_name FROM {AUTH_SCHEMA}.User_Retrieve_Access a "
                 f"INNER JOIN {AUTH_SCHEMA}.Retrieve_Index_Aliases b "
                 f"ON a.index_name=b.index_name "
                 f"WHERE a.username=%s;")
        permitted_rags = db_manager.execute_query(query, (username, ))
        print(permitted_rags)
        permitted_rags = [row[0] for row in permitted_rags]
        if index_name in permitted_rags:
            return True
        # If the index is not found in the aliases either, the user does not have access.
        return False
    except Exception as e:
        # If there are no aliases, we conclude that this user does not have access.
        print(e)
        return False
