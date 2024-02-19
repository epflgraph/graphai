from typing import Union

from db_cache_manager.db import DB
from passlib.context import CryptContext
from pydantic import BaseModel

from graphai.core.common.config import config

AUTH_SCHEMA = config['auth']['schema']


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(username: str):
    db_manager = DB(config['database'])
    columns = ['username', 'full_name', 'email', 'hashed_password', 'disabled']
    query = f"SELECT {', '.join(columns)} FROM {AUTH_SCHEMA}.Users WHERE username=%s"
    results = db_manager.execute_query(query, (username, ))
    if len(results) > 0:
        user_list = results[0]
        user_dict = {columns[i]: user_list[i] for i in range(len(columns))}
        return UserInDB(**user_dict)


def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user
