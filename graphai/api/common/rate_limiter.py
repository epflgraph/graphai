import os
import time
import random
import logging
from typing import Union, Callable

import redis.asyncio as redis
from fastapi import Request, HTTPException, status
from starlette.datastructures import Headers

from graphai.core.common.config import config


logger = logging.getLogger(__name__)

DEFAULT_REDIS_URL = 'redis://localhost:6379/1'


def _get_default_redis_url() -> str:
    """Return the Redis URL to use for the shared rate-limit pool.

    Resolution order:
      1. ``GRAPHAI_RATE_LIMITER_REDIS_URL`` environment variable.
      2. ``redis_url`` key under the ``[ratelimiting]`` config section.
      3. ``DEFAULT_REDIS_URL``.
    """
    env_url = os.getenv('GRAPHAI_RATE_LIMITER_REDIS_URL')
    if env_url:
        return env_url
    try:
        cfg_url = config['ratelimiting'].get('redis_url')
        if cfg_url:
            return cfg_url
    except Exception:
        pass
    return DEFAULT_REDIS_URL


class SharedRateLimiterConnection:
    """Redis-backed rate limiter that reuses a single connection pool.

    This is a drop-in replacement for the per-request connection creation done
    by ``fastapi-user-limiter``.  The pool is created once per process and is
    safe to share across concurrent async request handlers.
    """

    def __init__(self, redis_url: Union[str, None] = None):
        if redis_url is None:
            redis_url = _get_default_redis_url()
        self.redis_url = redis_url
        # redis.asyncio.Redis backed by the default pool is safe to share
        # across coroutines within the same process.
        self.redis = redis.from_url(redis_url, decode_responses=True)

    async def is_rate_limited(self, key: str, max_requests: int, window: int) -> bool:
        # Negative max_requests values disable rate-limiting, matching the
        # behaviour of the original fastapi-user-limiter implementation.
        if max_requests < 0:
            return False

        current_time = time.time()
        current_time_key = (('%.06f' % current_time).replace('.', '')
                            + '%08d' % random.randint(0, int(1e7)))
        window_start = current_time - window

        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zcard(key)
                pipe.zadd(key, {current_time_key: current_time})
                pipe.expire(key, window)
                results = await pipe.execute()
        except redis.RedisError as exc:
            logger.warning(f"Rate limiter Redis error: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Redis error: {str(exc)}"
            ) from exc

        # results[1] is the output of pipe.zcard(key), i.e. the number of
        # requests already made in the current window before this request.
        return results[1] >= max_requests

    async def close(self):
        await self.redis.close()


# Module-level singleton.  It is initialised lazily so import-time config
# loading is safe.
_connection: Union[SharedRateLimiterConnection, None] = None


def get_rate_limiter_connection(redis_url: Union[str, None] = None) -> SharedRateLimiterConnection:
    """Return the shared rate limiter Redis connection."""
    global _connection
    if _connection is None:
        _connection = SharedRateLimiterConnection(redis_url)
    return _connection


async def close_rate_limiter_connection():
    """Close the shared rate limiter Redis connection, if it was created."""
    global _connection
    if _connection is not None:
        await _connection.close()
        _connection = None


def _rate_limit_message(max_requests, window):
    return (f"Too many requests, no more than {max_requests} requests "
            f"are allowed every {window} seconds.")


def rate_limiter(
    max_requests: Union[int, None] = 10,
    window: Union[int, None] = 1,
    path: Union[str, None] = None,
    user: Union[Callable[[Headers, str], Union[str, dict]], None] = None,
    redis_url: Union[str, None] = None,
):
    """Drop-in replacement for ``fastapi_user_limiter.limiter.rate_limiter``.

    Uses a shared Redis connection pool instead of creating a new connection
    for every request.
    """
    conn = get_rate_limiter_connection(redis_url)

    async def _rate_limit(request: Request):
        # Providing a None value for either window or max_requests disables
        # rate limiting.
        if max_requests is None or window is None:
            return

        n_max_requests = max_requests
        window_size = window
        path_name = request.url.path if path is None else path

        if user is None:
            user_name = request.client.host
        else:
            user_output = await user(request.headers, path_name)
            if isinstance(user_output, str):
                user_name = user_output
            else:
                assert 'username' in user_output.keys()
                user_name = user_output['username']
                n_max_requests = user_output.get('max_requests', n_max_requests)
                window_size = user_output.get('window', window_size)
                # The values may have been overridden to None; if so, disable
                # rate limiting for this user/path.
                if n_max_requests is None or window_size is None:
                    return

        key = f"rate_limit:{path_name}:{window_size}:{n_max_requests}:{user_name}"
        if await conn.is_rate_limited(key, n_max_requests, window_size):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=_rate_limit_message(n_max_requests, window_size)
            )

    return _rate_limit
