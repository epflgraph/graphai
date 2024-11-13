from datetime import datetime
from time import time
from typing import Union, Dict
import json
import os
import logging
from logging.handlers import RotatingFileHandler

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from graphai.api.auth.auth_utils import extract_username_sync
from graphai.core.common.config import config
from graphai.core.common.common_utils import make_sure_path_exists


logger = logging.getLogger("graphai_logger")
logger.setLevel(logging.DEBUG)
logging_dir = config.get('logging', dict()).get('path', os.path.expanduser('~/graphai_logs'))
make_sure_path_exists(logging_dir)
handler = RotatingFileHandler(
    filename=os.path.join(logging_dir, 'graphai_logger.log'),
    maxBytes=50 * 1024 * 1024,
    backupCount=10
)
logger.addHandler(handler)


def get_user_agent(request: Request) -> Union[str, None]:
    if "user-agent" in request.headers:
        return request.headers["user-agent"]
    elif "User-Agent" in request.headers:
        return request.headers["User-Agent"]
    return None


def log_request(request_data: Dict):
    instance_name = config.get('logging', dict()).get('server_name', 'graphai')
    message = (f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}] "
               f"[{instance_name}] [{os.getpid()}] [DEBUG] {json.dumps(request_data)}")
    logger.debug(message)
    return


class LoggerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time()
        response = await call_next(request)

        try:
            ip_address = request.client.host
        except AttributeError:
            ip_address = ''

        request_data = {
            "hostname": request.url.hostname,
            "ip_address": ip_address,
            "path": request.url.path,
            "user_agent": get_user_agent(request),
            "method": request.method,
            "status": response.status_code,
            "response_time": int((time() - start) * 1000),
            "user_id": extract_username_sync(request.headers.get('authorization', '').replace('Bearer ', '')),
            "created_at": datetime.now().isoformat(),
        }

        log_request(request_data)

        return response
