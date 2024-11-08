from datetime import datetime
from time import time
from typing import Union, Dict
import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from graphai.api.auth.auth_utils import extract_username_sync


logger = logging.getLogger("graphai_logger")
logger.setLevel(logging.DEBUG)


def get_user_agent(request: Request) -> Union[str, None]:
    if "user-agent" in request.headers:
        return request.headers["user-agent"]
    elif "User-Agent" in request.headers:
        return request.headers["User-Agent"]
    return None


def log_request(request_data: Dict):
    # TODO this is currently a dummy
    logger.debug(f"Logging request: {request_data}")
    print(request_data)
    return


class LoggerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time()
        response = await call_next(request)

        request_data = {
            "hostname": request.url.hostname,
            "ip_address": request.client.host,
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
