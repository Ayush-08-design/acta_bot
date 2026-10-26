import logging
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request : Request , call_next):
        # generate a unique id or read it if already passed by a proxy /load balancer
        request_id = request.headers.get('X-Request-ID' , str(uuid.uuid4()))
        # attach the id to the request state so logs and routes can read it
        request.state.request_id = request_id
        # process the request
        response = await call_next(request)
        # send id back to the response headers so the frontend can read it
        response.headers['X-Request-ID'] = request_id
        return response