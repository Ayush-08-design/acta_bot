import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class GlobalErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request : Request , call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:  # noqa: BLE001
            logger.error(f'Error - {e!s}')
            # return a messsage to the user
            return Response(
                content = "{'error' : 'Internal Server Error' , 'message' : 'Some error occured'}",
                status_code=500,
                media_type='application/json'
            )