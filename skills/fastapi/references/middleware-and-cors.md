# Middleware and CORS

Cross-cutting request handling: CORS configuration and a request-ID + timing logging middleware.

## CORS

Register CORS in the application factory, driven by configuration — never a hardcoded origin list in production code:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

`allow_origins=["*"]` is acceptable for local development only. In production, load an explicit origin list from configuration; `allow_credentials=True` combined with a wildcard origin is rejected by browsers anyway.

## Request ID and Logging Middleware

```python
from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing and request ID."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "{} {} → {} ({:.1f}ms) [{}]",
            request.method, request.url.path, response.status_code, elapsed, request_id,
        )
        response.headers["X-Request-ID"] = request_id
        return response
```

## Notes

- The middleware stashes `request_id` on `request.state`, which exception handlers and background tasks read back to correlate logs with responses. The same id is echoed to the client in the `X-Request-ID` header.
- Middleware runs in reverse registration order on the way out. Register the logging middleware early so it wraps (and therefore times) everything else.
- Keep middleware non-blocking. Anything CPU-bound here inflates latency for every route; push that work into a background task instead.
- Use Loguru's brace-style formatting (`"{} {}"`, args) rather than f-strings so the log record keeps its structured fields.
