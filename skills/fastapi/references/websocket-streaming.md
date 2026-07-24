# WebSocket Streaming

Real-time frame-by-frame inference over a WebSocket, for live video or camera feeds where per-request HTTP overhead is unacceptable.

## Streaming Endpoint

```python
import base64

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()


@router.websocket("/ws/stream")
async def stream_inference(websocket: WebSocket) -> None:
    """Stream video frames and return detections in real time."""
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_json()
            frame_b64 = data.get("frame")
            if not frame_b64:
                await websocket.send_json({"error": "Missing 'frame' field"})
                continue

            frame_bytes = base64.b64decode(frame_b64)
            detections = await run_inference(frame_bytes)

            await websocket.send_json({
                "detections": [d.model_dump() for d in detections],
                "frame_id": data.get("frame_id"),
            })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
```

## Notes

- **Always `await websocket.accept()` first.** Nothing can be sent or received before the handshake completes.
- **Always catch `WebSocketDisconnect`.** Clients drop without closing cleanly; an uncaught disconnect fills logs with tracebacks and can leak per-connection resources.
- **Echo `frame_id` back.** WebSocket responses are not request-scoped, so the client needs a correlation key to match a detection payload to the frame it sent.
- **Validate the frame payload.** The example returns a JSON error and continues rather than closing the socket on a malformed message. For stricter services, validate the incoming dict with a Pydantic model and send back the `ErrorResponse` shape.
- **`model_dump()` before sending.** `send_json` cannot serialize Pydantic models, numpy arrays, or tensors directly.
- **Never block the event loop.** `run_inference` must be an async wrapper; for a synchronous model, dispatch it with `run_in_executor` so other connections keep flowing.
- **Consider back-pressure.** A producer sending frames faster than the GPU can process them will grow the receive queue without bound — drop stale frames client-side or gate on an acknowledgement.
