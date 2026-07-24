# Background Tasks

Deferring post-response work — persistence, audit logging, cache warming — so it does not inflate request latency.

## Async Post-Processing

Use `BackgroundTasks` to persist results or log after the response is sent, keeping request latency low.

```python
from fastapi import BackgroundTasks


async def save_prediction_to_db(request_id: str, detections: list[Detection]) -> None:
    await db.predictions.insert_one(
        {"request_id": request_id, "detections": [d.model_dump() for d in detections]}
    )


@router.post("/predict")
async def predict_with_logging(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    result = await run_prediction(request)
    background_tasks.add_task(save_prediction_to_db, request.state.request_id, result.detections)
    return result
```

## Notes

- Declare `background_tasks: BackgroundTasks` as a parameter; FastAPI injects it. Tasks run **after** the response is flushed to the client.
- `add_task` takes the callable plus its arguments — do not call it (`add_task(fn, arg)`, not `add_task(fn(arg))`).
- Both sync and async callables are accepted. Sync callables run in a threadpool; async ones run on the event loop, so a blocking call inside an `async def` task still stalls the server.
- **Failures are invisible to the client.** The response has already been sent, so a task exception only surfaces in logs. Wrap task bodies in try/except and log with Loguru.
- **Do not use this for work that must not be lost.** Background tasks die with the process. Anything requiring delivery guarantees (retraining triggers, billing events) belongs in a real queue — Celery, RQ, or SQS.
- **Do not use it for long jobs.** The worker is occupied until the task finishes, reducing concurrency. Keep tasks to short I/O like a database insert or a metrics push.
- Pass the request id through so background log lines correlate with the request that produced them.
