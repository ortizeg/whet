# FastAPI Skill

The FastAPI Skill provides expert patterns for building ML model serving APIs with FastAPI, covering async endpoints, Pydantic request/response validation, dependency injection for model loading, middleware for request logging and error handling, WebSocket streaming for real-time inference, and production deployment with Uvicorn.

## Purpose

When you need to serve ML models over HTTP — whether for batch predictions, real-time video inference, or integration with frontend applications — FastAPI is the standard framework. This skill encodes best practices for structuring FastAPI applications in ML/CV contexts: proper model lifecycle management via lifespan handlers, typed request/response schemas, async inference patterns, and production-ready Dockerfile configurations.

## When to Use

- When building REST APIs that serve trained models (ONNX, TensorRT, PyTorch).
- When you need real-time WebSocket streaming for video inference pipelines.
- When constructing microservices that wrap ML inference behind HTTP endpoints.
- When building model serving APIs that need health checks, batch endpoints, and structured error responses.

## Key Features

- **Application factory pattern** — configurable app creation with lifespan-managed model loading.
- **Pydantic schemas** — frozen BaseModel for all request and response types with field validators.
- **Dependency injection** — model and preprocessor provided via FastAPI's `Depends` system.
- **Middleware** — request ID injection, logging, timing, and structured exception handling.
- **WebSocket streaming** — real-time frame-by-frame inference over persistent connections.
- **Background tasks** — async result logging and post-processing without blocking responses.
- **Testing** — async test client patterns with httpx `ASGITransport`.

## Related Skills

- **[Pydantic](../pydantic/)** — frozen BaseModel patterns used for all API schemas.
- **[ONNX](../onnx/)** / **[TensorRT](../tensorrt/)** — optimized model formats loaded in the lifespan handler.
- **[Docker CV](../docker-cv/)** — production containerization for FastAPI ML services.
- **[Loguru](../loguru/)** — structured logging in middleware and exception handlers.
- **[Testing](../testing/)** — async endpoint testing with httpx.
