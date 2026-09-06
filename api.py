"""FastAPI transport layer for the Atlas RAG service."""

from __future__ import annotations

import json
import time
import uuid

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from api_models import HealthResponse, QueryRequest, QueryResponse
from observability import runtime_metrics
from retrieval import AuthorizationFilter
from service import RAGService, ServiceRejected, ServiceUnavailable, health_status


def create_app(service: RAGService | None = None) -> FastAPI:
    """Create an API with an injectable service for deterministic tests."""
    rag_service = service or RAGService()
    application = FastAPI(
        title="Atlas RAG Service",
        version="1.0.0",
        description="Grounded Atlas Logistics policy analysis using hybrid_v1.",
    )

    @application.get("/health", response_model=HealthResponse)
    async def health() -> dict:
        return health_status()

    @application.get("/metrics")
    async def metrics() -> dict:
        result = runtime_metrics.report() if service is None else rag_service.metrics.report()
        result["reliability"] = rag_service.reliability_status()
        return result

    @application.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest, x_request_id: str | None = Header(default=None)) -> dict:
        history = [message.model_dump() for message in request.history]
        authorization = AuthorizationFilter(
            request.authorization.tenant_id,
            tuple(request.authorization.access_scopes),
        )
        try:
            return await rag_service.query_async(
                request.question,
                request.conversation_id,
                history,
                authorization,
                x_request_id or str(uuid.uuid4()),
            )
        except ServiceRejected as error:
            raise HTTPException(
                status_code=error.status_code,
                detail={"message": str(error), "trace_id": error.trace_id, "reason": error.reason},
                headers={"Retry-After": str(error.retry_after)},
            ) from error
        except ServiceUnavailable as error:
            raise HTTPException(
                status_code=503,
                detail={"message": str(error), "trace_id": error.trace_id},
            ) from error

    @application.post("/query/stream")
    async def query_stream(
        request: QueryRequest,
        x_request_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        """Experimental safe streaming after full answer and citation validation."""
        started = time.perf_counter()
        result = await query(request, x_request_id)

        async def events():
            first = True
            for token in result["answer"].split(" "):
                payload = {"token": token + " "}
                if first:
                    payload["time_to_first_token_ms"] = (time.perf_counter() - started) * 1000
                    first = False
                yield f"event: token\ndata: {json.dumps(payload)}\n\n"
            yield "event: done\ndata: " + json.dumps({
                "trace_id": result["trace_id"],
                "total_completion_ms": (time.perf_counter() - started) * 1000,
                "streaming_mode": "validated_buffered",
            }) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return application


app = create_app()
