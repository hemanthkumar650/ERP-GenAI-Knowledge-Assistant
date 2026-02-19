from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from config import settings


logger = logging.getLogger("erp_rag_assistant")


def _safe_call(target: Any, method_name: str, **kwargs: Any) -> Any:
    method = getattr(target, method_name, None)
    if not callable(method):
        return None
    try:
        return method(**kwargs)
    except Exception as exc:
        logger.debug("Langfuse call failed for %s: %s", method_name, exc)
        return None


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, Mapping):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    return str(obj)


class LangfuseObserver:
    def __init__(self) -> None:
        self.client = None
        self.enabled = False
        if not settings.langfuse_ready:
            return
        try:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            self.enabled = True
            logger.info("Langfuse observability enabled.")
        except Exception as exc:  # pragma: no cover
            logger.warning("Langfuse disabled due to init error: %s", exc)

    def start_trace(self, name: str, input_data: dict | None = None, metadata: dict | None = None):
        if not self.enabled or self.client is None:
            return None
        payload = {
            "name": name,
            "input": _sanitize(input_data or {}),
            "metadata": _sanitize(metadata or {}),
        }
        # Langfuse v2 style
        trace = _safe_call(self.client, "trace", **payload)
        if trace is not None:
            return trace
        # Langfuse v3 style: represent trace as a root span
        return _safe_call(self.client, "start_span", **payload)

    def start_span(self, trace: Any, name: str, input_data: dict | None = None, metadata: dict | None = None):
        payload = {
            "name": name,
            "input": _sanitize(input_data or {}),
            "metadata": _sanitize(metadata or {}),
        }
        if trace is not None:
            # Langfuse v2 style nested span API.
            span = _safe_call(trace, "span", **payload)
            if span is not None:
                return span
            # Langfuse v3 style using explicit trace context.
            trace_id = getattr(trace, "trace_id", None)
            parent_span_id = getattr(trace, "id", None)
            if self.client is not None and trace_id and parent_span_id:
                return _safe_call(
                    self.client,
                    "start_span",
                    trace_context={"trace_id": trace_id, "parent_span_id": parent_span_id},
                    **payload,
                )
        if self.client is None:
            return None
        return _safe_call(self.client, "start_span", **payload)

    def end_span(self, span: Any, output_data: dict | None = None, metadata: dict | None = None) -> None:
        if span is None:
            return
        update_payload: dict[str, Any] = {}
        if output_data is not None:
            update_payload["output"] = _sanitize(output_data)
        if metadata is not None:
            update_payload["metadata"] = _sanitize(metadata)
        if update_payload:
            _safe_call(span, "update", **update_payload)
        # v2 allowed payload in end(); v3 prefers update() then end().
        _safe_call(span, "end")

    def update_trace(self, trace: Any, output_data: dict | None = None, metadata: dict | None = None) -> None:
        if trace is None:
            return
        payload: dict[str, Any] = {}
        if output_data is not None:
            payload["output"] = _sanitize(output_data)
        if metadata is not None:
            payload["metadata"] = _sanitize(metadata)
        _safe_call(trace, "update", **payload)
        # v3 traces represented as root spans should be explicitly closed.
        _safe_call(trace, "end")

    def score_trace(
        self,
        trace: Any,
        *,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        if not self.enabled or self.client is None or trace is None:
            return
        if _safe_call(trace, "score", name=name, value=float(value), comment=comment) is not None:
            return
        trace_id = getattr(trace, "id", None) or getattr(trace, "trace_id", None)
        if trace_id:
            payload: dict[str, Any] = {"trace_id": trace_id, "name": name, "value": float(value)}
            if comment:
                payload["comment"] = comment
            if _safe_call(self.client, "score", **payload) is not None:
                return
        _safe_call(self.client, "score_current_trace", name=name, value=float(value), comment=comment)

    def flush(self) -> None:
        if not self.enabled or self.client is None:
            return
        _safe_call(self.client, "flush")


observer = LangfuseObserver()
