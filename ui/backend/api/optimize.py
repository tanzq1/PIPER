# -*- coding: utf-8 -*-
"""Optimization task endpoints."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..models.schemas import (
    OptimizeRequest,
    OptimizeResultPayload,
    TaskCreatedResponse,
    TaskInfo,
)
from ..services import optimizer_runner, preload_cache
from ..services.task_manager import TaskRecord, get_task_manager
from ..settings import get_settings


router = APIRouter(prefix="/api/optimize", tags=["optimize"])


class PreloadRequest(BaseModel):
    dataset: str
    train_mode: Optional[str] = None
    use_estimates: Optional[bool] = None
    train_ratio: Optional[float] = None
    model_dataset: Optional[str] = None
    force_refresh: bool = False


class PreloadResponse(BaseModel):
    dataset: str
    train_mode: str
    use_estimates: bool
    train_ratio: Optional[float]
    model_dataset: str
    row_count: int
    loaded_at: float
    data_signature: str
    cache_hit: bool


@router.post("", response_model=TaskCreatedResponse)
def submit_optimize(request: OptimizeRequest) -> TaskCreatedResponse:
    task_manager = get_task_manager()
    record = task_manager.create("optimize", metadata={
        "dataset": request.dataset,
        "query_id": request.query_id,
        "algorithm": request.algorithm,
    })

    params = request.dict(exclude_unset=False)
    params.pop("dataset", None)
    params.pop("query_id", None)
    params.pop("algorithm", None)
    params.pop("force_rerun", None)

    def _worker() -> None:
        try:
            payload = optimizer_runner.run_optimization(
                task_manager=task_manager,
                task_id=record.task_id,
                dataset=request.dataset,
                query_id=request.query_id,
                algorithm=request.algorithm,
                params=params,
                force_rerun=request.force_rerun,
            )
            task_manager.update_metadata(
                record.task_id,
                optimization_csv_path=payload["optimization_csv_path"],
                optimization_json_path=payload["optimization_json_path"],
                total_cpu_time=payload.get("total_cpu_time"),
                max_dop=payload.get("max_dop"),
            )
            task_manager.mark_completed(record.task_id)
        except Exception as exc:  # noqa: BLE001
            task_manager.mark_failed(record.task_id, str(exc))

    threading.Thread(target=_worker, daemon=True, name=f"optimize-{record.task_id}").start()
    return TaskCreatedResponse(task_id=record.task_id, status=record.status)


@router.get("/{task_id}", response_model=TaskInfo)
def get_optimize_status(task_id: str) -> TaskInfo:
    record = _require_task(task_id)
    return TaskInfo(**record.to_public_dict())


@router.get("/{task_id}/result", response_model=OptimizeResultPayload)
def get_optimize_result(task_id: str) -> OptimizeResultPayload:
    record = _require_task(task_id)
    if record.status != "completed":
        raise HTTPException(status_code=409, detail=f"Task not completed yet: {record.status}")
    payload = optimizer_runner.load_persisted_payload(task_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="No persisted payload for this task")
    return OptimizeResultPayload(**payload)


def _require_task(task_id: str) -> TaskRecord:
    record = get_task_manager().get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="task not found")
    return record


@router.post("/preload", response_model=PreloadResponse)
def preload_state(request: PreloadRequest) -> PreloadResponse:
    """Warm the per-dataset cache (test CSV + ONNX manager) ahead of predictions.

    The first time this is called for a dataset the response includes the time
    spent loading; subsequent calls return immediately and report ``cache_hit``.
    """
    settings = get_settings()
    defaults = settings.optimize_defaults
    train_mode = request.train_mode or defaults.get("train_mode") or "exact_train"
    use_estimates = request.use_estimates if request.use_estimates is not None else bool(defaults.get("use_estimates", False))
    train_ratio = request.train_ratio if request.train_ratio is not None else defaults.get("train_ratio")
    model_dataset = request.model_dataset or defaults.get("model_dataset") or request.dataset

    existing = preload_cache.cache_status().get("entries", [])
    was_hit = any(
        entry["dataset"] == request.dataset
        and entry["train_mode"] == train_mode
        and bool(entry["use_estimates"]) == bool(use_estimates)
        and (entry["train_ratio"] == train_ratio if entry["train_ratio"] is not None or train_ratio is not None else True)
        and entry["model_dataset"] == model_dataset
        for entry in existing
    )

    try:
        state = preload_cache.get_state(
            dataset=request.dataset,
            train_mode=train_mode,
            use_estimates=use_estimates,
            train_ratio=train_ratio,
            model_dataset=model_dataset,
            force_refresh=request.force_refresh,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"preload failed: {exc}") from exc

    row_count = int(getattr(state.df_plans_all_dops, "shape", (0,))[0] or 0)
    return PreloadResponse(
        dataset=state.dataset,
        train_mode=state.train_mode,
        use_estimates=state.use_estimates,
        train_ratio=state.train_ratio,
        model_dataset=state.model_dataset,
        row_count=row_count,
        loaded_at=state.loaded_at,
        data_signature=state.data_signature,
        cache_hit=was_hit and not request.force_refresh,
    )


@router.get("/preload/status")
def preload_status() -> Dict[str, Any]:
    """Diagnostics: list the (dataset, train_mode, ...) keys currently cached."""
    return preload_cache.cache_status()


@router.delete("/preload")
def preload_clear(dataset: Optional[str] = None) -> Dict[str, Any]:
    """Drop cached preload entries (optionally filtered by ``dataset``)."""
    removed = preload_cache.clear_cache(dataset=dataset)
    return {"removed": removed}
