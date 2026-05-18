# -*- coding: utf-8 -*-
"""Process-wide cache for expensive optimization preprocessing.

Loading a dataset CSV and instantiating the ONNX model manager dominates the
per-request latency of single-query predictions. Caching them across
predictions lets the optimizer run only the per-query DOP search.

The cache key includes anything that changes the underlying tensors or
predictions (dataset, train mode, estimate flag, train ratio, model dataset),
plus a signature derived from the test CSV files so the cache invalidates
automatically when the source data changes.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..settings import Settings, get_settings


def _ensure_predictor_on_path(settings: Settings) -> None:
    root = settings.predictor_root
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass
class PreloadedState:
    """Cached preprocessing state for a (dataset, train_mode, ...) combination."""

    dataset: str
    train_mode: str
    use_estimates: bool
    train_ratio: Optional[float]
    model_dataset: str
    df_plans_all_dops: Any
    onnx_manager: Any
    data_signature: str
    loaded_at: float
    lock: threading.Lock


# Cache key omits ``data_signature``; the signature is stored on the entry and
# checked on every fetch so we can refresh stale state in place.
_CacheKey = Tuple[str, str, bool, Optional[float], str]

_CACHE: Dict[_CacheKey, PreloadedState] = {}
_CACHE_LOCK = threading.Lock()


def _data_signature(settings: Settings, dataset: str) -> str:
    """Return a fingerprint of the dataset's source CSV files (mtime + size)."""
    _ensure_predictor_on_path(settings)
    from config.main_config import DATASETS, PROJECT_ROOT

    source_dir = settings.dataset_data_dirs.get(dataset)
    if not source_dir:
        cfg = DATASETS.get(dataset, {})
        test_subdir = cfg.get("test_dir", "")
        if settings.dataset_data_root:
            candidate = (
                os.path.join(settings.dataset_data_root, test_subdir)
                if test_subdir
                else settings.dataset_data_root
            )
            source_dir = candidate if os.path.isdir(candidate) else settings.dataset_data_root
        else:
            source_dir = os.path.join(PROJECT_ROOT, "data_kunpeng", test_subdir)
    plan_path = os.path.join(source_dir, "plan_info.csv")
    query_path = os.path.join(source_dir, "query_info.csv")
    parts = [f"dataset={dataset}", f"dir={source_dir}"]
    for path in (plan_path, query_path):
        try:
            stat = os.stat(path)
            parts.append(f"{os.path.basename(path)}:{int(stat.st_mtime)}:{stat.st_size}")
        except OSError:
            parts.append(f"{os.path.basename(path)}:missing")
    return "|".join(parts)


def _install_dataset_source_override(settings: Settings) -> None:
    import json

    if settings.dataset_data_dirs:
        os.environ["SERVERLESS_DATASET_DIRS_JSON"] = json.dumps(settings.dataset_data_dirs)
    else:
        os.environ.pop("SERVERLESS_DATASET_DIRS_JSON", None)


def make_cache_key(dataset: str, train_mode: str, use_estimates: bool,
                   train_ratio: Optional[float], model_dataset: str) -> _CacheKey:
    return (
        dataset,
        train_mode,
        bool(use_estimates),
        float(train_ratio) if train_ratio is not None else None,
        model_dataset or dataset,
    )


def get_state(*, dataset: str, train_mode: str, use_estimates: bool,
              train_ratio: Optional[float], model_dataset: Optional[str],
              force_refresh: bool = False,
              logger=None) -> PreloadedState:
    """Return a cached :class:`PreloadedState`, loading or refreshing as needed.

    ``logger`` may be a callable accepting a single message string; it is used
    to surface progress so the UI's task log reflects cache hits and misses.
    """
    settings = get_settings()
    _ensure_predictor_on_path(settings)
    _install_dataset_source_override(settings)

    model_dataset_resolved = model_dataset or dataset
    key = make_cache_key(dataset, train_mode, use_estimates, train_ratio, model_dataset_resolved)
    signature = _data_signature(settings, dataset)

    with _CACHE_LOCK:
        existing = _CACHE.get(key)
        if existing and not force_refresh and existing.data_signature == signature:
            if logger:
                logger(f"[preload] cache hit: dataset={dataset} train_mode={train_mode} "
                       f"loaded_at={time.strftime('%H:%M:%S', time.localtime(existing.loaded_at))}")
            return existing
        if existing:
            entry_lock = existing.lock
        else:
            entry_lock = threading.Lock()
            placeholder = PreloadedState(
                dataset=dataset,
                train_mode=train_mode,
                use_estimates=bool(use_estimates),
                train_ratio=float(train_ratio) if train_ratio is not None else None,
                model_dataset=model_dataset_resolved,
                df_plans_all_dops=None,
                onnx_manager=None,
                data_signature="",
                loaded_at=0.0,
                lock=entry_lock,
            )
            _CACHE[key] = placeholder

    with entry_lock:
        with _CACHE_LOCK:
            entry = _CACHE.get(key)
        if entry and entry.df_plans_all_dops is not None and not force_refresh \
                and entry.data_signature == signature:
            if logger:
                logger(f"[preload] cache hit (post-lock): dataset={dataset} train_mode={train_mode}")
            return entry

        if logger:
            reason = "refresh" if force_refresh else ("first load" if entry is None else "stale signature")
            logger(f"[preload] loading state ({reason}): dataset={dataset} train_mode={train_mode} "
                   f"use_estimates={use_estimates} train_ratio={train_ratio} "
                   f"model_dataset={model_dataset_resolved}")

        load_start = time.time()
        from utils import create_dataset_loader, get_model_paths
        from core.onnx_manager import ONNXModelManager

        loader = create_dataset_loader(dataset)
        df_plans_all_dops = loader.load_test_data(use_estimates, train_ratio=train_ratio)
        if df_plans_all_dops is None:
            raise RuntimeError(f"Failed to load test data for dataset={dataset}")
        if logger:
            logger(f"[preload] loaded {len(df_plans_all_dops)} rows in "
                   f"{time.time() - load_start:.2f}s")

        onnx_start = time.time()
        model_paths = get_model_paths(model_dataset_resolved, train_mode, 'pipeline')
        onnx_manager = ONNXModelManager(
            no_dop_model_dir=model_paths['non_dop_aware_dir'],
            dop_model_dir=model_paths['dop_aware_dir'],
        )
        if logger:
            logger(f"[preload] ONNX manager initialized in {time.time() - onnx_start:.2f}s")

        new_state = PreloadedState(
            dataset=dataset,
            train_mode=train_mode,
            use_estimates=bool(use_estimates),
            train_ratio=float(train_ratio) if train_ratio is not None else None,
            model_dataset=model_dataset_resolved,
            df_plans_all_dops=df_plans_all_dops,
            onnx_manager=onnx_manager,
            data_signature=signature,
            loaded_at=time.time(),
            lock=entry_lock,
        )
        with _CACHE_LOCK:
            _CACHE[key] = new_state

        if logger:
            logger(f"[preload] state ready in {time.time() - load_start:.2f}s total")
        return new_state


def cache_status() -> Dict[str, Any]:
    """Snapshot of cache contents for diagnostics / preload endpoint responses."""
    with _CACHE_LOCK:
        entries = []
        for key, state in _CACHE.items():
            if state.df_plans_all_dops is None:
                continue
            entries.append({
                "dataset": state.dataset,
                "train_mode": state.train_mode,
                "use_estimates": state.use_estimates,
                "train_ratio": state.train_ratio,
                "model_dataset": state.model_dataset,
                "loaded_at": state.loaded_at,
                "row_count": int(getattr(state.df_plans_all_dops, "shape", (0,))[0] or 0),
                "data_signature": state.data_signature,
            })
        return {"entries": entries}


def clear_cache(*, dataset: Optional[str] = None) -> int:
    """Drop cached entries; returns the number of entries removed."""
    removed = 0
    with _CACHE_LOCK:
        to_remove = []
        for key in list(_CACHE.keys()):
            if dataset is None or key[0] == dataset:
                to_remove.append(key)
        for key in to_remove:
            del _CACHE[key]
            removed += 1
    return removed
