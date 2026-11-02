"""Evaluation utilities with lazy imports for fast unit-test collection."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "compute_guardrails_metrics",
    "load_golden_dataset",
    "run_all_metrics",
    "run_guardrails_eval",
    "run_pipeline",
]


def __getattr__(name: str):
    modules = {
        "run_pipeline": ("evals.pipeline", "run_pipeline"),
        "load_golden_dataset": ("evals.pipeline", "load_golden_dataset"),
        "run_guardrails_eval": ("evals.guardrails_eval", "run_guardrails_eval"),
        "compute_guardrails_metrics": (
            "evals.guardrails_eval",
            "compute_guardrails_metrics",
        ),
        "run_all_metrics": ("evals.metrics", "run_all_metrics"),
    }
    try:
        module_name, attribute_name = modules[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(module_name), attribute_name)
