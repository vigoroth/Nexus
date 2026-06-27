import time
import psycopg
from contextlib import contextmanager
from app.core.config import get_settings


def _conn():
    url = get_settings().database_url.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(url)


def init_metrics_table() -> None:
    """Create the run_metrics table if it doesn't exist."""
    with _conn() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS run_metrics ("
            "id SERIAL PRIMARY KEY, "
            "ts TIMESTAMPTZ DEFAULT now(), "
            "label TEXT, "
            "duration_ms DOUBLE PRECISION, "
            "input_tokens INTEGER, "
            "output_tokens INTEGER, "
            "cost_usd DOUBLE PRECISION, "
            "success BOOLEAN, "
            "error TEXT)"
        )


def record_run(label, duration_ms, input_tokens, output_tokens, cost_usd, success, error=None):
    """Write one run's metrics to Postgres."""
    with _conn() as conn:
        conn.execute(
            "INSERT INTO run_metrics "
            "(label, duration_ms, input_tokens, output_tokens, cost_usd, success, error) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (label, duration_ms, input_tokens, output_tokens, cost_usd, success, error),
        )


@contextmanager
def track(label):
    """Context manager that times a block and records it, even on error.

    Usage:
        with track("agent_run") as m:
            ... do work ...
            m["input_tokens"] = 30
            m["cost_usd"] = 0.0001
    """
    start = time.perf_counter()
    m = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    success = True
    error = None
    try:
        yield m
    except Exception as e:
        success = False
        error = str(e)
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        record_run(
            label, duration_ms,
            m["input_tokens"], m["output_tokens"], m["cost_usd"],
            success, error,
        )