"""Lightweight per-stage progress reporting for the annotation pipeline.

A :class:`Progress` prints ``[i/total] label`` lines to a sink (stderr in the
CLI) so a long ``annotate()`` run streams what it is doing instead of going
silent until the end. Each stage prints a start line and, on exit, a completion
line with elapsed time; finer-grained updates inside a stage use :meth:`note`.

Construct with ``sink=None`` for a no-op instance, so the pipeline can call
``progress.stage(...)`` unconditionally whether or not the caller wants output.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

__all__ = ["Progress"]


class Progress:
    def __init__(
        self,
        total: int,
        *,
        sink: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.total = max(total, 0)
        self._n = 0
        self._sink = sink
        self._lock = threading.Lock()

    def _emit(self, line: str) -> None:
        if self._sink is None:
            return
        # Serialize writes so the parallel refine pass can report safely.
        with self._lock:
            self._sink(line)

    def note(self, message: str) -> None:
        """Emit an indented sub-step line under the current stage."""
        self._emit(f"      ↳ {message}")

    def summary(self, lines: list[str]) -> None:
        """Emit a trailing block (blank separator + each line), for end-of-run stats."""
        if self._sink is None or not lines:
            return
        self._emit("")
        for line in lines:
            self._emit(line)

    @contextmanager
    def stage(self, label: str) -> Iterator["Progress"]:
        """Run a block as one counted stage, printing start and done lines."""
        with self._lock:
            self._n += 1
            n = self._n
        prefix = f"[{n}/{self.total}]"
        self._emit(f"{prefix} {label} …")
        start = time.monotonic()
        try:
            yield self
        except BaseException:
            self._emit(f"{prefix} {label} ✗ ({time.monotonic() - start:.1f}s)")
            raise
        self._emit(f"{prefix} {label} ✓ ({time.monotonic() - start:.1f}s)")
