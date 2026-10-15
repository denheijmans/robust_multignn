"""Wall-clock and peak-memory measurement for cost benchmarks.

Used by ``analysis/scripts/benchmark_aml_cost.py`` to support the scalability
claim: PR-BCD's search space is the block, not the graph, so attack memory is
governed by ``block_size`` rather than by |E|.

Two details matter for the numbers to mean anything:

* CUDA kernels are asynchronous, so a region must be synchronised at both ends
  or the elapsed time measures queueing rather than execution.
* ``max_memory_allocated`` is reset per region; without the reset every
  measurement reports the running peak of the whole process.

Host RAM is reported alongside GPU memory because AML graphs are loaded from
multi-GB CSVs and sampled on the CPU, so the host side can dominate.
"""

from __future__ import annotations

import gc
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

import torch


def _host_peak_bytes() -> int:
    """Peak resident set size of this process, in bytes.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


@dataclass
class Measurement:
    """One timed region.

    ``host_peak_bytes`` is a process-wide high-water mark, not a per-region
    figure: RSS is never returned to the OS, so it only ever rises. It is
    meaningful as "peak host memory up to and including this phase".
    """

    label: str
    seconds: float
    gpu_peak_bytes: int | None
    gpu_reserved_bytes: int | None
    host_peak_bytes: int
    extra: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        extra = row.pop("extra")
        row["gpu_peak_gb"] = (
            None if self.gpu_peak_bytes is None else self.gpu_peak_bytes / 1e9
        )
        row["gpu_reserved_gb"] = (
            None if self.gpu_reserved_bytes is None else self.gpu_reserved_bytes / 1e9
        )
        row["host_peak_gb"] = self.host_peak_bytes / 1e9
        return {**row, **extra}


@contextmanager
def measure(
    label: str,
    device: torch.device | str | int | None = None,
    *,
    collect: bool = True,
    **extra: Any,
) -> Iterator[Measurement]:
    """Time a region and record its peak memory.

    The yielded :class:`Measurement` is populated on exit, so read its fields
    only after the ``with`` block closes.

    Parameters
    ----------
    label
        Name of the phase being measured (e.g. ``"attack"``).
    device
        CUDA device to synchronise and read memory stats from. ``None`` or a
        CPU device records timing and host RSS only.
    collect
        Run ``gc.collect()`` and empty the CUDA caching allocator before
        starting. Keeps a previous phase's garbage out of this phase's peak.
    **extra
        Additional columns to carry into :meth:`Measurement.as_row`, such as
        the dataset and block size the region ran under.
    """
    dev = torch.device(device) if device is not None else None
    on_cuda = dev is not None and dev.type == "cuda" and torch.cuda.is_available()

    if collect:
        gc.collect()
        if on_cuda:
            torch.cuda.empty_cache()

    if on_cuda:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    record = Measurement(
        label=label,
        seconds=float("nan"),
        gpu_peak_bytes=None,
        gpu_reserved_bytes=None,
        host_peak_bytes=0,
        extra=dict(extra),
    )

    start = time.perf_counter()
    try:
        yield record
    finally:
        # Drain the queue before stopping the clock, otherwise the elapsed
        # time only covers kernel launches.
        if on_cuda:
            torch.cuda.synchronize(dev)
        record.seconds = time.perf_counter() - start
        if on_cuda:
            record.gpu_peak_bytes = int(torch.cuda.max_memory_allocated(dev))
            record.gpu_reserved_bytes = int(torch.cuda.max_memory_reserved(dev))
        record.host_peak_bytes = _host_peak_bytes()


def predicted_block_bytes(
    block_num_edges: int,
    block_num_nodes: int,
    *,
    bytes_per_scalar: int = 4,
    optimizer_multiplier: int = 4,
) -> int:
    """Analytic size of PR-BCD's learnable block state.

    ``target_logits`` (block_edges x block_nodes) dominates ``edge_probs``
    (block_edges), and AdamW holds a gradient plus two moment buffers per
    parameter -- hence four copies by default.

    Independent of the graph: this is the quantity the scalability claim says
    replaces an |E|-sized search space.
    """
    params = block_num_edges * block_num_nodes + block_num_edges
    return params * bytes_per_scalar * optimizer_multiplier
