"""Benchmark the computational difference between CUPS .ckpt and ONNX inference.

Measures and compares **backbone + FPN** inference performance for two paths:

1. **ckpt path** – loads the PyTorch Lightning ``.ckpt`` via
   :func:`cups.model.panoptic_cascade_mask_r_cnn_from_checkpoint`, wraps the
   backbone with :class:`convert.BackboneWrapper`, and runs it under
   ``torch.no_grad()``.

2. **onnx path** – loads the ONNX file produced by
   ``python convert.py --format onnx`` with ONNX Runtime and runs
   ``InferenceSession.run()``.

Metrics reported for each path
--------------------------------
* Mean / median / std-dev / min / max latency (ms)
* Throughput (inferences per second)
* Peak RSS memory increase during the timed runs (MB)  –  requires ``psutil``

A final **summary table** is printed to stdout and, if ``--output_json`` is
provided, written to a JSON file for programmatic consumption.

Usage
-----
# Both paths (full comparison)
python benchmark_ckpt_vs_onnx.py \\
    --checkpoint assets/cups.ckpt \\
    --onnx_model cups_backbone.onnx \\
    --image     assets/stuttgart_02_000000_005445_leftImg8bit.png

# Only ONNX (skip --checkpoint)
python benchmark_ckpt_vs_onnx.py \\
    --onnx_model cups_backbone.onnx \\
    --image      assets/stuttgart_02_000000_005445_leftImg8bit.png \\
    --warmup 5 --runs 50

# Save results to JSON
python benchmark_ckpt_vs_onnx.py \\
    --checkpoint assets/cups.ckpt \\
    --onnx_model cups_backbone.onnx \\
    --image      assets/stuttgart_02_000000_005445_leftImg8bit.png \\
    --output_json results/benchmark.json

Requirements
------------
    pip install torch torchvision   # for ckpt path
    pip install onnxruntime          # for onnx path (CPU)
    pip install onnxruntime-gpu      # for onnx path (CUDA)
    pip install Pillow numpy
    pip install psutil               # optional – for memory measurement
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _compute_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute descriptive statistics from a list of per-run latencies.

    Args:
        latencies_ms: Per-run wall-clock latencies in **milliseconds**.

    Returns:
        Dictionary with keys ``mean``, ``median``, ``std``, ``min``, ``max``,
        ``throughput_fps``.
    """
    arr = np.array(latencies_ms, dtype=np.float64)
    mean = float(arr.mean())
    return {
        "mean_ms": mean,
        "median_ms": float(np.median(arr)),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "throughput_fps": 1000.0 / mean if mean > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    """Return the current process RSS memory in megabytes.

    Returns 0.0 if ``psutil`` is not installed.
    """
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def _psutil_available() -> bool:
    try:
        import psutil  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Image loading / pre-processing
# ---------------------------------------------------------------------------


def _load_image_nchw(path: str, height: Optional[int], width: Optional[int]) -> np.ndarray:
    """Load *path* and return a float32 ``(1, 3, H, W)`` NumPy array in [0, 1].

    Args:
        path: Path to a PNG / JPEG / BMP image file.
        height: Resize target height.  ``None`` keeps the original size.
        width: Resize target width.   ``None`` keeps the original size.

    Returns:
        Float32 array ``(1, 3, H, W)`` with pixel values in ``[0.0, 1.0]``.
    """
    try:
        from PIL import Image
    except ImportError:
        log.error("Pillow is required.  Install with:  pip install Pillow")
        sys.exit(1)

    img = Image.open(path).convert("RGB")
    if height is not None or width is not None:
        h = height if height is not None else img.height
        w = width if width is not None else img.width
        img = img.resize((w, h), Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    nchw = np.expand_dims(arr.transpose(2, 0, 1), axis=0)  # (1, 3, H, W)
    return nchw


# ---------------------------------------------------------------------------
# ckpt benchmark
# ---------------------------------------------------------------------------


def _benchmark_ckpt(
    checkpoint_path: str,
    image_nchw: np.ndarray,
    device: str,
    warmup: int,
    runs: int,
) -> Tuple[List[float], float, float]:
    """Benchmark the PyTorch backbone loaded from a ``.ckpt`` checkpoint.

    Args:
        checkpoint_path: Path to the PyTorch Lightning ``.ckpt`` file.
        image_nchw: Float32 NumPy array ``(1, 3, H, W)`` in ``[0, 1]``.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
        warmup: Number of warm-up forward passes (not timed).
        runs: Number of timed forward passes.

    Returns:
        Tuple of:
        * latencies_ms: per-run latency list in milliseconds
        * mem_before_mb: RSS memory before timed runs
        * mem_after_mb: RSS memory after timed runs
    """
    try:
        import torch
    except ImportError:
        log.error("PyTorch is required.  Install with:  pip install torch")
        sys.exit(1)

    # Import BackboneWrapper from convert.py (top-level script)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert import BackboneWrapper, load_checkpoint

    log.info("[ckpt] Loading checkpoint from %s …", checkpoint_path)
    model, _, _ = load_checkpoint(checkpoint_path, device)
    wrapper = BackboneWrapper(model).to(device).eval()

    tensor_input = torch.from_numpy(image_nchw).to(device)

    # Warm-up
    log.info("[ckpt] Warming up (%d runs) …", warmup)
    with torch.no_grad():
        for _ in range(warmup):
            _ = wrapper(tensor_input)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Timed runs
    log.info("[ckpt] Benchmarking (%d runs) …", runs)
    mem_before = _rss_mb()
    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(runs):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = wrapper(tensor_input)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    mem_after = _rss_mb()

    return latencies_ms, mem_before, mem_after


# ---------------------------------------------------------------------------
# ONNX Runtime benchmark
# ---------------------------------------------------------------------------


def _benchmark_onnx(
    onnx_path: str,
    image_nchw: np.ndarray,
    provider: str,
    warmup: int,
    runs: int,
) -> Tuple[List[float], float, float]:
    """Benchmark the backbone ONNX model via ONNX Runtime.

    Args:
        onnx_path: Path to the ``.onnx`` file.
        image_nchw: Float32 NumPy array ``(1, 3, H, W)`` in ``[0, 1]``.
        provider: ONNX Runtime execution provider name.
        warmup: Number of warm-up forward passes (not timed).
        runs: Number of timed forward passes.

    Returns:
        Tuple of:
        * latencies_ms: per-run latency list in milliseconds
        * mem_before_mb: RSS memory before timed runs
        * mem_after_mb: RSS memory after timed runs
    """
    try:
        import onnxruntime as ort
    except ImportError:
        log.error(
            "onnxruntime is required.  Install with:\n"
            "  pip install onnxruntime      (CPU)\n"
            "  pip install onnxruntime-gpu  (GPU)"
        )
        sys.exit(1)

    available = ort.get_available_providers()
    if provider not in available:
        log.warning(
            "[onnx] Provider '%s' not available (%s).  Falling back to CPUExecutionProvider.",
            provider,
            available,
        )
        provider = "CPUExecutionProvider"

    log.info("[onnx] Loading model from %s (provider: %s) …", onnx_path, provider)
    session = ort.InferenceSession(onnx_path, providers=[provider])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    # Warm-up
    log.info("[onnx] Warming up (%d runs) …", warmup)
    for _ in range(warmup):
        session.run(output_names, {input_name: image_nchw})

    # Timed runs
    log.info("[onnx] Benchmarking (%d runs) …", runs)
    mem_before = _rss_mb()
    latencies_ms: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(output_names, {input_name: image_nchw})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    mem_after = _rss_mb()

    return latencies_ms, mem_before, mem_after


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_COL_W = 20


def _hr() -> str:
    return "+" + "-" * (_COL_W + 2) + "+" + "-" * (_COL_W + 2) + "+" + "-" * (_COL_W + 2) + "+"


def _row(label: str, ckpt_val: str, onnx_val: str) -> str:
    return (
        f"| {label:<{_COL_W}} | {ckpt_val:>{_COL_W}} | {onnx_val:>{_COL_W}} |"
    )


def _print_report(
    ckpt_stats: Optional[Dict[str, Any]],
    onnx_stats: Optional[Dict[str, Any]],
    input_shape: Tuple[int, ...],
    warmup: int,
    runs: int,
) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        ckpt_stats: Stats dict from :func:`_compute_stats` for the ckpt path,
            plus ``"mem_delta_mb"`` key.  ``None`` if ckpt was not benchmarked.
        onnx_stats: Stats dict for the ONNX path.  ``None`` if not benchmarked.
        input_shape: NCHW shape of the benchmark input.
        warmup: Number of warm-up runs used.
        runs: Number of timed runs used.
    """

    def _fmt_ms(val: Optional[float]) -> str:
        return f"{val:.2f} ms" if val is not None else "n/a"

    def _fmt_fps(val: Optional[float]) -> str:
        return f"{val:.2f} fps" if val is not None else "n/a"

    def _fmt_mem(val: Optional[float]) -> str:
        if val is None:
            return "n/a"
        return f"{val:+.1f} MB" if _psutil_available() else "n/a (no psutil)"

    def _fmt_speedup(ckpt_mean: Optional[float], onnx_mean: Optional[float]) -> str:
        if ckpt_mean is None or onnx_mean is None or onnx_mean == 0:
            return "n/a"
        ratio = ckpt_mean / onnx_mean
        direction = "faster" if ratio > 1.0 else "slower"
        return f"{ratio:.2f}x  (ONNX {direction})"

    log.info("")
    log.info("=" * (_COL_W * 3 + 10))
    log.info("  CUPS Backbone Inference Benchmark")
    log.info("  Input shape : %s  |  Warmup: %d  |  Runs: %d", input_shape, warmup, runs)
    log.info("=" * (_COL_W * 3 + 10))
    log.info(_hr())
    log.info(_row("Metric", ".ckpt (PyTorch)", ".onnx (ORT)"))
    log.info(_hr())

    cs = ckpt_stats
    os_ = onnx_stats

    log.info(
        _row(
            "Mean latency",
            _fmt_ms(cs["mean_ms"] if cs else None),
            _fmt_ms(os_["mean_ms"] if os_ else None),
        )
    )
    log.info(
        _row(
            "Median latency",
            _fmt_ms(cs["median_ms"] if cs else None),
            _fmt_ms(os_["median_ms"] if os_ else None),
        )
    )
    log.info(
        _row(
            "Std-dev",
            _fmt_ms(cs["std_ms"] if cs else None),
            _fmt_ms(os_["std_ms"] if os_ else None),
        )
    )
    log.info(
        _row(
            "Min latency",
            _fmt_ms(cs["min_ms"] if cs else None),
            _fmt_ms(os_["min_ms"] if os_ else None),
        )
    )
    log.info(
        _row(
            "Max latency",
            _fmt_ms(cs["max_ms"] if cs else None),
            _fmt_ms(os_["max_ms"] if os_ else None),
        )
    )
    log.info(
        _row(
            "Throughput",
            _fmt_fps(cs["throughput_fps"] if cs else None),
            _fmt_fps(os_["throughput_fps"] if os_ else None),
        )
    )
    log.info(
        _row(
            "Peak RSS delta",
            _fmt_mem(cs["mem_delta_mb"] if cs else None),
            _fmt_mem(os_["mem_delta_mb"] if os_ else None),
        )
    )
    log.info(_hr())

    speedup = _fmt_speedup(
        cs["mean_ms"] if cs else None,
        os_["mean_ms"] if os_ else None,
    )
    log.info("  Speedup (mean): %s", speedup)
    log.info("=" * (_COL_W * 3 + 10))
    log.info("")


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def _save_json(
    path: str,
    ckpt_stats: Optional[Dict[str, Any]],
    onnx_stats: Optional[Dict[str, Any]],
    args_dict: Dict[str, Any],
) -> None:
    """Save benchmark results to *path* as a JSON file.

    Args:
        path: Destination ``.json`` file path.
        ckpt_stats: Stats dict for the ckpt path.  ``None`` if not run.
        onnx_stats: Stats dict for the ONNX path.  ``None`` if not run.
        args_dict: Parsed CLI arguments as a plain dict (for provenance).
    """
    speedup: Optional[float] = None
    if ckpt_stats is not None and onnx_stats is not None and onnx_stats["mean_ms"] > 0:
        speedup = ckpt_stats["mean_ms"] / onnx_stats["mean_ms"]

    payload: Dict[str, Any] = {
        "config": args_dict,
        "ckpt": ckpt_stats,
        "onnx": onnx_stats,
        "speedup_mean": speedup,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info("Benchmark results saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark CUPS backbone inference: "
            ".ckpt (PyTorch) vs .onnx (ONNX Runtime)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to the .ckpt checkpoint (PyTorch Lightning).  "
            "Omit to skip the ckpt benchmark."
        ),
    )
    parser.add_argument(
        "--onnx_model",
        default=None,
        help=(
            "Path to the backbone .onnx file produced by convert.py.  "
            "Omit to skip the ONNX benchmark."
        ),
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image (PNG / JPEG / BMP).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warm-up inferences (not timed).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of timed inferences.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Resize input image to this height before inference.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Resize input image to this width before inference.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the ckpt benchmark (cpu / cuda).",
    )
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="ONNX Runtime execution provider for the ONNX benchmark.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to save benchmark results as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the benchmark CLI."""
    args = _parse_args(argv)

    if args.checkpoint is None and args.onnx_model is None:
        log.error("At least one of --checkpoint or --onnx_model must be provided.")
        sys.exit(1)

    if not os.path.isfile(args.image):
        log.error("Image not found: %s", args.image)
        sys.exit(1)

    if args.checkpoint is not None and not os.path.isfile(args.checkpoint):
        log.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    if args.onnx_model is not None and not os.path.isfile(args.onnx_model):
        log.error("ONNX model not found: %s", args.onnx_model)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load image once (shared between both benchmarks)
    # -----------------------------------------------------------------------
    log.info("Loading image from %s …", args.image)
    image_nchw = _load_image_nchw(args.image, args.height, args.width)
    log.info("  Input shape: %s  dtype: %s", image_nchw.shape, image_nchw.dtype)

    # -----------------------------------------------------------------------
    # ckpt benchmark
    # -----------------------------------------------------------------------
    ckpt_stats: Optional[Dict[str, Any]] = None
    if args.checkpoint is not None:
        latencies, mem_before, mem_after = _benchmark_ckpt(
            checkpoint_path=args.checkpoint,
            image_nchw=image_nchw,
            device=args.device,
            warmup=args.warmup,
            runs=args.runs,
        )
        ckpt_stats = _compute_stats(latencies)
        ckpt_stats["mem_delta_mb"] = mem_after - mem_before

    # -----------------------------------------------------------------------
    # ONNX benchmark
    # -----------------------------------------------------------------------
    onnx_stats: Optional[Dict[str, Any]] = None
    if args.onnx_model is not None:
        latencies, mem_before, mem_after = _benchmark_onnx(
            onnx_path=args.onnx_model,
            image_nchw=image_nchw,
            provider=args.provider,
            warmup=args.warmup,
            runs=args.runs,
        )
        onnx_stats = _compute_stats(latencies)
        onnx_stats["mem_delta_mb"] = mem_after - mem_before

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    _print_report(
        ckpt_stats=ckpt_stats,
        onnx_stats=onnx_stats,
        input_shape=image_nchw.shape,
        warmup=args.warmup,
        runs=args.runs,
    )

    # -----------------------------------------------------------------------
    # JSON export
    # -----------------------------------------------------------------------
    if args.output_json is not None:
        args_dict = vars(args)
        _save_json(
            path=args.output_json,
            ckpt_stats=ckpt_stats,
            onnx_stats=onnx_stats,
            args_dict=args_dict,
        )


if __name__ == "__main__":
    main()
