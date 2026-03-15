"""Run inference with a CUPS backbone exported to ONNX and visualise the results.

The ONNX model is the backbone + FPN produced by ``convert.py --format onnx``.
It accepts a single float32 image batch ``(1, 3, H, W)`` with pixel values in
``[0, 1]`` (RGB order) and returns five FPN feature maps:
``p2``, ``p3``, ``p4``, ``p5``, ``p6``.

For each FPN level the script:
  1. Computes the **mean activation** across feature channels.
  2. Normalises the result to ``[0, 255]`` and applies a viridis colormap.
  3. Saves a PNG file ``<output_dir>/<level>_activation.png``.

A summary figure with all levels side-by-side is saved as
``<output_dir>/summary.png``.

Usage
-----
# Basic – auto-resize to the ONNX fixed input size (512×1024)
python infer_onnx.py --onnx_model cups_backbone.onnx \\
                     --image assets/stuttgart_02_000000_005445_leftImg8bit.png \\
                     --output_dir output/

# Override input size (must match the ONNX dynamic-axis constraints)
python infer_onnx.py --onnx_model cups_backbone.onnx \\
                     --image assets/stuttgart_02_000000_005445_leftImg8bit.png \\
                     --output_dir output/ \\
                     --height 512 --width 1024

# GPU inference via CUDA execution provider
python infer_onnx.py --onnx_model cups_backbone.onnx \\
                     --image assets/stuttgart_02_000000_005445_leftImg8bit.png \\
                     --output_dir output/ \\
                     --provider CUDAExecutionProvider

Requirements
------------
    pip install onnxruntime          # CPU
    pip install onnxruntime-gpu      # GPU (CUDA)
    pip install Pillow matplotlib numpy
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------


def _load_image_as_float32(path: str) -> np.ndarray:
    """Load an image file and return a float32 array in RGB [0, 1] order.

    Args:
        path: Path to a PNG / JPEG / BMP image file.

    Returns:
        Array of shape ``(H, W, 3)`` with dtype ``float32`` and values in
        ``[0.0, 1.0]``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the image cannot be read.
    """
    try:
        from PIL import Image
    except ImportError:
        log.error("Pillow is required.  Install with: pip install Pillow")
        sys.exit(1)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize *image* (H×W×3 float32) to (height×width×3) using LANCZOS.

    Args:
        image: Float32 array ``(H, W, 3)`` in ``[0, 1]``.
        height: Target height in pixels.
        width: Target width in pixels.

    Returns:
        Resized float32 array ``(height, width, 3)`` in ``[0, 1]``.
    """
    from PIL import Image

    h, w = image.shape[:2]
    if h == height and w == width:
        return image
    pil = Image.fromarray((image * 255).astype(np.uint8))
    pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.float32) / 255.0


def _hwc_to_nchw(image: np.ndarray) -> np.ndarray:
    """Convert a ``(H, W, 3)`` float32 array to an ``(1, 3, H, W)`` array.

    Args:
        image: Float32 array ``(H, W, 3)``.

    Returns:
        Float32 array ``(1, 3, H, W)``.
    """
    return np.expand_dims(image.transpose(2, 0, 1), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX Runtime helpers
# ---------------------------------------------------------------------------


def _build_session(
    onnx_path: str,
    provider: str,
) -> "onnxruntime.InferenceSession":  # type: ignore[name-defined]  # noqa: F821
    """Create an ONNX Runtime ``InferenceSession``.

    Args:
        onnx_path: Path to the ``.onnx`` model file.
        provider: Execution provider name, e.g. ``"CPUExecutionProvider"``
            or ``"CUDAExecutionProvider"``.

    Returns:
        Initialised ``InferenceSession``.

    Raises:
        SystemExit: If ``onnxruntime`` is not installed or the provider is
            not available.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        log.error(
            "onnxruntime is required.  "
            "Install with: pip install onnxruntime   (CPU)\n"
            "              pip install onnxruntime-gpu (GPU / CUDA)"
        )
        sys.exit(1)

    available = ort.get_available_providers()
    if provider not in available:
        log.warning(
            "Provider '%s' is not available.  Available providers: %s.  "
            "Falling back to CPUExecutionProvider.",
            provider,
            available,
        )
        provider = "CPUExecutionProvider"

    log.info("Creating ONNX Runtime session with provider '%s' …", provider)
    session = ort.InferenceSession(onnx_path, providers=[provider])
    return session


def _run_session(
    session: "onnxruntime.InferenceSession",  # type: ignore[name-defined]  # noqa: F821
    image_nchw: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Run a forward pass through the ONNX session.

    Args:
        session: Initialised ``InferenceSession``.
        image_nchw: Float32 input array ``(1, 3, H, W)`` with values in
            ``[0, 1]`` (RGB order).

    Returns:
        Dictionary mapping output name → numpy array.  For the backbone ONNX
        model the keys are ``"p2"``, ``"p3"``, ``"p4"``, ``"p5"``, ``"p6"``.
    """
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    log.info(
        "Running ONNX inference (input shape: %s) …",
        image_nchw.shape,
    )
    raw_outputs = session.run(output_names, {input_name: image_nchw})
    return dict(zip(output_names, raw_outputs))


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _feature_map_to_rgb(feature_map: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """Convert a 2-D feature map to an RGB image using a matplotlib colormap.

    Args:
        feature_map: 2-D float32 array ``(H, W)``.
        colormap: Matplotlib colormap name.  Defaults to ``"viridis"``.

    Returns:
        ``uint8`` array ``(H, W, 3)`` in ``[0, 255]``.
    """
    import matplotlib.cm as cm

    vmin, vmax = feature_map.min(), feature_map.max()
    if vmax - vmin < 1e-8:
        normalised = np.zeros_like(feature_map)
    else:
        normalised = (feature_map - vmin) / (vmax - vmin)

    rgba = cm.get_cmap(colormap)(normalised)  # (H, W, 4) float64 [0, 1]
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


def _save_rgb(rgb: np.ndarray, path: str) -> None:
    """Save an ``(H, W, 3)`` uint8 array as a PNG file.

    Args:
        rgb: ``uint8`` array ``(H, W, 3)``.
        path: Destination file path.
    """
    from PIL import Image

    Image.fromarray(rgb).save(path)
    log.info("  Saved → %s", path)


def visualise_feature_maps(
    outputs: Dict[str, np.ndarray],
    output_dir: str,
    colormap: str = "viridis",
) -> None:
    """Visualise FPN feature maps and save them to *output_dir*.

    For each FPN level the mean activation across channels is computed,
    normalised, and mapped through a colormap.  A combined summary figure
    is also saved.

    Args:
        outputs: Dictionary mapping FPN level name (e.g. ``"p2"``) to a
            float32 array ``(1, C, H, W)``.
        output_dir: Directory where output images will be written.
        colormap: Matplotlib colormap name.
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    level_names = sorted(outputs.keys())  # p2, p3, p4, p5, p6
    n_levels = len(level_names)

    fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 4))
    if n_levels == 1:
        axes = [axes]

    log.info("\nFPN feature-map statistics:")
    for ax, name in zip(axes, level_names):
        feat = outputs[name][0]  # (C, H, W)
        mean_act = feat.mean(axis=0)  # (H, W)

        log.info(
            "  %s: shape=%s  min=%.4f  mean=%.4f  max=%.4f",
            name,
            feat.shape,
            float(feat.min()),
            float(feat.mean()),
            float(feat.max()),
        )

        # Per-level PNG
        rgb = _feature_map_to_rgb(mean_act, colormap=colormap)
        _save_rgb(rgb, os.path.join(output_dir, f"{name}_activation.png"))

        # Summary subplot
        ax.imshow(mean_act, cmap=colormap, aspect="auto")
        ax.set_title(f"{name}  {feat.shape[1]}×{feat.shape[2]}", fontsize=9)
        ax.axis("off")

    fig.suptitle("FPN Feature-Map Mean Activations (backbone ONNX inference)", fontsize=11)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "summary.png")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Summary figure → %s", summary_path)


def save_input_image(image_rgb: np.ndarray, output_dir: str) -> None:
    """Save the pre-processed input image for reference.

    Args:
        image_rgb: Float32 array ``(H, W, 3)`` in ``[0, 1]``.
        output_dir: Destination directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    rgb_uint8 = (image_rgb * 255).astype(np.uint8)
    _save_rgb(rgb_uint8, os.path.join(output_dir, "input_image.png"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with a CUPS backbone ONNX model and save "
            "FPN feature-map visualisations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--onnx_model",
        required=True,
        help="Path to the backbone .onnx file produced by convert.py.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image (PNG / JPEG / BMP).",
    )
    parser.add_argument(
        "--output_dir",
        default="onnx_output",
        help="Directory where output visualisations will be saved.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            "Resize input image to this height before inference.  "
            "If omitted the original image height is used."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Resize input image to this width before inference.  "
            "If omitted the original image width is used."
        ),
    )
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help=(
            "ONNX Runtime execution provider.  "
            "Options: CPUExecutionProvider, CUDAExecutionProvider."
        ),
    )
    parser.add_argument(
        "--colormap",
        default="viridis",
        help="Matplotlib colormap used for feature-map visualisation.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the ONNX inference CLI."""
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    if not os.path.isfile(args.onnx_model):
        log.error("ONNX model not found: %s", args.onnx_model)
        sys.exit(1)
    if not os.path.isfile(args.image):
        log.error("Image not found: %s", args.image)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load and pre-process image
    # ------------------------------------------------------------------
    log.info("Loading image from %s …", args.image)
    image = _load_image_as_float32(args.image)
    log.info("  Original size: %dx%d (H×W)", image.shape[0], image.shape[1])

    if args.height is not None or args.width is not None:
        h = args.height if args.height is not None else image.shape[0]
        w = args.width if args.width is not None else image.shape[1]
        image = _resize_image(image, h, w)
        log.info("  Resized to: %dx%d (H×W)", h, w)

    image_nchw = _hwc_to_nchw(image)

    # ------------------------------------------------------------------
    # ONNX Runtime inference
    # ------------------------------------------------------------------
    session = _build_session(args.onnx_model, args.provider)
    outputs = _run_session(session, image_nchw)

    # ------------------------------------------------------------------
    # Save visualisations
    # ------------------------------------------------------------------
    log.info("\nSaving results to '%s' …", args.output_dir)
    save_input_image(image, args.output_dir)
    visualise_feature_maps(outputs, args.output_dir, colormap=args.colormap)

    log.info("\nDone.  All outputs written to: %s", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
