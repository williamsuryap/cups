"""Convert CUPS model checkpoints (.ckpt) to ONNX or TorchScript format.

Supported output formats
------------------------
* ``.pth``  – plain PyTorch state-dict (no PyTorch Lightning dependency at load time)
* ``.onnx`` – ONNX graph of the backbone + FPN feature extractor
* ``.pt``   – TorchScript traced model of the backbone + FPN feature extractor
* TensorFlow SavedModel – optional, requires ``onnx2tf`` (``pip install onnx2tf``)

Usage examples
--------------
# 1. Extract Lightning checkpoint → plain state-dict
python convert.py --checkpoint assets/cups.ckpt --output cups_weights.pth --format pth

# 2. Export backbone to ONNX (default 512×1024 input)
python convert.py --checkpoint assets/cups.ckpt --output cups_backbone.onnx --format onnx

# 3. Export backbone to TorchScript
python convert.py --checkpoint assets/cups.ckpt --output cups_backbone.pt --format torchscript

# 4. Export backbone to ONNX then convert to TensorFlow SavedModel
python convert.py --checkpoint assets/cups.ckpt --output cups_backbone.onnx --format onnx --to_tf

Notes
-----
The **full** panoptic model (Cascade Mask R-CNN + panoptic head) is difficult to export to
ONNX or TensorFlow because of:
  - Dynamic/variable-length instance outputs
  - Non-maximum suppression custom ops
  - Detectron2 ``ImageList`` preprocessing

Therefore the ONNX / TorchScript export targets the **backbone + FPN** sub-network only.
This produces multi-scale feature maps suitable for downstream tasks (feature extraction,
custom heads, distillation, etc.).  The plain ``.pth`` export saves *all* weights and can
be loaded back with :func:`cups.model.panoptic_cascade_mask_r_cnn_from_checkpoint`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logging.basicConfig(format="%(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Backbone wrapper – makes the backbone+FPN ONNX/TorchScript-exportable
# ---------------------------------------------------------------------------


class BackboneWrapper(nn.Module):
    """Wraps a Detectron2 backbone + FPN so that it can be traced/exported.

    The wrapped module accepts a single float32 image batch with values in
    **[0, 1]** and shape ``(N, C, H, W)``.  Internally it applies the same
    pixel-mean/std normalisation that the full Detectron2 model would apply,
    then runs the backbone and FPN.

    Returns a tuple of feature tensors ordered by FPN level
    (``p2``, ``p3``, ``p4``, ``p5``, ``p6``).
    """

    def __init__(self, detectron2_model: nn.Module) -> None:
        super().__init__()
        # Store backbone and FPN
        self.backbone = detectron2_model.backbone

        # Pixel statistics used by Detectron2 (BGR order, 0–255 scale)
        pixel_mean = torch.tensor(detectron2_model.pixel_mean).view(1, -1, 1, 1)
        pixel_std = torch.tensor(detectron2_model.pixel_std).view(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def preprocess(self, images: Tensor) -> Tensor:
        """Normalise images from [0, 1] float to Detectron2 convention.

        Detectron2 expects pixels in BGR order on a **0–255** scale (after
        subtracting ``pixel_mean`` and dividing by ``pixel_std``).  Images
        supplied to this wrapper are expected to be **RGB, [0, 1] float**.
        """
        # Scale to 0–255 and flip channels RGB→BGR
        x = images * 255.0
        x = x[:, [2, 1, 0], :, :]  # RGB → BGR
        # Normalise
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    def forward(self, images: Tensor) -> Tuple[Tensor, ...]:
        """Run backbone + FPN.

        Args:
            images: Float32 tensor ``(N, 3, H, W)`` with pixel values in
                ``[0, 1]`` in **RGB** order.

        Returns:
            Tuple of FPN feature maps ordered from finest (p2) to coarsest
            (p6).  All tensors have shape ``(N, 256, H_i, W_i)``.
        """
        x = self.preprocess(images)
        features: Dict[str, Tensor] = self.backbone(x)
        # Return features in a fixed order so ONNX output names are stable
        return tuple(features[k] for k in sorted(features.keys()))


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_path: str, device: str) -> Tuple[nn.Module, int, int]:
    """Load a CUPS PyTorch Lightning checkpoint and return the Detectron2 model.

    Args:
        checkpoint_path: Path to the ``.ckpt`` file produced by training.
        device: Torch device string (``"cpu"`` or ``"cuda"``).

    Returns:
        model: Detectron2 ``PanopticFPN`` model in eval mode.
        num_things: Number of thing (instance) pseudo-classes.
        num_stuffs: Number of stuff (semantic) pseudo-classes.
    """
    from cups.model import panoptic_cascade_mask_r_cnn_from_checkpoint

    log.info("Loading checkpoint from %s …", checkpoint_path)
    model, num_things, num_stuffs = panoptic_cascade_mask_r_cnn_from_checkpoint(
        path=checkpoint_path,
        device=device,
    )
    model = model.to(device).eval()
    log.info("Checkpoint loaded.  Things: %d  Stuffs: %d", num_things, num_stuffs)
    return model, num_things, num_stuffs


def export_pth(
    checkpoint_path: str,
    output_path: str,
    device: str = "cpu",
) -> None:
    """Extract the model state-dict from a Lightning ``.ckpt`` and save it as a
    plain ``.pth`` file that can be loaded with :func:`torch.load` without a
    PyTorch Lightning installation.

    The saved file is a ``dict`` with keys:
    * ``"state_dict"``      – ``OrderedDict`` of all model weights
    * ``"num_things"``      – number of thing pseudo-classes
    * ``"num_stuffs"``      – number of stuff pseudo-classes

    Args:
        checkpoint_path: Path to the source ``.ckpt`` file.
        output_path: Destination ``.pth`` file path.
        device: Torch device string used during loading.
    """
    model, num_things, num_stuffs = load_checkpoint(checkpoint_path, device)
    payload = {
        "state_dict": model.state_dict(),
        "num_things": num_things,
        "num_stuffs": num_stuffs,
    }
    torch.save(payload, output_path)
    log.info("State-dict saved to %s", output_path)


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    input_height: int = 512,
    input_width: int = 1024,
    device: str = "cpu",
    opset_version: int = 16,
) -> None:
    """Export the backbone + FPN to an ONNX graph.

    The exported model accepts a single ``float32`` tensor
    ``(1, 3, H, W)`` with pixel values in ``[0, 1]``.  The five FPN
    outputs ``(p2 … p6)`` are returned as separate ONNX outputs.

    Args:
        checkpoint_path: Path to the source ``.ckpt`` file.
        output_path: Destination ``.onnx`` file path.
        input_height: Height of the dummy input used for tracing.
        input_width: Width of the dummy input used for tracing.
        device: Torch device string used during export.
        opset_version: ONNX opset to target.  Defaults to 16.
    """
    try:
        import onnx
    except ImportError:
        log.error(
            "The 'onnx' package is required for ONNX export.  "
            "Install it with:  pip install onnx"
        )
        sys.exit(1)

    model, _, _ = load_checkpoint(checkpoint_path, device)
    wrapper = BackboneWrapper(model).to(device).eval()

    dummy_input = torch.zeros(1, 3, input_height, input_width, device=device)

    # Determine output names from a forward pass
    with torch.no_grad():
        sample_outputs = wrapper(dummy_input)
    num_outputs = len(sample_outputs)
    output_names = [f"p{i + 2}" for i in range(num_outputs)]

    log.info(
        "Exporting backbone to ONNX (opset %d, input %dx%d) …",
        opset_version,
        input_height,
        input_width,
    )
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes={
                "images": {0: "batch", 2: "height", 3: "width"},
                **{name: {0: "batch"} for name in output_names},
            },
        )
    log.info("ONNX model saved to %s", output_path)

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    log.info("ONNX graph validated successfully.")


def export_torchscript(
    checkpoint_path: str,
    output_path: str,
    input_height: int = 512,
    input_width: int = 1024,
    device: str = "cpu",
) -> None:
    """Export the backbone + FPN to a TorchScript traced model (``.pt``).

    The exported model has the same interface as :class:`BackboneWrapper`.

    Args:
        checkpoint_path: Path to the source ``.ckpt`` file.
        output_path: Destination ``.pt`` file path.
        input_height: Height of the dummy input used for tracing.
        input_width: Width of the dummy input used for tracing.
        device: Torch device string used during export.
    """
    model, _, _ = load_checkpoint(checkpoint_path, device)
    wrapper = BackboneWrapper(model).to(device).eval()

    dummy_input = torch.zeros(1, 3, input_height, input_width, device=device)

    log.info("Tracing backbone to TorchScript (input %dx%d) …", input_height, input_width)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy_input)
    traced.save(output_path)
    log.info("TorchScript model saved to %s", output_path)


def onnx_to_tensorflow(onnx_path: str, output_dir: str) -> None:
    """Convert an ONNX model to a TensorFlow SavedModel using ``onnx2tf``.

    Requires ``onnx2tf`` to be installed:  ``pip install onnx2tf``

    Args:
        onnx_path: Path to the source ``.onnx`` file.
        output_dir: Directory where the TensorFlow SavedModel will be written.
    """
    try:
        import onnx2tf
    except ImportError:
        log.error(
            "The 'onnx2tf' package is required for TensorFlow conversion.  "
            "Install it with:  pip install onnx2tf"
        )
        sys.exit(1)

    log.info("Converting ONNX model to TensorFlow SavedModel …")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_dir,
        non_verbose=False,
    )
    log.info("TensorFlow SavedModel written to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a CUPS .ckpt checkpoint to ONNX, TorchScript, or a plain .pth state-dict.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the input .ckpt checkpoint (PyTorch Lightning format).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output file (.pth / .onnx / .pt).",
    )
    parser.add_argument(
        "--format",
        choices=["pth", "onnx", "torchscript"],
        required=True,
        help=(
            "Output format: "
            "'pth' = plain state-dict, "
            "'onnx' = ONNX backbone, "
            "'torchscript' = TorchScript backbone."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Input image height used for tracing (ONNX / TorchScript only).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Input image width used for tracing (ONNX / TorchScript only).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for model loading and tracing.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=16,
        help="ONNX opset version (ONNX export only).",
    )
    parser.add_argument(
        "--to_tf",
        action="store_true",
        help=(
            "After exporting to ONNX, also convert to a TensorFlow SavedModel "
            "using onnx2tf.  Requires: pip install onnx2tf."
        ),
    )
    parser.add_argument(
        "--tf_output_dir",
        default=None,
        help=(
            "Directory for the TensorFlow SavedModel output.  "
            "Defaults to '<output_stem>_tf_savedmodel'."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry point for the conversion CLI."""
    args = _parse_args(argv)

    if not os.path.isfile(args.checkpoint):
        log.error("Checkpoint file not found: %s", args.checkpoint)
        sys.exit(1)

    if args.format == "pth":
        export_pth(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            device=args.device,
        )

    elif args.format == "onnx":
        export_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            input_height=args.height,
            input_width=args.width,
            device=args.device,
            opset_version=args.opset,
        )
        if args.to_tf:
            tf_dir = args.tf_output_dir or os.path.splitext(args.output)[0] + "_tf_savedmodel"
            onnx_to_tensorflow(onnx_path=args.output, output_dir=tf_dir)

    elif args.format == "torchscript":
        export_torchscript(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            input_height=args.height,
            input_width=args.width,
            device=args.device,
        )

    log.info("Conversion complete.")


if __name__ == "__main__":
    main()
