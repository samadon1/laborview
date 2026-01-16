"""
LaborView AI - Edge Export Utilities
Export trained models for mobile deployment (iOS/Android)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, Dict
import json
import shutil

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import LaborViewConfig, get_config
from model import LaborViewModel, LaborViewModelEdge


class EdgeExporter:
    """
    Export LaborView model for edge deployment.
    Supports ONNX, CoreML (iOS), and TFLite (Android).
    """

    def __init__(
        self,
        model: LaborViewModel,
        config: LaborViewConfig,
        output_dir: str = "./edge_models",
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Put model in eval mode
        self.model.eval()

    def export_all(self, input_size: Tuple[int, int] = (256, 256)):
        """Export to all supported formats"""
        print(f"\n{'='*60}")
        print("Exporting LaborView for Edge Deployment")
        print(f"{'='*60}")
        print(f"Output: {self.output_dir}")
        print(f"Input size: {input_size}")

        # Export ONNX first (required for other formats)
        onnx_path = self.export_onnx(input_size)

        # Export CoreML
        if self.config.edge.export_coreml:
            try:
                self.export_coreml(onnx_path, input_size)
            except Exception as e:
                print(f"CoreML export failed: {e}")

        # Export TFLite
        if self.config.edge.export_tflite:
            try:
                self.export_tflite(onnx_path, input_size)
            except Exception as e:
                print(f"TFLite export failed: {e}")

        # Save metadata
        self._save_metadata(input_size)

        print(f"\n{'='*60}")
        print("Export complete!")
        print(f"{'='*60}")

    def export_onnx(
        self,
        input_size: Tuple[int, int] = (256, 256),
        opset_version: int = 14,
    ) -> Path:
        """Export model to ONNX format"""
        print("\n[1/3] Exporting to ONNX...")

        onnx_path = self.output_dir / "laborview.onnx"
        dummy_input = torch.randn(1, 3, *input_size)

        # Wrap model for clean output
        wrapped_model = ONNXWrapper(self.model)

        torch.onnx.export(
            wrapped_model,
            dummy_input,
            str(onnx_path),
            input_names=["image"],
            output_names=["plane_logits", "seg_mask", "labor_params"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "plane_logits": {0: "batch_size"},
                "seg_mask": {0: "batch_size"},
                "labor_params": {0: "batch_size"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

        # Verify and optimize
        self._optimize_onnx(onnx_path)

        # Get model size
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX exported: {onnx_path} ({size_mb:.1f} MB)")

        return onnx_path

    def _optimize_onnx(self, onnx_path: Path):
        """Optimize ONNX model"""
        try:
            import onnx
            from onnxruntime.transformers import optimizer

            # Load and check
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)

            # Basic optimization
            optimized_path = self.output_dir / "laborview_optimized.onnx"
            optimized_model = optimizer.optimize_model(
                str(onnx_path),
                model_type="bert",  # Generic transformer optimization
                num_heads=0,
                hidden_size=0,
            )
            optimized_model.save_model_to_file(str(optimized_path))

            # Replace original with optimized
            shutil.move(str(optimized_path), str(onnx_path))
            print("  ONNX optimized successfully")

        except ImportError:
            print("  onnxruntime not available, skipping optimization")
        except Exception as e:
            print(f"  ONNX optimization failed: {e}")

    def export_coreml(
        self,
        onnx_path: Path,
        input_size: Tuple[int, int] = (256, 256),
    ) -> Path:
        """Export to CoreML format for iOS"""
        print("\n[2/3] Exporting to CoreML (iOS)...")

        try:
            import coremltools as ct
            from coremltools.models.neural_network import quantization_utils
        except ImportError:
            raise ImportError("coremltools not installed. Run: pip install coremltools")

        # Convert from ONNX
        mlmodel = ct.converters.onnx.convert(
            model=str(onnx_path),
            minimum_ios_deployment_target=self.config.edge.ios_min_version,
        )

        # Add metadata
        mlmodel.author = "LaborView AI"
        mlmodel.short_description = "Multi-task intrapartum ultrasound analysis"
        mlmodel.input_description["image"] = "Ultrasound image (RGB)"
        mlmodel.output_description["plane_logits"] = "Standard plane classification logits"
        mlmodel.output_description["seg_mask"] = "Segmentation mask"
        mlmodel.output_description["labor_params"] = "Labor parameters [AoP, HSD]"

        # Quantize if requested
        if self.config.edge.quantization in ["int8", "int4"]:
            print("  Applying quantization...")
            mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=8)

        # Save
        coreml_path = self.output_dir / "LaborView.mlpackage"
        mlmodel.save(str(coreml_path))

        size_mb = sum(f.stat().st_size for f in coreml_path.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"  CoreML exported: {coreml_path} ({size_mb:.1f} MB)")

        return coreml_path

    def export_tflite(
        self,
        onnx_path: Path,
        input_size: Tuple[int, int] = (256, 256),
    ) -> Path:
        """Export to TFLite format for Android"""
        print("\n[3/3] Exporting to TFLite (Android)...")

        try:
            import tensorflow as tf
            import onnx
            from onnx_tf.backend import prepare
        except ImportError:
            raise ImportError(
                "TensorFlow/onnx-tf not installed. Run: pip install tensorflow onnx-tf"
            )

        # Convert ONNX to TensorFlow SavedModel
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)

        tf_path = self.output_dir / "laborview_tf"
        tf_rep.export_graph(str(tf_path))

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))

        # Optimization options
        if self.config.edge.quantization == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif self.config.edge.quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # For full int8, would need representative dataset
        elif self.config.edge.quantization == "int4":
            # TFLite doesn't support int4 directly
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Convert
        tflite_model = converter.convert()

        # Save
        tflite_path = self.output_dir / "laborview.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        size_mb = tflite_path.stat().st_size / (1024 * 1024)
        print(f"  TFLite exported: {tflite_path} ({size_mb:.1f} MB)")

        # Clean up TF SavedModel
        shutil.rmtree(tf_path, ignore_errors=True)

        return tflite_path

    def _save_metadata(self, input_size: Tuple[int, int]):
        """Save model metadata for mobile apps"""
        metadata = {
            "model_name": "LaborView AI",
            "version": "1.0.0",
            "input_size": list(input_size),
            "input_format": "RGB",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "outputs": {
                "plane_logits": {
                    "shape": [1, self.config.model.num_plane_classes],
                    "labels": [
                        "transperineal",
                        "transabdominal",
                        "oblique",
                        "sagittal",
                        "axial",
                        "other",
                    ],
                },
                "seg_mask": {
                    "shape": [1, self.config.model.num_seg_classes, *input_size],
                    "labels": ["background", "pubic_symphysis", "fetal_head"],
                },
                "labor_params": {
                    "shape": [1, 2],
                    "labels": ["angle_of_progression", "head_symphysis_distance"],
                    "units": ["degrees", "mm"],
                },
            },
            "quantization": self.config.edge.quantization,
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n  Metadata saved: {metadata_path}")


class ONNXWrapper(nn.Module):
    """Wrapper for clean ONNX export"""

    def __init__(self, model: LaborViewModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        # Return tensors directly for ONNX
        return (
            outputs.plane_logits,
            outputs.seg_masks,
            outputs.labor_params,
        )


class ModelQuantizer:
    """
    Quantize model for reduced size and faster inference.
    """

    @staticmethod
    def quantize_dynamic(model: nn.Module) -> nn.Module:
        """Apply dynamic quantization (weights only)"""
        return torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8,
        )

    @staticmethod
    def quantize_static(
        model: nn.Module,
        calibration_loader,
        device: torch.device,
    ) -> nn.Module:
        """Apply static quantization (requires calibration data)"""
        model.eval()

        # Fuse modules
        model_fused = torch.quantization.fuse_modules(
            model,
            [["conv", "bn", "relu"]],  # Adjust based on model structure
            inplace=False,
        )

        # Prepare for calibration
        model_fused.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        model_prepared = torch.quantization.prepare(model_fused)

        # Calibrate
        print("Calibrating quantization...")
        with torch.no_grad():
            for batch in calibration_loader:
                x = batch["pixel_values"].to(device)
                model_prepared(x)

        # Convert
        model_quantized = torch.quantization.convert(model_prepared)

        return model_quantized


def benchmark_model(
    model: nn.Module,
    input_size: Tuple[int, int] = (256, 256),
    num_runs: int = 100,
    device: str = "cpu",
):
    """Benchmark model inference speed"""
    import time

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, *input_size).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Benchmark
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / num_runs) * 1000

    print(f"\nBenchmark Results ({device}):")
    print(f"  Average inference time: {avg_ms:.2f} ms")
    print(f"  Throughput: {1000/avg_ms:.1f} FPS")

    return avg_ms


def main():
    """Export model for edge deployment"""
    import argparse

    parser = argparse.ArgumentParser(description="Export LaborView for edge deployment")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="./edge_models",
                        help="Output directory")
    parser.add_argument("--input-size", type=int, default=256,
                        help="Input image size")
    parser.add_argument("--quantize", choices=["none", "fp16", "int8"],
                        default="int8", help="Quantization mode")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference benchmark")
    args = parser.parse_args()

    # Load config
    config = get_config(edge_mode=True)
    config.edge.quantization = args.quantize

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Create edge model
    model = LaborViewModelEdge()
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Benchmark before export
    if args.benchmark:
        print("\nBenchmarking PyTorch model...")
        benchmark_model(model, (args.input_size, args.input_size))

    # Export
    exporter = EdgeExporter(model, config, args.output_dir)
    exporter.export_all((args.input_size, args.input_size))

    # Benchmark ONNX
    if args.benchmark:
        try:
            import onnxruntime as ort

            print("\nBenchmarking ONNX model...")
            onnx_path = Path(args.output_dir) / "laborview.onnx"
            session = ort.InferenceSession(str(onnx_path))

            import numpy as np
            import time

            dummy_input = np.random.randn(1, 3, args.input_size, args.input_size).astype(np.float32)

            # Warmup
            for _ in range(10):
                session.run(None, {"image": dummy_input})

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                session.run(None, {"image": dummy_input})
            elapsed = time.perf_counter() - start

            print(f"  ONNX inference: {elapsed/100*1000:.2f} ms")
            print(f"  ONNX throughput: {100/elapsed:.1f} FPS")

        except ImportError:
            print("onnxruntime not available for benchmarking")


if __name__ == "__main__":
    main()
