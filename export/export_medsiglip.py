"""
LaborView AI - MedSigLIP Edge Export
Export trained MedSigLIP model to ONNX/CoreML/TFLite for mobile deployment
"""

# /// script
# dependencies = [
#   "torch>=2.0.0",
#   "transformers>=4.50.0",
#   "onnx>=1.14.0",
#   "onnxscript>=0.1.0",
#   "onnxruntime>=1.16.0",
#   "huggingface_hub>=0.20.0",
#   "numpy>=1.24.0",
#   "pillow>=10.0.0",
#   "coremltools>=7.0",
# ]
# ///

import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


@dataclass
class Config:
    # Model
    encoder_pretrained: str = "google/medsiglip-448"
    encoder_hidden_dim: int = 1152
    projection_dim: int = 256
    num_plane_classes: int = 2
    num_seg_classes: int = 3
    image_size: int = 448

    # Export
    hub_model_id: str = "samwell/laborview-medsiglip"
    output_dir: Path = Path("./exports")
    opset_version: int = 17


class SegmentationDecoder(nn.Module):
    """Decoder for upsampling vision features to segmentation mask"""

    def __init__(self, input_dim: int, num_classes: int, decoder_channels=[512, 256, 128, 64]):
        super().__init__()

        self.input_proj = nn.Conv2d(input_dim, decoder_channels[0], 1)

        self.up_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]
        for out_ch in decoder_channels[1:]:
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            ))
            in_ch = out_ch

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[-1], 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(32, num_classes, 1)

    def forward(self, x, target_size=None):
        B = x.shape[0]

        if x.dim() == 3:
            num_patches = x.shape[1]
            H = W = int(num_patches ** 0.5)
            x = x.transpose(1, 2).reshape(B, -1, H, W)

        x = self.input_proj(x)

        for block in self.up_blocks:
            x = block(x)

        x = self.final_up(x)
        x = self.classifier(x)

        if target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x


class LaborViewMedSigLIP(nn.Module):
    """LaborView model with MedSigLIP vision encoder"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        from transformers import AutoModel

        print(f"Loading MedSigLIP from {config.encoder_pretrained}...")
        self.encoder = AutoModel.from_pretrained(
            config.encoder_pretrained,
            trust_remote_code=True
        )

        if hasattr(self.encoder, 'vision_model'):
            self.vision_encoder = self.encoder.vision_model
        else:
            self.vision_encoder = self.encoder

        if hasattr(self.vision_encoder.config, 'hidden_size'):
            hidden_dim = self.vision_encoder.config.hidden_size
        else:
            hidden_dim = config.encoder_hidden_dim

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
            nn.GELU(),
            nn.Linear(config.projection_dim, config.projection_dim)
        )

        self.cls_head = nn.Linear(config.projection_dim, config.num_plane_classes)
        self.seg_decoder = SegmentationDecoder(hidden_dim, config.num_seg_classes)

    def forward(self, pixel_values):
        if hasattr(self, 'vision_encoder'):
            outputs = self.vision_encoder(pixel_values)
        else:
            outputs = self.encoder.get_image_features(pixel_values, return_dict=True)

        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'):
            hidden = outputs.pooler_output
        else:
            hidden = outputs

        if hidden.dim() == 2:
            pooled = hidden
            B, D = hidden.shape
            seq = hidden.unsqueeze(1).expand(B, 32*32, D)
        elif hidden.dim() == 3:
            pooled = hidden.mean(dim=1)
            seq = hidden
        else:
            B, D, H, W = hidden.shape
            pooled = hidden.mean(dim=[2, 3])
            seq = hidden.flatten(2).transpose(1, 2)

        projected = self.projector(pooled)
        plane_logits = self.cls_head(projected)
        seg_masks = self.seg_decoder(seq, target_size=pixel_values.shape[-2:])

        return plane_logits, seg_masks


class LaborViewExportWrapper(nn.Module):
    """Wrapper for ONNX export - returns dict-like outputs"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        plane_logits, seg_masks = self.model(pixel_values)
        # Return segmentation probabilities and class prediction
        seg_probs = F.softmax(seg_masks, dim=1)
        plane_pred = plane_logits.argmax(dim=1)
        return seg_probs, plane_pred


def load_trained_model(config: Config):
    """Load trained model from HuggingFace Hub"""
    from huggingface_hub import hf_hub_download

    print(f"Downloading model from {config.hub_model_id}...")

    # Download checkpoint
    checkpoint_path = hf_hub_download(
        repo_id=config.hub_model_id,
        filename="best.pt"
    )

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Create model
    model = LaborViewMedSigLIP(config)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model loaded successfully!")
    if "val_loss" in checkpoint:
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    if "val_iou" in checkpoint:
        print(f"  Val IoU: {checkpoint['val_iou']:.4f}")

    return model


def export_to_onnx(model, config: Config, quantize: bool = True):
    """Export model to ONNX format"""
    import onnx
    import os

    config.output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    wrapper = LaborViewExportWrapper(model)
    wrapper.eval()

    # Create dummy input on CPU
    dummy_input = torch.randn(1, 3, config.image_size, config.image_size)

    # Move model to CPU for export
    wrapper = wrapper.cpu()

    # Export paths
    onnx_path = config.output_dir / "laborview_medsiglip.onnx"
    onnx_quant_path = config.output_dir / "laborview_medsiglip_int8.onnx"

    print(f"Exporting to ONNX: {onnx_path}")

    # Export using dynamo exporter (required for scaled_dot_product_attention)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['seg_probs', 'plane_pred']
        )

    # Note: Using onnxruntime 1.17+ on iOS which supports IR version 10
    onnx_model = onnx.load(str(onnx_path))
    print(f"IR version: {onnx_model.ir_version}")
    print(f"Opset version: {onnx_model.opset_import[0].version}")

    # Verify ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model verified successfully!")

    # Get model size
    onnx_size = onnx_path.stat().st_size / (1024 * 1024 * 1024)
    print(f"ONNX model size: {onnx_size:.2f} GB")

    # Quantize to INT8 (with error handling)
    onnx_quant_path_result = None
    if quantize and onnx_size > 0.001:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            print(f"Quantizing to INT8: {onnx_quant_path}")
            quantize_dynamic(
                str(onnx_path),
                str(onnx_quant_path),
                weight_type=QuantType.QInt8
            )

            quant_size = onnx_quant_path.stat().st_size / (1024 * 1024 * 1024)
            print(f"Quantized model size: {quant_size:.2f} GB")
            print(f"Size reduction: {(1 - quant_size/onnx_size) * 100:.1f}%")
            onnx_quant_path_result = onnx_quant_path
        except Exception as e:
            print(f"Quantization failed: {e}")
            print("Continuing with FP32 model only")

    return onnx_path, onnx_quant_path_result


def export_to_coreml(model, config: Config):
    """Export model to CoreML format for iOS with Neural Engine optimization"""
    try:
        import coremltools as ct
        from coremltools.models.neural_network import quantization_utils
    except ImportError:
        print("coremltools not installed. Skipping CoreML export.")
        print("Install with: pip install coremltools")
        return None

    config.output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model = model.cpu()  # Move to CPU for export
    wrapper = LaborViewExportWrapper(model)
    wrapper.eval()

    # Trace the model
    print("Tracing model for CoreML...")
    dummy_input = torch.randn(1, 3, config.image_size, config.image_size)

    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, dummy_input)

    coreml_path = config.output_dir / "laborview_medsiglip.mlpackage"
    coreml_fp16_path = config.output_dir / "laborview_medsiglip_fp16.mlpackage"

    print(f"Converting to CoreML: {coreml_path}")

    try:
        # Convert to CoreML with FP32 first
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="pixel_values",
                    shape=(1, 3, config.image_size, config.image_size),
                    dtype=np.float32
                )
            ],
            outputs=[
                ct.TensorType(name="seg_probs"),
                ct.TensorType(name="plane_pred")
            ],
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.ALL  # Use Neural Engine + GPU + CPU
        )

        # Add metadata
        mlmodel.author = "LaborView AI"
        mlmodel.short_description = "MedSigLIP ultrasound segmentation for labor monitoring"
        mlmodel.version = "1.0"

        # Save FP32 version
        mlmodel.save(str(coreml_path))
        print(f"CoreML FP32 model saved: {coreml_path}")

        # Create FP16 version for smaller size and faster inference
        print("Creating FP16 quantized version...")
        mlmodel_fp16 = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="pixel_values",
                    shape=(1, 3, config.image_size, config.image_size),
                    dtype=np.float32
                )
            ],
            outputs=[
                ct.TensorType(name="seg_probs"),
                ct.TensorType(name="plane_pred")
            ],
            minimum_deployment_target=ct.target.iOS16,
            compute_units=ct.ComputeUnit.ALL,
            compute_precision=ct.precision.FLOAT16
        )

        mlmodel_fp16.author = "LaborView AI"
        mlmodel_fp16.short_description = "MedSigLIP ultrasound segmentation (FP16)"
        mlmodel_fp16.version = "1.0"
        mlmodel_fp16.save(str(coreml_fp16_path))
        print(f"CoreML FP16 model saved: {coreml_fp16_path}")

        return coreml_path

    except Exception as e:
        print(f"CoreML export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_onnx_model(onnx_path: Path, config: Config):
    """Verify ONNX model with sample inference"""
    import onnxruntime as ort

    print(f"\nVerifying ONNX model: {onnx_path}")

    # Create session
    session = ort.InferenceSession(str(onnx_path))

    # Get input/output info
    input_info = session.get_inputs()[0]
    print(f"Input: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")

    for output in session.get_outputs():
        print(f"Output: {output.name}, shape: {output.shape}, type: {output.type}")

    # Run inference
    dummy_input = np.random.randn(1, 3, config.image_size, config.image_size).astype(np.float32)

    outputs = session.run(None, {"pixel_values": dummy_input})

    seg_probs, plane_pred = outputs
    print(f"\nTest inference:")
    print(f"  Segmentation output shape: {seg_probs.shape}")
    print(f"  Plane prediction: {plane_pred}")

    # Measure inference time
    import time
    times = []
    for _ in range(10):
        start = time.time()
        session.run(None, {"pixel_values": dummy_input})
        times.append(time.time() - start)

    avg_time = np.mean(times) * 1000
    print(f"  Average inference time: {avg_time:.1f} ms")

    return True


def main():
    parser = argparse.ArgumentParser(description="Export MedSigLIP for edge deployment")
    parser.add_argument("--output-dir", type=str, default="./exports", help="Output directory")
    parser.add_argument("--quantize", action="store_true", default=True, help="Quantize to INT8")
    parser.add_argument("--coreml", action="store_true", default=True, help="Export to CoreML")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify exported model")
    args = parser.parse_args()

    config = Config()
    config.output_dir = Path(args.output_dir)

    # Load trained model
    model = load_trained_model(config)
    model.eval()

    # Export to ONNX
    onnx_path, onnx_quant_path = export_to_onnx(model, config, quantize=args.quantize)

    # Verify
    if args.verify and onnx_path:
        verify_onnx_model(onnx_path, config)
        if onnx_quant_path:
            verify_onnx_model(onnx_quant_path, config)

    # Export to CoreML
    if args.coreml:
        export_to_coreml(model, config)

    print("\n" + "="*50)
    print("Export Summary:")
    print("="*50)

    for f in config.output_dir.glob("*"):
        size_mb = f.stat().st_size / (1024 * 1024)
        if size_mb > 1024:
            print(f"  {f.name}: {size_mb/1024:.2f} GB")
        else:
            print(f"  {f.name}: {size_mb:.1f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
