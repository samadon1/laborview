#!/usr/bin/env python3
"""
Export LaborView MobileViT model to ONNX for Flutter deployment
Much smaller than MedSigLIP (~22MB vs ~1.6GB)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download


class SegmentationDecoder(nn.Module):
    """Lightweight segmentation decoder for MobileViT"""

    def __init__(self, input_dim: int, num_classes: int, decoder_channels=[128, 64, 32]):
        super().__init__()
        self.input_proj = nn.Conv2d(input_dim, decoder_channels[0], 1)

        # Simpler up_blocks - just ConvTranspose2d + BatchNorm + GELU
        self.up_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]
        for out_ch in decoder_channels[1:]:
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ))
            in_ch = out_ch

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[-1], 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(16, num_classes, 1)

    def forward(self, x, target_size=None):
        B = x.shape[0]
        if x.dim() == 3:
            H = W = int(x.shape[1] ** 0.5)
            x = x.transpose(1, 2).reshape(B, -1, H, W)

        x = self.input_proj(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.final_up(x)
        x = self.classifier(x)

        if target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class LaborViewMobileViT(nn.Module):
    """LaborView model with MobileViT encoder"""

    def __init__(self):
        super().__init__()
        from transformers import MobileViTModel

        print("Loading MobileViT encoder...")
        self.encoder = MobileViTModel.from_pretrained("apple/mobilevit-small")

        # MobileViT-small hidden dim is 640
        hidden_dim = 640
        projection_dim = 256

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )

        self.cls_head = nn.Linear(projection_dim, 2)  # 2 plane classes
        self.seg_decoder = SegmentationDecoder(hidden_dim, 3, [128, 64, 32])  # 3 seg classes

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values)

        # MobileViT returns (B, D, H, W) for last_hidden_state
        hidden = outputs.last_hidden_state  # (B, 640, H, W)
        pooled = outputs.pooler_output  # (B, 640)

        # Reshape for segmentation decoder
        B, D, H, W = hidden.shape
        seq = hidden.flatten(2).transpose(1, 2)  # (B, H*W, D)

        projected = self.projector(pooled)
        plane_logits = self.cls_head(projected)
        seg_masks = self.seg_decoder(seq, target_size=pixel_values.shape[-2:])

        return plane_logits, seg_masks


class LaborViewMobileViTExport(nn.Module):
    """Wrapper for ONNX export"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        plane_logits, seg_masks = self.model(pixel_values)
        seg_probs = F.softmax(seg_masks, dim=1)
        plane_pred = plane_logits.argmax(dim=1)
        return seg_probs, plane_pred


def load_model():
    """Load trained MobileViT model"""
    print("Downloading checkpoint from HuggingFace...")
    checkpoint_path = hf_hub_download(
        repo_id="samwell/laborview-ultrasound",
        filename="best.pt"
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"Val IoU: {checkpoint['val_iou']:.4f}")

    model = LaborViewMobileViT()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def test_model(model, image_path=None, input_size=256):
    """Test model inference"""
    print(f"\n--- Testing Model (input_size={input_size}) ---")

    if image_path and os.path.exists(image_path):
        print(f"Loading image: {image_path}")
        img = Image.open(image_path).convert("RGB")
        img = img.resize((input_size, input_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
    else:
        print("Creating dummy test image...")
        arr = np.random.rand(input_size, input_size, 3).astype(np.float32)

    # Normalize (ImageNet style for MobileViT)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std

    arr = arr.transpose(2, 0, 1)
    tensor = torch.from_numpy(arr).unsqueeze(0).float()

    print(f"Input shape: {tensor.shape}")

    with torch.no_grad():
        plane_logits, seg_masks = model(tensor)
        seg_probs = F.softmax(seg_masks, dim=1)
        class_preds = seg_probs.argmax(dim=1).squeeze().numpy()

    print(f"Plane prediction: {plane_logits.argmax(dim=1).item()}")

    unique, counts = np.unique(class_preds, return_counts=True)
    print("Class distribution:")
    for cls, cnt in zip(unique, counts):
        pct = cnt / class_preds.size * 100
        print(f"  Class {cls}: {cnt} pixels ({pct:.1f}%)")

    return class_preds


def export_onnx(model, output_path, input_size=256):
    """Export to ONNX"""
    print(f"\n--- Exporting to ONNX ---")

    wrapper = LaborViewMobileViTExport(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    # Use legacy exporter for better compatibility
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=17,  # Higher opset for better compatibility
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['seg_probs', 'plane_pred'],
        dynamo=False,  # Use legacy exporter
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported to: {output_path}")
    print(f"Model size: {size_mb:.2f} MB")

    return output_path


def verify_onnx(onnx_path, input_size=256):
    """Verify ONNX model"""
    import onnxruntime as ort

    print(f"\n--- Verifying ONNX ---")

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    print("Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape}")

    print("Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape}")

    # Test inference
    dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    outputs = session.run(None, {"pixel_values": dummy})

    print(f"\nTest inference successful!")
    print(f"  seg_probs shape: {outputs[0].shape}")
    print(f"  plane_pred: {outputs[1]}")

    # Timing
    import time
    times = []
    for _ in range(10):
        start = time.time()
        session.run(None, {"pixel_values": dummy})
        times.append(time.time() - start)

    print(f"  Avg inference time: {np.mean(times)*1000:.1f} ms")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load model
    model = load_model()

    # Test with image if available
    test_image = "test_ultrasound.jpg"
    test_model(model, test_image if os.path.exists(test_image) else None, input_size=256)

    # Export to ONNX
    output_dir = Path("exports")
    output_dir.mkdir(exist_ok=True)

    onnx_path = output_dir / "laborview_mobilevit.onnx"
    export_onnx(model, str(onnx_path), input_size=256)

    # Verify
    verify_onnx(str(onnx_path), input_size=256)

    print("\n" + "="*50)
    print("Export complete!")
    print(f"MobileViT ONNX model: {onnx_path}")
    print("="*50)


if __name__ == "__main__":
    main()
