"""
LaborView AI - HuggingFace Space Entry Point
MedSigLIP-powered intrapartum ultrasound analysis
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from huggingface_hub import hf_hub_download


# ============================================================================
# Model Definition
# ============================================================================

@dataclass
class Config:
    encoder_pretrained: str = "google/medsiglip-448"
    encoder_hidden_dim: int = 1152
    projection_dim: int = 256
    num_plane_classes: int = 2
    num_seg_classes: int = 3
    image_size: int = 448
    hub_model_id: str = "samwell/laborview-medsiglip"


class SegmentationDecoder(nn.Module):
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
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        from transformers import AutoModel
        import os
        token = os.environ.get("HF_TOKEN")
        self.encoder = AutoModel.from_pretrained(config.encoder_pretrained, trust_remote_code=True, token=token)
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
        outputs = self.vision_encoder(pixel_values) if hasattr(self, 'vision_encoder') else self.encoder.get_image_features(pixel_values, return_dict=True)
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


# ============================================================================
# Clinical Metrics
# ============================================================================

def extract_contours(mask: np.ndarray, class_id: int):
    binary = (mask == class_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def get_lowest_point(contour):
    lowest_idx = contour[:, :, 1].argmax()
    return tuple(contour[lowest_idx, 0])


def compute_aop(symphysis_contour, head_contour, image_height: int):
    try:
        points = symphysis_contour.reshape(-1, 2).astype(np.float32)
        line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line.flatten()
        symphysis_dir = np.array([vx, vy])
        symphysis_inferior = get_lowest_point(symphysis_contour)
        head_leading = get_lowest_point(head_contour)
        head_vector = np.array([head_leading[0] - symphysis_inferior[0], head_leading[1] - symphysis_inferior[1]])
        symphysis_dir_norm = symphysis_dir / (np.linalg.norm(symphysis_dir) + 1e-6)
        head_vector_norm = head_vector / (np.linalg.norm(head_vector) + 1e-6)
        dot_product = np.clip(np.dot(symphysis_dir_norm, head_vector_norm), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(abs(dot_product)))
        return 90 + (90 - angle_deg) if head_leading[1] > symphysis_inferior[1] else angle_deg
    except:
        return None


def compute_hsd(symphysis_contour, head_contour):
    try:
        symphysis_inferior = get_lowest_point(symphysis_contour)
        head_points = head_contour.reshape(-1, 2)
        distances = np.sqrt((head_points[:, 0] - symphysis_inferior[0])**2 + (head_points[:, 1] - symphysis_inferior[1])**2)
        return float(distances.min())
    except:
        return None


def compute_hc(head_contour):
    try:
        return cv2.arcLength(head_contour, closed=True), cv2.contourArea(head_contour)
    except:
        return None, None


def interpret_aop(aop):
    if aop is None: return "Unable to measure"
    if aop < 100: return "Very early labor - high fetal station"
    elif aop < 110: return "Early labor - fetal head not engaged"
    elif aop < 120: return "Active labor - head descending"
    elif aop < 130: return "Good progress - head well descended"
    elif aop < 140: return "Advanced labor - delivery approaching"
    else: return "Late labor - delivery imminent"


def interpret_hsd(hsd, image_height):
    if hsd is None: return "Unable to measure"
    ratio = hsd / image_height
    if ratio > 0.3: return "Large distance - head not engaged"
    elif ratio > 0.2: return "Moderate distance - early descent"
    elif ratio > 0.1: return "Small distance - good descent"
    else: return "Minimal distance - head at symphysis level"


def assess_progress(aop, hsd, image_height):
    if aop is None and hsd is None:
        return "unknown", "Unable to assess - segmentation quality insufficient"
    score = 0
    if aop is not None:
        score += 2 if aop >= 120 else (1 if aop >= 110 else -1)
    if hsd is not None:
        ratio = hsd / image_height
        score += 1 if ratio < 0.15 else (-1 if ratio > 0.25 else 0)
    if score >= 2:
        return "normal", "Labor progressing well. Continue routine monitoring."
    elif score >= 0:
        return "monitor", "Labor progressing. Continue close monitoring and reassess in 30-60 minutes."
    else:
        return "concern", "Slow progress noted. Consider clinical examination and evaluate need for intervention."


# ============================================================================
# Inference
# ============================================================================

class LaborViewInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.config = Config()
        self._load_model()
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

    def _load_model(self):
        print(f"Loading model from {self.config.hub_model_id}...")
        checkpoint_path = hf_hub_download(repo_id=self.config.hub_model_id, filename="best.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = LaborViewMedSigLIP(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device).eval()
        print(f"Model loaded! IoU: {checkpoint.get('val_iou', 'N/A')}")

    def preprocess(self, image: Image.Image):
        image = image.convert("RGB").resize((self.config.image_size, self.config.image_size), Image.BILINEAR)
        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return ((tensor - self.mean) / self.std).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image):
        x = self.preprocess(image)
        orig_size = image.size[::-1]
        plane_logits, seg_masks = self.model(x)
        plane_probs = F.softmax(plane_logits, dim=1)[0].cpu().numpy()
        plane_idx = plane_probs.argmax()
        plane_names = ["Transperineal", "Other"]
        seg = F.interpolate(seg_masks, size=orig_size, mode="bilinear", align_corners=False)
        seg_mask = seg.argmax(dim=1)[0].cpu().numpy()
        h, w = seg_mask.shape
        sym = extract_contours(seg_mask, 1)
        head = extract_contours(seg_mask, 2)
        aop = compute_aop(sym, head, h) if sym is not None and head is not None else None
        hsd = compute_hsd(sym, head) if sym is not None and head is not None else None
        hc, area = compute_hc(head) if head is not None else (None, None)
        progress, rec = assess_progress(aop, hsd, h)
        return {
            "plane_class": plane_names[plane_idx],
            "plane_confidence": float(plane_probs[plane_idx]),
            "seg_mask": seg_mask,
            "aop": aop, "aop_interpretation": interpret_aop(aop),
            "hsd": hsd, "hsd_interpretation": interpret_hsd(hsd, h),
            "head_circumference": hc, "head_area": area,
            "labor_progress": progress, "recommendation": rec,
        }

    def visualize(self, image: Image.Image, seg_mask: np.ndarray, alpha=0.5):
        h, w = seg_mask.shape
        color = np.zeros((h, w, 3), dtype=np.uint8)
        color[seg_mask == 1] = [255, 100, 100]
        color[seg_mask == 2] = [100, 255, 100]
        img_arr = np.array(image.resize((w, h), Image.BILINEAR))
        blended = ((1 - alpha) * img_arr + alpha * color).astype(np.uint8)
        sym = extract_contours(seg_mask, 1)
        head = extract_contours(seg_mask, 2)
        if sym is not None:
            cv2.drawContours(blended, [sym], -1, (255, 50, 50), 2)
        if head is not None:
            cv2.drawContours(blended, [head], -1, (50, 255, 50), 2)
        return Image.fromarray(blended)


# ============================================================================
# Gradio App
# ============================================================================

print("Loading LaborView AI...")
inference = LaborViewInference()


def analyze(image):
    if image is None:
        return None, "", "", ""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    r = inference.predict(image)
    vis = inference.visualize(image, r["seg_mask"])

    plane = f"**{r['plane_class']}** ({r['plane_confidence']:.1%})"

    metrics = []
    if r['aop']:
        metrics.append(f"**AoP:** {r['aop']:.1f} - *{r['aop_interpretation']}*")
    else:
        metrics.append("**AoP:** Unable to compute")
    if r['hsd']:
        metrics.append(f"**HSD:** {r['hsd']:.1f} px - *{r['hsd_interpretation']}*")
    else:
        metrics.append("**HSD:** Unable to compute")
    if r['head_circumference']:
        metrics.append(f"**HC:** {r['head_circumference']:.1f} px")

    colors = {"normal": "green", "monitor": "orange", "concern": "red", "unknown": "gray"}
    progress = f"**Status:** <span style='color:{colors[r['labor_progress']]};font-weight:bold'>{r['labor_progress'].upper()}</span>\n\n{r['recommendation']}"

    return vis, plane, "\n\n".join(metrics), progress


with gr.Blocks(title="LaborView AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # LaborView AI - Intrapartum Ultrasound Analysis
    **AI-powered labor monitoring using MedSigLIP from MedGemma**

    Upload a transperineal ultrasound image for automated analysis of:
    - **Segmentation** - Pubic symphysis (red) and fetal head (green)
    - **Angle of Progression (AoP)** - Key labor progress indicator
    - **Head-Symphysis Distance (HSD)** - Distance measurement
    - **Clinical Assessment** - AI-generated progress evaluation
    """)

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Upload Ultrasound", type="pil", height=400)
            btn = gr.Button("Analyze", variant="primary", size="lg")
        with gr.Column():
            img_out = gr.Image(label="Segmentation", height=400)

    with gr.Row():
        plane_out = gr.Markdown(label="Plane")
        metrics_out = gr.Markdown(label="Metrics")
        progress_out = gr.Markdown(label="Assessment")

    btn.click(analyze, [img_in], [img_out, plane_out, metrics_out, progress_out])
    img_in.change(analyze, [img_in], [img_out, plane_out, metrics_out, progress_out])

    gr.Markdown("""
    ---
    **Disclaimer:** Research prototype for MedGemma Impact Challenge. Not for clinical use.

    *Built with MedSigLIP (878M params) from Google's MedGemma family*
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
