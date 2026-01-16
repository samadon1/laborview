"""
LaborView AI - MedSigLIP Gradio Demo
Deployable as HuggingFace Space for the MedGemma Impact Challenge
"""

# /// script
# dependencies = [
#   "torch>=2.0.0",
#   "transformers>=4.50.0",
#   "gradio>=4.0.0",
#   "pillow>=10.0.0",
#   "numpy>=1.24.0",
#   "opencv-python-headless>=4.8.0",
#   "huggingface_hub>=0.20.0",
# ]
# ///

import os
import sys
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model Definition (must match training)
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


# ============================================================================
# Clinical Metrics (inline for Space deployment)
# ============================================================================

def extract_contours(mask: np.ndarray, class_id: int):
    """Extract the largest contour for a given class"""
    binary = (mask == class_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def get_lowest_point(contour: np.ndarray) -> Tuple[int, int]:
    """Get the lowest point of a contour"""
    lowest_idx = contour[:, :, 1].argmax()
    return tuple(contour[lowest_idx, 0])


def fit_line_to_symphysis(contour: np.ndarray):
    """Fit a line to the symphysis contour"""
    points = contour.reshape(-1, 2).astype(np.float32)
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()
    return np.array([x0, y0]), np.array([vx, vy])


def compute_aop(symphysis_contour, head_contour, image_height: int) -> Optional[float]:
    """Compute Angle of Progression"""
    try:
        symphysis_point, symphysis_dir = fit_line_to_symphysis(symphysis_contour)
        symphysis_inferior = get_lowest_point(symphysis_contour)
        head_leading = get_lowest_point(head_contour)

        head_vector = np.array([
            head_leading[0] - symphysis_inferior[0],
            head_leading[1] - symphysis_inferior[1]
        ])

        symphysis_dir_norm = symphysis_dir / (np.linalg.norm(symphysis_dir) + 1e-6)
        head_vector_norm = head_vector / (np.linalg.norm(head_vector) + 1e-6)

        dot_product = np.clip(np.dot(symphysis_dir_norm, head_vector_norm), -1.0, 1.0)
        angle_rad = np.arccos(abs(dot_product))
        angle_deg = np.degrees(angle_rad)

        aop = angle_deg
        if head_leading[1] > symphysis_inferior[1]:
            aop = 90 + (90 - angle_deg)

        return aop
    except:
        return None


def compute_hsd(symphysis_contour, head_contour) -> Optional[float]:
    """Compute Head-Symphysis Distance"""
    try:
        symphysis_inferior = get_lowest_point(symphysis_contour)
        head_points = head_contour.reshape(-1, 2)
        distances = np.sqrt(
            (head_points[:, 0] - symphysis_inferior[0])**2 +
            (head_points[:, 1] - symphysis_inferior[1])**2
        )
        return float(distances.min())
    except:
        return None


def compute_head_circumference(head_contour) -> Tuple[Optional[float], Optional[float]]:
    """Compute head circumference and area"""
    try:
        circumference = cv2.arcLength(head_contour, closed=True)
        area = cv2.contourArea(head_contour)
        return circumference, area
    except:
        return None, None


def interpret_aop(aop: Optional[float]) -> str:
    """Clinical interpretation of AoP"""
    if aop is None:
        return "Unable to measure"
    if aop < 100:
        return "Very early labor - high fetal station"
    elif aop < 110:
        return "Early labor - fetal head not engaged"
    elif aop < 120:
        return "Active labor - head descending"
    elif aop < 130:
        return "Good progress - head well descended"
    elif aop < 140:
        return "Advanced labor - delivery approaching"
    else:
        return "Late labor - delivery imminent"


def interpret_hsd(hsd: Optional[float], image_height: int) -> str:
    """Clinical interpretation of HSD"""
    if hsd is None:
        return "Unable to measure"
    hsd_ratio = hsd / image_height
    if hsd_ratio > 0.3:
        return "Large distance - head not engaged"
    elif hsd_ratio > 0.2:
        return "Moderate distance - early descent"
    elif hsd_ratio > 0.1:
        return "Small distance - good descent"
    else:
        return "Minimal distance - head at symphysis level"


def assess_labor_progress(aop: Optional[float], hsd: Optional[float], image_height: int) -> Tuple[str, str]:
    """Overall labor progress assessment"""
    if aop is None and hsd is None:
        return "unknown", "Unable to assess - segmentation quality insufficient"

    score = 0
    if aop is not None:
        if aop >= 120:
            score += 2
        elif aop >= 110:
            score += 1
        else:
            score -= 1

    if hsd is not None:
        hsd_ratio = hsd / image_height
        if hsd_ratio < 0.15:
            score += 1
        elif hsd_ratio > 0.25:
            score -= 1

    if score >= 2:
        return "normal", "Labor progressing well. Continue routine monitoring."
    elif score >= 0:
        return "monitor", "Labor progressing. Continue close monitoring and reassess in 30-60 minutes."
    else:
        return "concern", "Slow progress noted. Consider clinical examination and evaluate need for intervention."


# ============================================================================
# Inference Class
# ============================================================================

class LaborViewInference:
    """Inference wrapper for LaborView MedSigLIP model"""

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.config = Config()
        self.model = None
        self._load_model()

        self.image_size = self.config.image_size
        # MedSigLIP uses different normalization
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

    def _load_model(self):
        """Load model from HuggingFace Hub"""
        from huggingface_hub import hf_hub_download

        print(f"Downloading model from {self.config.hub_model_id}...")

        checkpoint_path = hf_hub_download(
            repo_id=self.config.hub_model_id,
            filename="best.pt"
        )

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.model = LaborViewMedSigLIP(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully!")
        if "val_iou" in checkpoint:
            print(f"  Model IoU: {checkpoint['val_iou']:.4f}")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

        # Normalize to [-1, 1] for SigLIP
        image_tensor = (image_tensor - self.mean) / self.std

        return image_tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict:
        """Run inference on a single image"""
        x = self.preprocess(image)
        original_size = image.size[::-1]  # (H, W)

        plane_logits, seg_masks = self.model(x)

        # Plane classification
        plane_probs = F.softmax(plane_logits, dim=1)[0].cpu().numpy()
        plane_class_idx = plane_probs.argmax()
        plane_names = ["Transperineal", "Other"]
        plane_class = plane_names[plane_class_idx]
        plane_confidence = float(plane_probs[plane_class_idx])

        # Segmentation
        seg_logits = F.interpolate(
            seg_masks,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        seg_mask = seg_logits.argmax(dim=1)[0].cpu().numpy()

        # Clinical metrics
        h, w = seg_mask.shape
        symphysis_contour = extract_contours(seg_mask, 1)
        head_contour = extract_contours(seg_mask, 2)

        aop = None
        hsd = None
        hc = None
        head_area = None

        if symphysis_contour is not None and head_contour is not None:
            aop = compute_aop(symphysis_contour, head_contour, h)
            hsd = compute_hsd(symphysis_contour, head_contour)
            hc, head_area = compute_head_circumference(head_contour)

        aop_interp = interpret_aop(aop)
        hsd_interp = interpret_hsd(hsd, h)
        progress, recommendation = assess_labor_progress(aop, hsd, h)

        return {
            "plane_class": plane_class,
            "plane_confidence": plane_confidence,
            "seg_mask": seg_mask,
            "aop": aop,
            "aop_interpretation": aop_interp,
            "hsd": hsd,
            "hsd_interpretation": hsd_interp,
            "head_circumference": hc,
            "head_area": head_area,
            "labor_progress": progress,
            "recommendation": recommendation,
        }

    def visualize_segmentation(
        self,
        image: Image.Image,
        seg_mask: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Overlay segmentation mask on image"""
        h, w = seg_mask.shape

        # Color map
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        color_mask[seg_mask == 1] = [255, 100, 100]  # Symphysis - red/blue
        color_mask[seg_mask == 2] = [100, 255, 100]  # Fetal head - green

        image_resized = image.resize((w, h), Image.BILINEAR)
        image_array = np.array(image_resized)

        # Blend
        blended = (1 - alpha) * image_array + alpha * color_mask
        blended = blended.astype(np.uint8)

        # Draw contours
        symphysis_contour = extract_contours(seg_mask, 1)
        head_contour = extract_contours(seg_mask, 2)

        if symphysis_contour is not None:
            cv2.drawContours(blended, [symphysis_contour], -1, (255, 50, 50), 2)
        if head_contour is not None:
            cv2.drawContours(blended, [head_contour], -1, (50, 255, 50), 2)

        return Image.fromarray(blended)


# ============================================================================
# Gradio Interface
# ============================================================================

def create_gradio_demo(inference: LaborViewInference):
    """Create Gradio demo interface"""
    import gradio as gr

    def predict_and_visualize(image):
        if image is None:
            return None, "", "", "", ""

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        results = inference.predict(image)
        seg_overlay = inference.visualize_segmentation(image, results["seg_mask"])

        # Format outputs
        plane_text = f"**{results['plane_class']}** (Confidence: {results['plane_confidence']:.1%})"

        # Clinical metrics
        metrics_lines = []
        if results['aop'] is not None:
            metrics_lines.append(f"**Angle of Progression (AoP):** {results['aop']:.1f}")
            metrics_lines.append(f"   *{results['aop_interpretation']}*")
        else:
            metrics_lines.append("**Angle of Progression (AoP):** Unable to compute")

        metrics_lines.append("")

        if results['hsd'] is not None:
            metrics_lines.append(f"**Head-Symphysis Distance (HSD):** {results['hsd']:.1f} px")
            metrics_lines.append(f"   *{results['hsd_interpretation']}*")
        else:
            metrics_lines.append("**Head-Symphysis Distance (HSD):** Unable to compute")

        metrics_lines.append("")

        if results['head_circumference'] is not None:
            metrics_lines.append(f"**Head Circumference:** {results['head_circumference']:.1f} px")
        if results['head_area'] is not None:
            metrics_lines.append(f"**Head Area:** {results['head_area']:.0f} px")

        metrics_text = "\n".join(metrics_lines)

        # Progress assessment
        progress_color = {
            "normal": "green",
            "monitor": "orange",
            "concern": "red",
            "unknown": "gray"
        }.get(results['labor_progress'], "gray")

        progress_text = f"""
**Status:** <span style="color:{progress_color}; font-weight:bold;">{results['labor_progress'].upper()}</span>

{results['recommendation']}
"""

        return seg_overlay, plane_text, metrics_text, progress_text

    # Create interface
    with gr.Blocks(
        title="LaborView AI - MedSigLIP",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {max-width: 1200px !important}
        .status-normal {color: green; font-weight: bold}
        .status-monitor {color: orange; font-weight: bold}
        .status-concern {color: red; font-weight: bold}
        """
    ) as demo:
        gr.Markdown("""
        # LaborView AI - Intrapartum Ultrasound Analysis

        **AI-powered labor monitoring using MedSigLIP from MedGemma**

        Upload a transperineal ultrasound image to get:
        - **Segmentation** - Identifies pubic symphysis (red) and fetal head (green)
        - **Angle of Progression (AoP)** - Key labor progress indicator
        - **Head-Symphysis Distance (HSD)** - Distance measurement
        - **Clinical Assessment** - AI-generated progress evaluation

        ---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Ultrasound Image",
                    type="pil",
                    height=400,
                )
                predict_btn = gr.Button("Analyze Image", variant="primary", size="lg")

                gr.Markdown("""
                **Segmentation Legend:**
                - Red/Pink: Pubic Symphysis
                - Green: Fetal Head
                """)

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Segmentation Result",
                    height=400,
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Plane Classification")
                plane_output = gr.Markdown()

            with gr.Column():
                gr.Markdown("### Clinical Measurements")
                metrics_output = gr.Markdown()

            with gr.Column():
                gr.Markdown("### Labor Assessment")
                progress_output = gr.Markdown()

        # Event handlers
        predict_btn.click(
            predict_and_visualize,
            inputs=[input_image],
            outputs=[output_image, plane_output, metrics_output, progress_output],
        )

        input_image.change(
            predict_and_visualize,
            inputs=[input_image],
            outputs=[output_image, plane_output, metrics_output, progress_output],
        )

        gr.Markdown("""
        ---
        ### About LaborView AI

        This tool assists healthcare providers in monitoring labor progress using transperineal ultrasound.
        It is designed to support clinical decision-making in resource-limited settings.

        **Key Features:**
        - Built on **MedSigLIP** (878M parameters) from Google's MedGemma family
        - Fine-tuned on transperineal ultrasound images for labor monitoring
        - Computes clinically validated measurements (AoP, HSD)
        - Provides interpretive guidance based on established thresholds

        **Disclaimer:** This is a research prototype for the MedGemma Impact Challenge.
        Always consult qualified medical professionals for clinical decisions.

        ---
        *Built with MedGemma for the MedGemma Impact Challenge*
        """)

    return demo


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="LaborView AI MedSigLIP Demo")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--image", type=str, help="Path to single image for CLI inference")
    args = parser.parse_args()

    # Load model
    print("Initializing LaborView AI...")
    inference = LaborViewInference()

    if args.image:
        # CLI inference
        print(f"\nProcessing: {args.image}")
        image = Image.open(args.image)
        results = inference.predict(image)

        print(f"\n{'='*50}")
        print(f"Plane: {results['plane_class']} ({results['plane_confidence']:.1%})")
        print(f"\nClinical Metrics:")
        if results['aop']:
            print(f"  AoP: {results['aop']:.1f} - {results['aop_interpretation']}")
        if results['hsd']:
            print(f"  HSD: {results['hsd']:.1f} px - {results['hsd_interpretation']}")
        if results['head_circumference']:
            print(f"  HC: {results['head_circumference']:.1f} px")
        print(f"\nAssessment: {results['labor_progress'].upper()}")
        print(f"Recommendation: {results['recommendation']}")
        print(f"{'='*50}")

        # Save visualization
        seg_vis = inference.visualize_segmentation(image, results["seg_mask"])
        output_path = Path(args.image).stem + "_analyzed.png"
        seg_vis.save(output_path)
        print(f"\nVisualization saved to: {output_path}")

    else:
        # Launch Gradio
        print("\nLaunching Gradio demo...")
        demo = create_gradio_demo(inference)
        demo.launch(
            server_port=args.port,
            share=args.share,
            show_error=True,
        )


if __name__ == "__main__":
    main()
