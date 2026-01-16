"""
LaborView AI - Demo & Inference Script
Interactive demo with Gradio for competition submission
"""

# /// script
# dependencies = [
#   "torch>=2.0.0",
#   "transformers>=4.40.0",
#   "gradio>=4.0.0",
#   "pillow>=10.0.0",
#   "numpy>=1.24.0",
#   "matplotlib>=3.7.0",
# ]
# ///

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from model import LaborViewModel, LaborViewModelEdge


# Color map for segmentation visualization
SEG_COLORS = {
    0: (0, 0, 0),        # Background - black
    1: (255, 0, 0),      # Pubic symphysis - red
    2: (0, 255, 0),      # Fetal head - green
}

PLANE_NAMES = [
    "Transperineal",
    "Transabdominal",
    "Oblique",
    "Sagittal",
    "Axial",
    "Other",
]


class LaborViewInference:
    """
    Inference wrapper for LaborView model.
    Handles preprocessing, inference, and postprocessing.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        edge_mode: bool = False,
        device: str = "auto",
    ):
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load model
        self.config = get_config(edge_mode=edge_mode)

        if edge_mode:
            self.model = LaborViewModelEdge()
        else:
            self.model = LaborViewModel(
                encoder_name=self.config.model.encoder_name,
                encoder_pretrained=self.config.model.encoder_pretrained,
                encoder_hidden_dim=self.config.model.encoder_hidden_dim,
                projection_dim=self.config.model.projection_dim,
                num_plane_classes=self.config.model.num_plane_classes,
                num_seg_classes=self.config.model.num_seg_classes,
                num_labor_params=self.config.model.num_labor_params,
                freeze_encoder=False,
            )

        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.image_size = self.config.data.image_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _load_checkpoint(self, path: str):
        """Load model weights from checkpoint"""
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu")

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        # Resize
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # To tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        # Normalize
        image = (image - self.mean) / self.std

        return image.to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict:
        """
        Run inference on a single image.

        Args:
            image: PIL Image

        Returns:
            Dict with predictions:
            - plane_class: Predicted plane class name
            - plane_confidence: Confidence score
            - plane_probs: All class probabilities
            - seg_mask: Segmentation mask (H, W)
            - aop: Angle of Progression (degrees)
            - hsd: Head-Symphysis Distance (mm)
        """
        # Preprocess
        x = self.preprocess(image)
        original_size = image.size[::-1]  # (H, W)

        # Forward pass
        outputs = self.model(x)

        # Classification
        plane_probs = F.softmax(outputs.plane_logits, dim=1)[0].cpu().numpy()
        plane_class_idx = plane_probs.argmax()
        plane_class = PLANE_NAMES[plane_class_idx]
        plane_confidence = plane_probs[plane_class_idx]

        # Segmentation - resize to original
        seg_logits = F.interpolate(
            outputs.seg_masks,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        seg_mask = seg_logits.argmax(dim=1)[0].cpu().numpy()

        # Labor parameters
        labor_params = outputs.labor_params[0].cpu().numpy()
        aop = float(labor_params[0])
        hsd = float(labor_params[1])

        return {
            "plane_class": plane_class,
            "plane_confidence": plane_confidence,
            "plane_probs": {PLANE_NAMES[i]: float(p) for i, p in enumerate(plane_probs)},
            "seg_mask": seg_mask,
            "aop": aop,
            "hsd": hsd,
        }

    def visualize_segmentation(
        self,
        image: Image.Image,
        seg_mask: np.ndarray,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Overlay segmentation mask on image"""
        # Create color mask
        h, w = seg_mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx, color in SEG_COLORS.items():
            color_mask[seg_mask == class_idx] = color

        # Resize image to mask size
        image_resized = image.resize((w, h), Image.BILINEAR)
        image_array = np.array(image_resized)

        # Blend
        blended = (1 - alpha) * image_array + alpha * color_mask
        blended = blended.astype(np.uint8)

        return Image.fromarray(blended)


def create_gradio_demo(inference: LaborViewInference):
    """Create Gradio demo interface"""
    import gradio as gr

    def predict_and_visualize(image):
        if image is None:
            return None, "", "", ""

        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Run inference
        results = inference.predict(image)

        # Create visualization
        seg_overlay = inference.visualize_segmentation(image, results["seg_mask"])

        # Format outputs
        plane_text = f"**{results['plane_class']}** ({results['plane_confidence']:.1%})"

        params_text = f"""
        **Angle of Progression (AoP):** {results['aop']:.1f}°
        **Head-Symphysis Distance (HSD):** {results['hsd']:.1f} mm
        """

        probs_text = "\n".join([
            f"- {name}: {prob:.1%}"
            for name, prob in sorted(
                results["plane_probs"].items(),
                key=lambda x: x[1],
                reverse=True
            )
        ])

        return seg_overlay, plane_text, params_text, probs_text

    # Create interface
    with gr.Blocks(title="LaborView AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🩺 LaborView AI
        ### AI-Powered Intrapartum Ultrasound Analysis

        Upload an ultrasound image to get:
        - **Standard Plane Classification** - Identifies the ultrasound view type
        - **Segmentation** - Highlights pubic symphysis (red) and fetal head (green)
        - **Labor Parameters** - Estimates Angle of Progression and Head-Symphysis Distance

        *Built with MedGemma for the MedGemma Impact Challenge*
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Ultrasound Image",
                    type="pil",
                    height=400,
                )
                predict_btn = gr.Button("Analyze", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Segmentation Overlay",
                    height=400,
                )

        with gr.Row():
            with gr.Column():
                plane_output = gr.Markdown(label="Plane Classification")
            with gr.Column():
                params_output = gr.Markdown(label="Labor Parameters")
            with gr.Column():
                probs_output = gr.Markdown(label="Class Probabilities")

        # Examples
        gr.Markdown("### Example Images")
        gr.Examples(
            examples=[
                ["examples/transperineal_1.png"],
                ["examples/transabdominal_1.png"],
            ] if Path("examples").exists() else [],
            inputs=input_image,
        )

        # Event handlers
        predict_btn.click(
            predict_and_visualize,
            inputs=[input_image],
            outputs=[output_image, plane_output, params_output, probs_output],
        )

        input_image.change(
            predict_and_visualize,
            inputs=[input_image],
            outputs=[output_image, plane_output, params_output, probs_output],
        )

        gr.Markdown("""
        ---
        **About LaborView AI**

        This tool assists healthcare providers in monitoring labor progress using
        transperineal ultrasound. It is designed to work in low-resource settings
        where expert sonographers may not be available.

        **Disclaimer:** This is a research prototype. Always consult qualified
        medical professionals for clinical decisions.

        *Part of the MedGemma Impact Challenge submission by Team [Your Team Name]*
        """)

    return demo


def main():
    """Run demo or single inference"""
    import argparse

    parser = argparse.ArgumentParser(description="LaborView AI Demo")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--edge", action="store_true",
                        help="Use edge-optimized model")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to single image for inference")
    parser.add_argument("--gradio", action="store_true",
                        help="Launch Gradio demo")
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860,
                        help="Gradio server port")
    args = parser.parse_args()

    # Create inference engine
    inference = LaborViewInference(
        checkpoint_path=args.checkpoint,
        edge_mode=args.edge,
    )

    if args.image:
        # Single image inference
        print(f"\nProcessing: {args.image}")
        image = Image.open(args.image)
        results = inference.predict(image)

        print(f"\n{'='*40}")
        print(f"Plane: {results['plane_class']} ({results['plane_confidence']:.1%})")
        print(f"AoP: {results['aop']:.1f}°")
        print(f"HSD: {results['hsd']:.1f} mm")
        print(f"{'='*40}")

        # Save segmentation visualization
        seg_vis = inference.visualize_segmentation(image, results["seg_mask"])
        output_path = Path(args.image).stem + "_segmented.png"
        seg_vis.save(output_path)
        print(f"\nSegmentation saved to: {output_path}")

    elif args.gradio:
        # Launch Gradio demo
        print("\nLaunching Gradio demo...")
        demo = create_gradio_demo(inference)
        demo.launch(
            server_port=args.port,
            share=args.share,
            show_error=True,
        )

    else:
        print("Use --image for single inference or --gradio for demo")
        print("Example: python demo.py --checkpoint best.pt --gradio")


if __name__ == "__main__":
    main()
