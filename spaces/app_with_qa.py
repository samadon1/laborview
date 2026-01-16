"""
LaborView AI - HuggingFace Space with MedGemma Q&A
MedSigLIP for segmentation + MedGemma 1.5 for clinical Q&A
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from huggingface_hub import hf_hub_download


# ============================================================================
# MedSigLIP Model (Segmentation)
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
        self.encoder = AutoModel.from_pretrained(config.encoder_pretrained, trust_remote_code=True)
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
# LaborView Inference (Segmentation)
# ============================================================================

class LaborViewInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Segmentation device: {self.device}")
        self.config = Config()
        self._load_model()
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.last_results = None  # Store last analysis for Q&A context

    def _load_model(self):
        print(f"Loading segmentation model from {self.config.hub_model_id}...")
        checkpoint_path = hf_hub_download(repo_id=self.config.hub_model_id, filename="best.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = LaborViewMedSigLIP(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device).eval()
        print(f"Segmentation model loaded! IoU: {checkpoint.get('val_iou', 'N/A')}")

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

        results = {
            "plane_class": plane_names[plane_idx],
            "plane_confidence": float(plane_probs[plane_idx]),
            "seg_mask": seg_mask,
            "aop": aop, "aop_interpretation": interpret_aop(aop),
            "hsd": hsd, "hsd_interpretation": interpret_hsd(hsd, h),
            "head_circumference": hc, "head_area": area,
            "labor_progress": progress, "recommendation": rec,
        }
        self.last_results = results  # Store for Q&A
        return results

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

    def get_context_for_qa(self):
        """Generate context string for Q&A from last analysis"""
        if self.last_results is None:
            return "No ultrasound analysis has been performed yet."

        r = self.last_results
        context = f"""
Current ultrasound analysis results:
- Plane classification: {r['plane_class']} (confidence: {r['plane_confidence']:.1%})
- Angle of Progression (AoP): {f"{r['aop']:.1f} degrees" if r['aop'] else "Not measured"}
  Interpretation: {r['aop_interpretation']}
- Head-Symphysis Distance (HSD): {f"{r['hsd']:.1f} pixels" if r['hsd'] else "Not measured"}
  Interpretation: {r['hsd_interpretation']}
- Head Circumference: {f"{r['head_circumference']:.1f} pixels" if r['head_circumference'] else "Not measured"}
- Labor Progress Assessment: {r['labor_progress'].upper()}
- Recommendation: {r['recommendation']}
"""
        return context


# ============================================================================
# MedGemma Q&A
# ============================================================================

class MedGemmaQA:
    def __init__(self, model_id: str = "google/medgemma-4b-it"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Lazy load MedGemma model"""
        if self.model is not None:
            return

        print(f"Loading MedGemma from {self.model_id}...")
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        print("MedGemma loaded!")

    @torch.no_grad()
    def answer(self, question: str, context: str, image: Optional[Image.Image] = None) -> str:
        """Answer a clinical question with context from the analysis"""
        self.load_model()

        # Build the prompt
        system_prompt = """You are a medical AI assistant specializing in obstetric ultrasound interpretation.
You are helping analyze transperineal ultrasound images for labor monitoring.
Answer questions based on the provided analysis results. Be concise and clinically relevant.
Always remind users that AI analysis should be confirmed by qualified healthcare professionals."""

        user_prompt = f"""
{context}

Question: {question}

Please provide a helpful, clinically-informed response based on the analysis above.
"""

        # Prepare input - text only for now (context-based Q&A)
        messages = [
            {"role": "user", "content": user_prompt}
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # Decode response
        response = self.processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()


# ============================================================================
# Gradio App
# ============================================================================

print("Loading LaborView AI...")
seg_model = LaborViewInference()

# MedGemma is loaded lazily on first Q&A
qa_model = None


def get_qa_model():
    global qa_model
    if qa_model is None:
        qa_model = MedGemmaQA()
    return qa_model


def analyze(image):
    if image is None:
        return None, "", "", ""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    r = seg_model.predict(image)
    vis = seg_model.visualize(image, r["seg_mask"])

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


def ask_question(question: str, history: List[List[str]]) -> Tuple[List[List[str]], str]:
    """Handle Q&A with MedGemma"""
    if not question.strip():
        return history, ""

    # Get context from last analysis
    context = seg_model.get_context_for_qa()

    try:
        qa = get_qa_model()
        response = qa.answer(question, context)
    except Exception as e:
        response = f"Error: Could not generate response. {str(e)}\n\nPlease ensure MedGemma access is granted at huggingface.co/google/medgemma-4b-it"

    history = history + [[question, response]]
    return history, ""


def clear_chat():
    return [], ""


# Build the interface
with gr.Blocks(title="LaborView AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # LaborView AI - Intrapartum Ultrasound Analysis
    **MedSigLIP for segmentation + MedGemma for clinical Q&A**
    """)

    with gr.Tabs():
        # Tab 1: Image Analysis
        with gr.TabItem("Image Analysis"):
            gr.Markdown("""
            Upload a transperineal ultrasound image for automated analysis:
            - **Segmentation** - Pubic symphysis (red) and fetal head (green)
            - **Clinical Metrics** - AoP, HSD, Head Circumference
            - **Assessment** - AI-generated labor progress evaluation
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

        # Tab 2: Clinical Q&A
        with gr.TabItem("Clinical Q&A"):
            gr.Markdown("""
            ### Ask MedGemma about your analysis

            After analyzing an ultrasound image, you can ask clinical questions about:
            - What the measurements mean
            - Labor progress interpretation
            - Clinical guidelines and thresholds
            - Next steps and recommendations

            **Note:** First analyze an image in the "Image Analysis" tab before asking questions.
            """)

            chatbot = gr.Chatbot(
                label="Clinical Q&A",
                height=400,
                show_label=False
            )

            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What does an AoP of 125 degrees indicate?",
                    scale=4
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1)

            clear_btn = gr.Button("Clear Chat", variant="secondary")

            # Example questions
            gr.Markdown("**Example questions:**")
            gr.Examples(
                examples=[
                    ["What does the Angle of Progression indicate about labor progress?"],
                    ["Is this AoP value normal for active labor?"],
                    ["What clinical interventions might be considered based on these results?"],
                    ["How often should ultrasound monitoring be repeated?"],
                    ["What are the limitations of AI-based labor assessment?"],
                ],
                inputs=question_input,
            )

            # Event handlers
            ask_btn.click(
                ask_question,
                [question_input, chatbot],
                [chatbot, question_input]
            )
            question_input.submit(
                ask_question,
                [question_input, chatbot],
                [chatbot, question_input]
            )
            clear_btn.click(clear_chat, [], [chatbot, question_input])

    gr.Markdown("""
    ---
    **Disclaimer:** Research prototype for MedGemma Impact Challenge. Not for clinical use.

    *Built with MedSigLIP + MedGemma from Google's Health AI Developer Foundation*
    """)


if __name__ == "__main__":
    demo.launch()
