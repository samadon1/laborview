#!/usr/bin/env python3
"""
Test script to verify the ONNX model works correctly.
This validates the model before deploying to Flutter.
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
import os

# Paths
MODEL_PATH = "exports/laborview_medsiglip.onnx"
INPUT_SIZE = 448
TEST_IMAGE = "test_ultrasound.jpg"

def preprocess_image(image_path: str = None) -> tuple[np.ndarray, Image.Image]:
    """Preprocess image for model input. Creates dummy image if no path provided."""
    if image_path and os.path.exists(image_path):
        print(f"Loading image: {image_path}")
        img = Image.open(image_path).convert("RGB")
        print(f"Original size: {img.size}")
        img_resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        arr = np.array(img_resized, dtype=np.float32) / 255.0
    else:
        # Create dummy test image (gray with some variation)
        print("Creating dummy test image...")
        arr = np.random.rand(INPUT_SIZE, INPUT_SIZE, 3).astype(np.float32) * 0.5 + 0.25
        img_resized = Image.fromarray((arr * 255).astype(np.uint8))

    # Normalize to [-1, 1] for SigLIP
    arr = (arr - 0.5) / 0.5

    # Convert HWC to CHW format
    arr = arr.transpose(2, 0, 1)

    # Add batch dimension [1, 3, 448, 448]
    arr = np.expand_dims(arr, axis=0)

    return arr, img_resized

def test_model(image_path: str = None):
    """Test ONNX model inference."""
    print("=" * 60)
    print("ONNX Model Test for LaborView MedSigLIP")
    print("=" * 60)

    # Check model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return False

    data_path = MODEL_PATH + ".data"
    if not os.path.exists(data_path):
        print(f"ERROR: Model weights not found at {data_path}")
        return False

    model_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
    data_size = os.path.getsize(data_path) / 1024 / 1024 / 1024
    print(f"\nModel file: {model_size:.2f} MB")
    print(f"Weights file: {data_size:.2f} GB")

    # Load ONNX model
    print("\nLoading ONNX Runtime session...")
    try:
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        print("Session created successfully!")
    except Exception as e:
        print(f"ERROR: Failed to create session: {e}")
        return False

    # Print model info
    print("\n--- Model Inputs ---")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")

    print("\n--- Model Outputs ---")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    # Prepare input
    print("\n--- Running Inference ---")
    input_tensor, original_img = preprocess_image(image_path)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input dtype: {input_tensor.dtype}")
    print(f"Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

    # Run inference
    try:
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        print("\nInference successful!")
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        return False

    # Parse outputs
    print("\n--- Output Results ---")
    output_names = [out.name for out in session.get_outputs()]

    for i, (name, output) in enumerate(zip(output_names, outputs)):
        print(f"\n{name}:")
        print(f"  Shape: {output.shape}")
        print(f"  Dtype: {output.dtype}")
        print(f"  Range: [{output.min():.4f}, {output.max():.4f}]")

        if "seg" in name.lower():
            # Segmentation output - show class distribution
            if len(output.shape) == 4:  # [1, C, H, W]
                probs = output[0]  # [C, H, W]
                class_preds = np.argmax(probs, axis=0)  # [H, W]
                unique, counts = np.unique(class_preds, return_counts=True)
                print(f"  Class distribution:")
                for cls, cnt in zip(unique, counts):
                    pct = cnt / class_preds.size * 100
                    print(f"    Class {cls}: {cnt} pixels ({pct:.1f}%)")

        elif "plane" in name.lower():
            # Plane prediction
            pred = int(output[0])
            plane_name = "Transperineal" if pred == 0 else "Other"
            print(f"  Predicted plane: {plane_name} (class {pred})")

    # Save visualization
    print("\n--- Saving Visualization ---")
    save_segmentation_overlay(original_img, class_preds if 'class_preds' in dir() else None, output)

    print("\n" + "=" * 60)
    print("TEST PASSED - Model is working correctly!")
    print("=" * 60)

    return True


def save_segmentation_overlay(original_img, class_preds, outputs):
    """Save segmentation overlay visualization."""
    import matplotlib.pyplot as plt

    # Get segmentation from outputs
    seg_output = None
    for name, output in zip(['seg_probs', 'plane_pred'], outputs):
        if 'seg' in name.lower():
            seg_output = output
            break

    if seg_output is None:
        seg_output = outputs[0]  # First output is segmentation

    # Process segmentation
    if len(seg_output.shape) == 4:
        probs = seg_output[0]  # [C, H, W]
        class_preds = np.argmax(probs, axis=0)  # [H, W]

    # Create color overlay
    colors = np.array([
        [0, 0, 0],       # Class 0: Background (black/transparent)
        [255, 0, 0],     # Class 1: Symphysis (red)
        [0, 255, 0],     # Class 2: Fetal head (green)
    ])

    # Create RGB overlay
    overlay = colors[class_preds]

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Segmentation mask
    axes[1].imshow(class_preds, cmap='viridis')
    axes[1].set_title('Segmentation Mask\n(0=BG, 1=Symphysis, 2=Head)')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(original_img)
    mask_overlay = np.zeros((*class_preds.shape, 4))
    mask_overlay[class_preds == 1] = [1, 0, 0, 0.5]  # Red for symphysis
    mask_overlay[class_preds == 2] = [0, 1, 0, 0.5]  # Green for head
    axes[2].imshow(mask_overlay)
    axes[2].set_title('Overlay (Red=Symphysis, Green=Head)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('segmentation_result.png', dpi=150)
    print(f"Saved visualization to: segmentation_result.png")
    plt.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Use test image if available
    image_path = TEST_IMAGE if os.path.exists(TEST_IMAGE) else None
    success = test_model(image_path)
    exit(0 if success else 1)
