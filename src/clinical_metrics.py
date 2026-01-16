"""
Clinical Metrics Module for LaborView AI
Computes AoP, HSD, and HC from segmentation masks

Reference: Angle of Progression (AoP) and Head-Symphysis Distance (HSD)
are key intrapartum ultrasound measurements for assessing labor progress.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math


@dataclass
class ClinicalMetrics:
    """Clinical measurements derived from segmentation masks"""
    # Angle of Progression (degrees)
    aop: Optional[float] = None
    aop_interpretation: Optional[str] = None

    # Head-Symphysis Distance (pixels, can convert to mm with calibration)
    hsd: Optional[float] = None
    hsd_interpretation: Optional[str] = None

    # Head Circumference/Perimeter (pixels)
    head_circumference: Optional[float] = None
    head_area: Optional[float] = None

    # Pubic Symphysis measurements
    symphysis_length: Optional[float] = None

    # Quality indicators
    segmentation_quality: Optional[str] = None
    confidence: Optional[float] = None

    # Overall assessment
    labor_progress: Optional[str] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'aop_degrees': self.aop,
            'aop_interpretation': self.aop_interpretation,
            'hsd_pixels': self.hsd,
            'hsd_interpretation': self.hsd_interpretation,
            'head_circumference_pixels': self.head_circumference,
            'head_area_pixels': self.head_area,
            'symphysis_length_pixels': self.symphysis_length,
            'segmentation_quality': self.segmentation_quality,
            'confidence': self.confidence,
            'labor_progress': self.labor_progress,
            'recommendation': self.recommendation
        }


def extract_contours(mask: np.ndarray, class_id: int) -> Optional[np.ndarray]:
    """Extract the largest contour for a given class from segmentation mask"""
    binary = (mask == class_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Return largest contour
    return max(contours, key=cv2.contourArea)


def get_centroid(contour: np.ndarray) -> Tuple[float, float]:
    """Get centroid of a contour"""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return (0, 0)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)


def get_lowest_point(contour: np.ndarray) -> Tuple[int, int]:
    """Get the lowest (highest y-value) point of a contour"""
    # In image coordinates, y increases downward
    lowest_idx = contour[:, :, 1].argmax()
    return tuple(contour[lowest_idx, 0])


def get_highest_point(contour: np.ndarray) -> Tuple[int, int]:
    """Get the highest (lowest y-value) point of a contour"""
    highest_idx = contour[:, :, 1].argmin()
    return tuple(contour[highest_idx, 0])


def fit_line_to_symphysis(contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a line to the pubic symphysis contour
    Returns: (point on line, direction vector)
    """
    # Fit line using PCA or linear regression
    points = contour.reshape(-1, 2).astype(np.float32)

    # Use cv2.fitLine for robust line fitting
    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten()

    return np.array([x0, y0]), np.array([vx, vy])


def compute_aop(
    symphysis_contour: np.ndarray,
    head_contour: np.ndarray,
    image_height: int
) -> Tuple[Optional[float], dict]:
    """
    Compute Angle of Progression (AoP)

    AoP is the angle between:
    1. A line along the long axis of the pubic symphysis
    2. A line from the inferior edge of symphysis to the leading edge of fetal head

    Returns:
        aop: Angle in degrees (None if cannot compute)
        debug_info: Dictionary with intermediate values for visualization
    """
    debug_info = {}

    try:
        # Get symphysis line
        symphysis_point, symphysis_dir = fit_line_to_symphysis(symphysis_contour)
        debug_info['symphysis_point'] = symphysis_point
        debug_info['symphysis_dir'] = symphysis_dir

        # Get inferior (lowest) point of symphysis
        symphysis_inferior = get_lowest_point(symphysis_contour)
        debug_info['symphysis_inferior'] = symphysis_inferior

        # Get leading (lowest) point of fetal head
        head_leading = get_lowest_point(head_contour)
        debug_info['head_leading'] = head_leading

        # Vector from symphysis inferior to head leading point
        head_vector = np.array([
            head_leading[0] - symphysis_inferior[0],
            head_leading[1] - symphysis_inferior[1]
        ])
        debug_info['head_vector'] = head_vector

        # Normalize vectors
        symphysis_dir_norm = symphysis_dir / (np.linalg.norm(symphysis_dir) + 1e-6)
        head_vector_norm = head_vector / (np.linalg.norm(head_vector) + 1e-6)

        # Compute angle using dot product
        dot_product = np.clip(np.dot(symphysis_dir_norm, head_vector_norm), -1.0, 1.0)
        angle_rad = np.arccos(abs(dot_product))
        angle_deg = np.degrees(angle_rad)

        # AoP is typically measured as the angle from the symphysis axis
        # Ensure we're measuring the correct angle (0-180 range)
        aop = angle_deg

        # Adjust based on relative positions
        # If head is anterior to symphysis, AoP should be larger
        if head_leading[1] > symphysis_inferior[1]:
            # Head is below symphysis (more descended)
            aop = 90 + (90 - angle_deg)

        debug_info['aop'] = aop
        return aop, debug_info

    except Exception as e:
        debug_info['error'] = str(e)
        return None, debug_info


def compute_hsd(
    symphysis_contour: np.ndarray,
    head_contour: np.ndarray
) -> Tuple[Optional[float], dict]:
    """
    Compute Head-Symphysis Distance (HSD)

    HSD is the shortest distance from the pubic symphysis to the fetal head

    Returns:
        hsd: Distance in pixels (None if cannot compute)
        debug_info: Dictionary with intermediate values
    """
    debug_info = {}

    try:
        # Get inferior point of symphysis
        symphysis_inferior = get_lowest_point(symphysis_contour)
        debug_info['symphysis_inferior'] = symphysis_inferior

        # Find closest point on head contour to symphysis inferior
        head_points = head_contour.reshape(-1, 2)
        distances = np.sqrt(
            (head_points[:, 0] - symphysis_inferior[0])**2 +
            (head_points[:, 1] - symphysis_inferior[1])**2
        )
        min_idx = distances.argmin()
        closest_head_point = tuple(head_points[min_idx])

        debug_info['closest_head_point'] = closest_head_point

        hsd = distances[min_idx]
        debug_info['hsd'] = hsd

        return hsd, debug_info

    except Exception as e:
        debug_info['error'] = str(e)
        return None, debug_info


def compute_head_circumference(head_contour: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute head circumference (perimeter) and area from contour

    Note: This is the visible head perimeter from the transperineal view,
    NOT the standard prenatal HC measurement.

    Returns:
        circumference: Perimeter in pixels
        area: Area in pixels^2
    """
    try:
        circumference = cv2.arcLength(head_contour, closed=True)
        area = cv2.contourArea(head_contour)
        return circumference, area
    except:
        return None, None


def compute_symphysis_length(symphysis_contour: np.ndarray) -> Optional[float]:
    """Compute the length of the pubic symphysis"""
    try:
        # Fit minimum area rectangle
        rect = cv2.minAreaRect(symphysis_contour)
        width, height = rect[1]
        # Length is the longer dimension
        return max(width, height)
    except:
        return None


def assess_segmentation_quality(
    symphysis_contour: Optional[np.ndarray],
    head_contour: Optional[np.ndarray],
    image_size: Tuple[int, int]
) -> Tuple[str, float]:
    """
    Assess quality of segmentation for clinical use

    Returns:
        quality: "good", "acceptable", "poor"
        confidence: 0.0 to 1.0
    """
    if symphysis_contour is None or head_contour is None:
        return "poor", 0.0

    image_area = image_size[0] * image_size[1]

    symphysis_area = cv2.contourArea(symphysis_contour)
    head_area = cv2.contourArea(head_contour)

    # Check if areas are reasonable (not too small or too large)
    symphysis_ratio = symphysis_area / image_area
    head_ratio = head_area / image_area

    confidence = 1.0
    issues = []

    # Symphysis should be small relative to image (typically 1-5%)
    if symphysis_ratio < 0.005:
        issues.append("symphysis_too_small")
        confidence -= 0.3
    elif symphysis_ratio > 0.15:
        issues.append("symphysis_too_large")
        confidence -= 0.2

    # Head should be moderate size (typically 5-30%)
    if head_ratio < 0.02:
        issues.append("head_too_small")
        confidence -= 0.3
    elif head_ratio > 0.5:
        issues.append("head_too_large")
        confidence -= 0.2

    # Check contour complexity (should be relatively smooth)
    symphysis_perimeter = cv2.arcLength(symphysis_contour, True)
    head_perimeter = cv2.arcLength(head_contour, True)

    # Circularity check for head (should be roughly circular)
    head_circularity = 4 * np.pi * head_area / (head_perimeter ** 2 + 1e-6)
    if head_circularity < 0.3:
        issues.append("head_not_circular")
        confidence -= 0.15

    confidence = max(0.0, min(1.0, confidence))

    if confidence >= 0.7:
        quality = "good"
    elif confidence >= 0.4:
        quality = "acceptable"
    else:
        quality = "poor"

    return quality, confidence


def interpret_aop(aop: Optional[float]) -> str:
    """
    Interpret AoP value for clinical guidance

    Clinical thresholds (approximate):
    - AoP < 110°: Early labor / high station
    - AoP 110-120°: Active labor
    - AoP 120-140°: Advanced labor, good progress
    - AoP > 140°: Late labor, imminent delivery
    """
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
    """
    Interpret HSD value for clinical guidance

    Note: HSD in pixels needs calibration for mm conversion.
    Using relative thresholds based on image size.
    """
    if hsd is None:
        return "Unable to measure"

    # Normalize to image height
    hsd_ratio = hsd / image_height

    if hsd_ratio > 0.3:
        return "Large distance - head not engaged"
    elif hsd_ratio > 0.2:
        return "Moderate distance - early descent"
    elif hsd_ratio > 0.1:
        return "Small distance - good descent"
    else:
        return "Minimal distance - head at symphysis level"


def assess_labor_progress(
    aop: Optional[float],
    hsd: Optional[float],
    image_height: int
) -> Tuple[str, str]:
    """
    Overall assessment of labor progress

    Returns:
        progress: "normal", "monitor", "concern"
        recommendation: Clinical guidance string
    """
    if aop is None and hsd is None:
        return "unknown", "Unable to assess - segmentation quality insufficient"

    # Weight AoP more heavily (more validated clinically)
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
        progress = "normal"
        recommendation = "Labor progressing well. Continue routine monitoring."
    elif score >= 0:
        progress = "monitor"
        recommendation = "Labor progressing. Continue close monitoring and reassess in 30-60 minutes."
    else:
        progress = "concern"
        recommendation = "Slow progress noted. Consider clinical examination and evaluate need for intervention."

    return progress, recommendation


def compute_all_metrics(
    segmentation_mask: np.ndarray,
    symphysis_class: int = 1,
    head_class: int = 2,
    pixel_spacing_mm: Optional[float] = None
) -> ClinicalMetrics:
    """
    Compute all clinical metrics from a segmentation mask

    Args:
        segmentation_mask: HxW numpy array with class labels
        symphysis_class: Class ID for pubic symphysis (default: 1)
        head_class: Class ID for fetal head (default: 2)
        pixel_spacing_mm: Optional pixel spacing for mm conversion

    Returns:
        ClinicalMetrics object with all measurements
    """
    metrics = ClinicalMetrics()

    image_height, image_width = segmentation_mask.shape[:2]

    # Extract contours
    symphysis_contour = extract_contours(segmentation_mask, symphysis_class)
    head_contour = extract_contours(segmentation_mask, head_class)

    # Assess quality first
    quality, confidence = assess_segmentation_quality(
        symphysis_contour, head_contour, (image_height, image_width)
    )
    metrics.segmentation_quality = quality
    metrics.confidence = confidence

    if symphysis_contour is None or head_contour is None:
        metrics.labor_progress = "unknown"
        metrics.recommendation = "Segmentation incomplete. Ensure proper ultrasound view and retry."
        return metrics

    # Compute AoP
    aop, aop_debug = compute_aop(symphysis_contour, head_contour, image_height)
    metrics.aop = aop
    metrics.aop_interpretation = interpret_aop(aop)

    # Compute HSD
    hsd, hsd_debug = compute_hsd(symphysis_contour, head_contour)
    metrics.hsd = hsd
    metrics.hsd_interpretation = interpret_hsd(hsd, image_height)

    # Compute head measurements
    hc, area = compute_head_circumference(head_contour)
    metrics.head_circumference = hc
    metrics.head_area = area

    # Compute symphysis length
    metrics.symphysis_length = compute_symphysis_length(symphysis_contour)

    # Convert to mm if pixel spacing provided
    if pixel_spacing_mm is not None and metrics.hsd is not None:
        metrics.hsd = metrics.hsd * pixel_spacing_mm
    if pixel_spacing_mm is not None and metrics.head_circumference is not None:
        metrics.head_circumference = metrics.head_circumference * pixel_spacing_mm

    # Overall assessment
    progress, recommendation = assess_labor_progress(aop, hsd, image_height)
    metrics.labor_progress = progress
    metrics.recommendation = recommendation

    return metrics


def visualize_metrics(
    image: np.ndarray,
    segmentation_mask: np.ndarray,
    metrics: ClinicalMetrics,
    symphysis_class: int = 1,
    head_class: int = 2
) -> np.ndarray:
    """
    Create visualization of segmentation and metrics

    Returns:
        Annotated image with overlays
    """
    # Create color overlay
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    # Color masks
    overlay = vis.copy()

    # Symphysis in blue
    symphysis_mask = (segmentation_mask == symphysis_class)
    overlay[symphysis_mask] = [255, 100, 100]  # Blue

    # Head in green
    head_mask = (segmentation_mask == head_class)
    overlay[head_mask] = [100, 255, 100]  # Green

    # Blend
    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    # Draw contours
    symphysis_contour = extract_contours(segmentation_mask, symphysis_class)
    head_contour = extract_contours(segmentation_mask, head_class)

    if symphysis_contour is not None:
        cv2.drawContours(vis, [symphysis_contour], -1, (255, 0, 0), 2)
    if head_contour is not None:
        cv2.drawContours(vis, [head_contour], -1, (0, 255, 0), 2)

    # Add text annotations
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6

    if metrics.aop is not None:
        cv2.putText(vis, f"AoP: {metrics.aop:.1f} deg", (10, y_offset),
                    font, font_scale, (255, 255, 255), 2)
        y_offset += 25

    if metrics.hsd is not None:
        cv2.putText(vis, f"HSD: {metrics.hsd:.1f} px", (10, y_offset),
                    font, font_scale, (255, 255, 255), 2)
        y_offset += 25

    if metrics.head_circumference is not None:
        cv2.putText(vis, f"HC: {metrics.head_circumference:.1f} px", (10, y_offset),
                    font, font_scale, (255, 255, 255), 2)
        y_offset += 25

    # Progress indicator
    if metrics.labor_progress:
        color = {
            "normal": (0, 255, 0),
            "monitor": (0, 255, 255),
            "concern": (0, 0, 255),
            "unknown": (128, 128, 128)
        }.get(metrics.labor_progress, (255, 255, 255))

        cv2.putText(vis, f"Status: {metrics.labor_progress.upper()}", (10, y_offset),
                    font, font_scale, color, 2)

    return vis


# Convenience function for inference pipeline
def analyze_segmentation(
    segmentation_logits: np.ndarray,
    original_image: Optional[np.ndarray] = None,
    pixel_spacing_mm: Optional[float] = None
) -> Tuple[ClinicalMetrics, Optional[np.ndarray]]:
    """
    Full analysis pipeline from model output to clinical metrics

    Args:
        segmentation_logits: Model output (C, H, W) or (H, W) class labels
        original_image: Optional original image for visualization
        pixel_spacing_mm: Optional pixel spacing for mm conversion

    Returns:
        metrics: ClinicalMetrics object
        visualization: Annotated image (if original_image provided)
    """
    # Convert logits to class labels if needed
    if len(segmentation_logits.shape) == 3:
        segmentation_mask = np.argmax(segmentation_logits, axis=0)
    else:
        segmentation_mask = segmentation_logits.astype(np.int32)

    # Compute metrics
    metrics = compute_all_metrics(
        segmentation_mask,
        symphysis_class=1,
        head_class=2,
        pixel_spacing_mm=pixel_spacing_mm
    )

    # Create visualization if image provided
    visualization = None
    if original_image is not None:
        visualization = visualize_metrics(
            original_image, segmentation_mask, metrics
        )

    return metrics, visualization


if __name__ == "__main__":
    # Test with synthetic data
    print("Clinical Metrics Module - Test")
    print("=" * 50)

    # Create synthetic segmentation mask
    mask = np.zeros((384, 384), dtype=np.uint8)

    # Draw synthetic symphysis (small rectangle at top)
    cv2.rectangle(mask, (150, 50), (250, 100), 1, -1)

    # Draw synthetic fetal head (ellipse below)
    cv2.ellipse(mask, (200, 250), (80, 100), 0, 0, 360, 2, -1)

    # Compute metrics
    metrics = compute_all_metrics(mask, symphysis_class=1, head_class=2)

    print(f"AoP: {metrics.aop:.1f}°" if metrics.aop else "AoP: N/A")
    print(f"  Interpretation: {metrics.aop_interpretation}")
    print(f"HSD: {metrics.hsd:.1f} px" if metrics.hsd else "HSD: N/A")
    print(f"  Interpretation: {metrics.hsd_interpretation}")
    print(f"Head Circumference: {metrics.head_circumference:.1f} px" if metrics.head_circumference else "HC: N/A")
    print(f"Quality: {metrics.segmentation_quality} (confidence: {metrics.confidence:.2f})")
    print(f"Labor Progress: {metrics.labor_progress}")
    print(f"Recommendation: {metrics.recommendation}")
