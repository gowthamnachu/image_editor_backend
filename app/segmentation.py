"""
Image segmentation using Mediapipe and OpenCV GrabCut
"""
import numpy as np
import cv2
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import Mediapipe
try:
    import mediapipe as mp
    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    MEDIAPIPE_AVAILABLE = True
    logger.info("Mediapipe loaded successfully")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    mp_selfie = None
    logger.warning(f"Mediapipe not available: {e}. Will use GrabCut fallback.")


def segment_person_mediapipe(img_bgr: np.ndarray, threshold: float = 0.5) -> Optional[np.ndarray]:
    """
    Segment person using Mediapipe Selfie Segmentation.
    Returns binary mask (0-255) or None if failed.
    """
    if not MEDIAPIPE_AVAILABLE or mp_selfie is None:
        return None
    
    try:
        # Convert to RGB for Mediapipe
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Process
        results = mp_selfie.process(rgb)
        
        if results.segmentation_mask is None:
            return None
        
        # Get the raw mask (float32, 0-1 range)
        mask_raw = results.segmentation_mask
        
        # Resize mask to match image size if needed
        if mask_raw.shape[:2] != img_bgr.shape[:2]:
            mask_raw = cv2.resize(mask_raw, (img_bgr.shape[1], img_bgr.shape[0]), 
                                 interpolation=cv2.INTER_LINEAR)
        
        # Convert to 8-bit (0-255)
        mask_8bit = (mask_raw * 255).astype(np.uint8)
        
        # Apply adaptive thresholding for better edge detection
        # Use a lower threshold to capture more details
        threshold_value = int(threshold * 255)
        _, mask_binary = cv2.threshold(mask_8bit, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Improve mask quality with morphological operations
        # Use smaller kernel for better edge preservation
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        
        # Fill small holes
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
        
        # Remove small noise
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        # Dilate slightly to ensure we capture edges
        mask_binary = cv2.dilate(mask_binary, kernel_small, iterations=1)
        
        # Apply edge-preserving blur for smooth transitions
        # Use bilateral filter to preserve edges while smoothing
        mask_smooth = cv2.bilateralFilter(mask_binary, 9, 75, 75)
        
        # Final Gaussian blur for feathering (smaller kernel for sharper edges)
        mask_feathered = cv2.GaussianBlur(mask_smooth, (5, 5), 0)
        
        return mask_feathered
        
    except Exception as e:
        logger.error(f"Mediapipe segmentation failed: {e}")
        return None


def segment_person_grabcut(img_bgr: np.ndarray, rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Segment person using OpenCV GrabCut algorithm.
    If rect is None, automatically detect person region.
    Returns binary mask (0-255).
    """
    h, w = img_bgr.shape[:2]
    
    # Create mask
    mask = np.zeros((h, w), np.uint8)
    
    # Background and foreground models for GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # If no rect provided, auto-detect region
    if rect is None:
        rect = auto_detect_person_region(img_bgr)
    
    # Ensure rect is within image bounds
    x, y, w_rect, h_rect = rect
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    w_rect = min(w_rect, w - x)
    h_rect = min(h_rect, h - y)
    rect = (x, y, w_rect, h_rect)
    
    try:
        # Apply GrabCut with more iterations for better accuracy
        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)
        
        # Create binary mask (foreground = 255, background = 0)
        # Include probable foreground for better coverage
        mask_binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        
        # Clean up mask with morphological operations
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        
        # Fill holes
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        
        # Remove noise
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_medium, iterations=2)
        
        # Smooth the mask
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        # Apply edge-preserving smoothing
        mask_smooth = cv2.bilateralFilter(mask_binary, 9, 75, 75)
        
        # Feather edges for natural blending
        mask_feathered = cv2.GaussianBlur(mask_smooth, (7, 7), 0)
        
        return mask_feathered
        
    except Exception as e:
        logger.error(f"GrabCut failed: {e}")
        # Return center region as fallback
        mask = np.zeros((h, w), np.uint8)
        center_rect = (w // 4, h // 6, w // 2, h * 2 // 3)
        x, y, w_r, h_r = center_rect
        mask[y:y+h_r, x:x+w_r] = 255
        return cv2.GaussianBlur(mask, (21, 21), 0)


def auto_detect_person_region(img_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Automatically detect person region using edge detection and contours.
    Returns (x, y, width, height) rectangle.
    """
    h, w = img_bgr.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while keeping edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Use Canny edge detection with automatic threshold
    # Calculate median for dynamic thresholding
    median = np.median(filtered)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(filtered, lower, upper)
    
    # Dilate edges to connect nearby contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    
    # Close gaps
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: use center region with reasonable margins
        margin_x, margin_y = w // 8, h // 10
        return (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    
    # Find the largest contour (likely to be person)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w_rect, h_rect = cv2.boundingRect(largest_contour)
    
    # Calculate contour area ratio
    contour_area = cv2.contourArea(largest_contour)
    rect_area = w_rect * h_rect
    fill_ratio = contour_area / rect_area if rect_area > 0 else 0
    
    # If fill ratio is too low, might be detecting wrong object
    # Use a larger, more centered region
    if fill_ratio < 0.3:
        margin_x, margin_y = w // 6, h // 8
        return (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    
    # Add smart padding based on image size and contour
    # More padding for better GrabCut initialization
    padding_x = int(w_rect * 0.15)
    padding_y = int(h_rect * 0.15)
    
    # Apply padding
    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    w_rect = min(w - x, w_rect + 2 * padding_x)
    h_rect = min(h - y, h_rect + 2 * padding_y)
    
    # Ensure minimum size (at least 20% of image)
    min_width = int(w * 0.2)
    min_height = int(h * 0.2)
    
    if w_rect < min_width or h_rect < min_height:
        # Use center region
        margin_x, margin_y = w // 6, h // 8
        return (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    
    return (x, y, w_rect, h_rect)


def refine_mask_with_strokes(
    img_bgr: np.ndarray,
    initial_mask: np.ndarray,
    fg_strokes: Optional[np.ndarray] = None,
    bg_strokes: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Refine segmentation mask using user-drawn foreground/background strokes.
    
    Args:
        img_bgr: Original image
        initial_mask: Initial segmentation mask (0-255)
        fg_strokes: Foreground strokes mask (255 where user marked foreground)
        bg_strokes: Background strokes mask (255 where user marked background)
    
    Returns:
        Refined mask (0-255)
    """
    h, w = img_bgr.shape[:2]
    
    # Initialize GrabCut mask with 4 possible values
    mask = np.zeros((h, w), np.uint8)
    
    # Convert initial mask to probabilities
    # Higher values in initial_mask = more likely foreground
    mask[initial_mask > 200] = cv2.GC_FGD      # Definite foreground
    mask[(initial_mask > 100) & (initial_mask <= 200)] = cv2.GC_PR_FGD  # Probable foreground
    mask[(initial_mask > 50) & (initial_mask <= 100)] = cv2.GC_PR_BGD   # Probable background
    mask[initial_mask <= 50] = cv2.GC_BGD      # Definite background
    
    # Apply user strokes (these override the initial mask)
    if fg_strokes is not None:
        # Dilate foreground strokes slightly for better coverage
        kernel = np.ones((5, 5), np.uint8)
        fg_dilated = cv2.dilate(fg_strokes, kernel, iterations=1)
        mask[fg_dilated > 127] = cv2.GC_FGD  # Definite foreground
    
    if bg_strokes is not None:
        # Dilate background strokes slightly for better coverage
        kernel = np.ones((5, 5), np.uint8)
        bg_dilated = cv2.dilate(bg_strokes, kernel, iterations=1)
        mask[bg_dilated > 127] = cv2.GC_BGD  # Definite background
    
    # Background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    try:
        # Refine with GrabCut (more iterations for better refinement)
        cv2.grabCut(img_bgr, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        
        # Create binary mask
        mask_binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        
        # Clean and smooth
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        
        # Close small gaps
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
        
        # Remove small noise
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Smooth with bilateral filter to preserve edges
        mask_smooth = cv2.bilateralFilter(mask_binary, 9, 75, 75)
        
        # Final feathering
        mask_feathered = cv2.GaussianBlur(mask_smooth, (5, 5), 0)
        
        return mask_feathered
        
    except Exception as e:
        logger.error(f"Mask refinement failed: {e}")
        return initial_mask


def segment_person(
    img_bgr: np.ndarray,
    method: str = "auto",
    threshold: float = 0.5,
    rect: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[np.ndarray, str]:
    """
    Segment person from image using best available method.
    
    Args:
        img_bgr: Input image in BGR format
        method: "auto", "mediapipe", or "grabcut"
        threshold: Threshold for Mediapipe (0.0-1.0)
        rect: Optional rectangle for GrabCut
    
    Returns:
        Tuple of (mask, method_used)
        mask: Binary mask (0-255) where 255 = person
        method_used: String indicating which method was used
    """
    
    if method == "auto":
        # Try Mediapipe first
        if MEDIAPIPE_AVAILABLE:
            mask = segment_person_mediapipe(img_bgr, threshold)
            if mask is not None:
                return mask, "mediapipe"
        
        # Fallback to GrabCut
        mask = segment_person_grabcut(img_bgr, rect)
        return mask, "grabcut"
    
    elif method == "mediapipe":
        if not MEDIAPIPE_AVAILABLE:
            raise ValueError("Mediapipe is not available")
        mask = segment_person_mediapipe(img_bgr, threshold)
        if mask is None:
            raise ValueError("Mediapipe segmentation failed")
        return mask, "mediapipe"
    
    elif method == "grabcut":
        mask = segment_person_grabcut(img_bgr, rect)
        return mask, "grabcut"
    
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
