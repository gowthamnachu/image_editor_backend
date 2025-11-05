"""
Compositing functions for placing text behind person and background operations
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Tuple, Optional, Union
import logging

from .utils import hex_to_bgr, get_default_font, wrap_text

logger = logging.getLogger(__name__)


def place_text_behind_person(
    image_bgr: np.ndarray,
    mask_gray: np.ndarray,
    text: str,
    position: Tuple[int, int] = (100, 200),
    font_size: int = 72,
    font_family: str = "Arial",
    font_weight: str = "normal",
    font_style: str = "normal",
    font_path: Optional[str] = None,
    color: Union[str, Tuple[int, int, int, int]] = "#FFFFFF",
    bg_color: Optional[Union[str, Tuple[int, int, int]]] = None,
    bg_image: Optional[np.ndarray] = None,
    max_width: Optional[int] = None,
    align: str = "left",
    line_spacing: int = 10,
    letter_spacing: int = 0,
    stroke: bool = False,
    stroke_color: str = "#000000",
    stroke_width: int = 2,
    shadow: bool = False,
    shadow_color: str = "#000000",
    shadow_blur: int = 10,
    shadow_offset_x: int = 5,
    shadow_offset_y: int = 5,
    text_transform: str = "none"
) -> np.ndarray:
    """
    Place text behind person using mask compositing with advanced styling.
    
    Args:
        image_bgr: Original image with person (BGR)
        mask_gray: Person segmentation mask (0-255, single channel)
        text: Text to render
        position: (x, y) position for text
        font_size: Font size in pixels
        font_family: Font family name
        font_weight: Font weight ("normal", "bold")
        font_style: Font style ("normal", "italic")
        font_path: Optional path to TTF font file
        color: Text color as hex string or RGBA tuple
        bg_color: Background color (if replacing background)
        bg_image: Background image (if replacing with image)
        max_width: Maximum width for text wrapping
        align: Text alignment ("left", "center", "right")
        line_spacing: Spacing between lines
        letter_spacing: Spacing between letters
        stroke: Enable text outline
        stroke_color: Outline color
        stroke_width: Outline width
        shadow: Enable drop shadow
        shadow_color: Shadow color
        shadow_blur: Shadow blur amount
        shadow_offset_x: Shadow X offset
        shadow_offset_y: Shadow Y offset
        text_transform: Text transformation ("none", "uppercase", "lowercase", "capitalize")
    
    Returns:
        Composited image with text behind person (BGR)
    """
    h, w = image_bgr.shape[:2]
    
    # Ensure mask is single channel
    if len(mask_gray.shape) == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)
    
    # Apply text transformations
    if text_transform == "uppercase":
        text = text.upper()
    elif text_transform == "lowercase":
        text = text.lower()
    elif text_transform == "capitalize":
        text = text.title()
    
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Create background layer
    if bg_image is not None:
        # Use provided background image
        bg_rgb = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
        if bg_rgb.shape[:2] != (h, w):
            bg_rgb = cv2.resize(bg_rgb, (w, h))
        bg_pil = Image.fromarray(bg_rgb)
    elif bg_color is not None:
        # Create solid color background
        if isinstance(bg_color, str):
            bg_bgr = hex_to_bgr(bg_color)
            bg_color_rgb = (bg_bgr[2], bg_bgr[1], bg_bgr[0])
        else:
            bg_color_rgb = bg_color
        bg_pil = Image.new("RGB", (w, h), bg_color_rgb)
    else:
        # Use original background from image
        bg_pil = Image.fromarray(img_rgb)
    
    # Draw text on background
    draw = ImageDraw.Draw(bg_pil)
    
    # Get font - try to use system fonts based on family and style
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Try to load system font with variations
            font_name = font_family
            if font_weight == "bold" and font_style == "italic":
                font_name_var = f"{font_family} Bold Italic"
            elif font_weight == "bold":
                font_name_var = f"{font_family} Bold"
            elif font_style == "italic":
                font_name_var = f"{font_family} Italic"
            else:
                font_name_var = font_family
            
            try:
                font = ImageFont.truetype(font_name_var, font_size)
            except:
                try:
                    font = ImageFont.truetype(font_family, font_size)
                except:
                    font = get_default_font(font_size)
    except Exception as e:
        logger.warning(f"Failed to load font: {e}. Using default.")
        font = get_default_font(font_size)
    
    # Parse colors
    if isinstance(color, str):
        color_bgr = hex_to_bgr(color)
        text_color = (color_bgr[2], color_bgr[1], color_bgr[0])
    else:
        text_color = color[:3] if len(color) > 3 else color
    
    if stroke:
        stroke_bgr = hex_to_bgr(stroke_color)
        stroke_rgb = (stroke_bgr[2], stroke_bgr[1], stroke_bgr[0])
    
    if shadow:
        shadow_bgr = hex_to_bgr(shadow_color)
        shadow_rgb = (shadow_bgr[2], shadow_bgr[1], shadow_bgr[0])
    
    # Wrap text if max_width specified
    if max_width:
        lines = wrap_text(text, font, max_width)
    else:
        lines = text.split('\n')
    
    # Calculate total text height
    line_heights = []
    for line in lines:
        bbox = font.getbbox(line)
        line_heights.append(bbox[3] - bbox[1])
    
    total_height = sum(line_heights) + line_spacing * (len(lines) - 1)
    
    # Draw each line with effects
    x, y = position
    current_y = y
    
    for i, line in enumerate(lines):
        bbox = font.getbbox(line)
        line_width = bbox[2] - bbox[0]
        line_height = line_heights[i]
        
        # Apply alignment
        if align == "center":
            line_x = x - line_width // 2
        elif align == "right":
            line_x = x - line_width
        else:  # left
            line_x = x
        
        # Draw shadow if enabled
        if shadow:
            # Create a temporary image for blur effect
            if shadow_blur > 0:
                shadow_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                shadow_draw = ImageDraw.Draw(shadow_img)
                shadow_draw.text((line_x + shadow_offset_x, current_y + shadow_offset_y), 
                               line, font=font, fill=shadow_rgb + (200,))
                shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(shadow_blur))
                bg_pil.paste(shadow_img, (0, 0), shadow_img)
            else:
                draw.text((line_x + shadow_offset_x, current_y + shadow_offset_y), 
                         line, font=font, fill=shadow_rgb)
        
        # Draw stroke/outline if enabled
        if stroke:
            # Draw outline by drawing text multiple times with offset
            for adj_x in range(-stroke_width, stroke_width + 1):
                for adj_y in range(-stroke_width, stroke_width + 1):
                    if adj_x != 0 or adj_y != 0:
                        draw.text((line_x + adj_x, current_y + adj_y), 
                                line, font=font, fill=stroke_rgb)
        
        # Draw main text
        draw.text((line_x, current_y), line, font=font, fill=text_color)
        
        current_y += line_height + line_spacing
    
    # Convert back to numpy
    bg_with_text = np.array(bg_pil)
    
    # Debug: Log dimensions
    logger.info(f"Image shape: {img_rgb.shape}, BG with text shape: {bg_with_text.shape}, Mask shape: {mask_gray.shape}")
    logger.info(f"Mask values - min: {mask_gray.min()}, max: {mask_gray.max()}, mean: {mask_gray.mean():.2f}")
    
    # Feather mask for smooth blending
    # Use bilateral filter first to preserve edges
    mask_bilateral = cv2.bilateralFilter(mask_gray, 9, 75, 75)
    
    # Then apply Gaussian for final smoothing
    mask_feathered = cv2.GaussianBlur(mask_bilateral, (7, 7), 0)
    
    # Normalize mask to 0-1
    mask_norm = mask_feathered.astype(np.float32) / 255.0
    
    # Apply slight edge enhancement to prevent halos
    # Sharpen the mask edges slightly
    mask_sharp = cv2.addWeighted(mask_feathered, 1.5, 
                                  cv2.GaussianBlur(mask_feathered, (15, 15), 0), -0.5, 0)
    mask_sharp = np.clip(mask_sharp, 0, 255).astype(np.uint8)
    mask_norm = mask_sharp.astype(np.float32) / 255.0
    
    logger.info(f"Mask normalized - min: {mask_norm.min():.3f}, max: {mask_norm.max():.3f}, mean: {mask_norm.mean():.3f}")
    
    # Expand mask to 3 channels
    mask_3ch = np.stack([mask_norm] * 3, axis=2)
    
    # Composite: foreground where mask=1, background-with-text where mask=0
    # person * mask + background_with_text * (1 - mask)
    # This means: where mask is WHITE (255/1.0) = person pixels, where mask is BLACK (0/0.0) = background pixels with text
    composite = (img_rgb.astype(np.float32) * mask_3ch + 
                 bg_with_text.astype(np.float32) * (1 - mask_3ch))
    
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    
    logger.info(f"Composite complete - shape: {composite.shape}")
    
    # Convert back to BGR
    result_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    
    return result_bgr


def remove_background(
    image_bgr: np.ndarray,
    mask_gray: np.ndarray,
    mode: str = "transparent",
    replacement_color: Optional[str] = None,
    replacement_image: Optional[np.ndarray] = None,
    gradient: Optional[dict] = None
) -> Union[np.ndarray, Image.Image]:
    """
    Remove or replace background.
    
    Args:
        image_bgr: Original image (BGR)
        mask_gray: Person segmentation mask (0-255)
        mode: "transparent", "color", "image", or "gradient"
        replacement_color: Hex color for "color" mode
        replacement_image: BGR image for "image" mode
        gradient: Dict with gradient params for "gradient" mode
    
    Returns:
        For transparent: PIL Image with alpha channel
        For others: numpy array (BGR)
    """
    h, w = image_bgr.shape[:2]
    
    # Ensure mask is single channel
    if len(mask_gray.shape) == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)
    
    # Feather mask
    mask_feathered = cv2.GaussianBlur(mask_gray, (5, 5), 0)
    
    if mode == "transparent":
        # Create RGBA with transparent background
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgba = np.dstack((rgb, mask_feathered))
        return Image.fromarray(rgba, 'RGBA')
    
    elif mode == "color":
        # Replace with solid color
        if replacement_color is None:
            replacement_color = "#FFFFFF"
        
        color_bgr = hex_to_bgr(replacement_color)
        background = np.full_like(image_bgr, color_bgr, dtype=np.uint8)
        
        # Composite
        mask_norm = mask_feathered.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_norm] * 3, axis=2)
        
        result = (image_bgr.astype(np.float32) * mask_3ch + 
                 background.astype(np.float32) * (1 - mask_3ch))
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    elif mode == "image":
        # Replace with image
        if replacement_image is None:
            raise ValueError("replacement_image required for 'image' mode")
        
        # Resize replacement to match
        if replacement_image.shape[:2] != (h, w):
            background = cv2.resize(replacement_image, (w, h))
        else:
            background = replacement_image
        
        # Composite
        mask_norm = mask_feathered.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_norm] * 3, axis=2)
        
        result = (image_bgr.astype(np.float32) * mask_3ch + 
                 background.astype(np.float32) * (1 - mask_3ch))
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    elif mode == "gradient":
        # Create gradient background
        background = create_gradient_background(w, h, gradient or {})
        
        # Composite
        mask_norm = mask_feathered.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_norm] * 3, axis=2)
        
        result = (image_bgr.astype(np.float32) * mask_3ch + 
                 background.astype(np.float32) * (1 - mask_3ch))
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def create_gradient_background(
    width: int,
    height: int,
    gradient_params: dict
) -> np.ndarray:
    """
    Create gradient background.
    
    gradient_params can include:
        - start_color: hex color
        - end_color: hex color
        - direction: "horizontal", "vertical", "diagonal", "radial"
    """
    start_color = gradient_params.get("start_color", "#000000")
    end_color = gradient_params.get("end_color", "#FFFFFF")
    direction = gradient_params.get("direction", "vertical")
    
    start_bgr = hex_to_bgr(start_color)
    end_bgr = hex_to_bgr(end_color)
    
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    
    if direction == "horizontal":
        for x in range(width):
            ratio = x / width
            color = tuple(int(start_bgr[i] * (1 - ratio) + end_bgr[i] * ratio) for i in range(3))
            gradient[:, x] = color
    
    elif direction == "vertical":
        for y in range(height):
            ratio = y / height
            color = tuple(int(start_bgr[i] * (1 - ratio) + end_bgr[i] * ratio) for i in range(3))
            gradient[y, :] = color
    
    elif direction == "diagonal":
        for y in range(height):
            for x in range(width):
                ratio = (x + y) / (width + height)
                color = tuple(int(start_bgr[i] * (1 - ratio) + end_bgr[i] * ratio) for i in range(3))
                gradient[y, x] = color
    
    elif direction == "radial":
        center_x, center_y = width // 2, height // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                ratio = min(dist / max_dist, 1.0)
                color = tuple(int(start_bgr[i] * (1 - ratio) + end_bgr[i] * ratio) for i in range(3))
                gradient[y, x] = color
    
    return gradient


def calculate_background_text_position(
    mask_gray: np.ndarray,
    text: str,
    font_size: int,
    font_path: Optional[str] = None
) -> Tuple[int, int]:
    """
    Calculate optimal position for text in background area.
    Finds center of largest background region.
    """
    h, w = mask_gray.shape[:2]
    
    # Invert mask (background = 255, person = 0)
    bg_mask = cv2.bitwise_not(mask_gray)
    
    # Find contours of background regions
    contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to center
        return (w // 2, h // 2)
    
    # Find largest background region
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get moments to find centroid
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = w // 2, h // 2
    
    return (cx, cy)
