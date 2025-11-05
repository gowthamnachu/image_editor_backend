"""
Utility functions for image processing
"""
import base64
import io
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Tuple, Optional


def base64_to_cv2(base64_str: str) -> np.ndarray:
    """Convert base64 string to OpenCV image (BGR)."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        img_data = base64.b64decode(base64_str)
        pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def read_imagefile_as_cv2(file_bytes: bytes) -> np.ndarray:
    """Read uploaded file bytes as OpenCV image (BGR)."""
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"Failed to read image file: {str(e)}")


def cv2_to_base64_png(cv2_img: np.ndarray) -> str:
    """Convert OpenCV image to base64 PNG string."""
    _, buf = cv2.imencode('.png', cv2_img)
    return base64.b64encode(buf).decode('utf-8')


def cv2_to_base64_jpeg(cv2_img: np.ndarray, quality: int = 95) -> str:
    """Convert OpenCV image to base64 JPEG string."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode('.jpg', cv2_img, encode_params)
    return base64.b64encode(buf).decode('utf-8')


def pil_to_base64_png(pil_img: Image.Image) -> str:
    """Convert PIL image to base64 PNG string."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def ensure_max_size(img: np.ndarray, max_width: int = 3840, max_height: int = 3840) -> np.ndarray:
    """Downscale image if it exceeds maximum dimensions."""
    h, w = img.shape[:2]
    
    if w <= max_width and h <= max_height:
        return img
    
    # Calculate scaling factor
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def feather_mask(mask: np.ndarray, feather_amount: int = 5) -> np.ndarray:
    """Apply Gaussian blur to mask edges for smooth blending."""
    if feather_amount <= 0:
        return mask
    
    # Ensure odd kernel size
    kernel_size = feather_amount * 2 + 1
    return cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to BGR tuple."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR


def apply_brightness_contrast(img: np.ndarray, brightness: float = 0, contrast: float = 1.0) -> np.ndarray:
    """
    Apply brightness and contrast adjustment.
    brightness: -100 to 100
    contrast: 0.5 to 3.0
    """
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def apply_saturation(img: np.ndarray, saturation: float = 1.0) -> np.ndarray:
    """
    Apply saturation adjustment.
    saturation: 0.0 (grayscale) to 2.0 (highly saturated)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_blur(img: np.ndarray, blur_amount: int = 5) -> np.ndarray:
    """Apply Gaussian blur."""
    if blur_amount <= 0:
        return img
    kernel_size = blur_amount * 2 + 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def apply_vignette(img: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Apply vignette effect."""
    h, w = img.shape[:2]
    
    # Create radial gradient
    center_x, center_y = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    
    # Calculate distance from center
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Normalize and apply intensity
    mask = 1 - (dist / max_dist) * intensity
    mask = np.clip(mask, 0, 1)
    
    # Apply to each channel
    result = img.astype(np.float32)
    for i in range(3):
        result[:, :, i] *= mask
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_sharpen(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Apply unsharp masking for sharpening."""
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return sharpened


def composite_images(foreground: np.ndarray, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Composite foreground and background using mask.
    mask: 0-255 single channel, where 255 = foreground, 0 = background
    """
    # Ensure mask is single channel
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Normalize mask to 0-1
    mask_norm = mask.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask_norm, mask_norm, mask_norm])
    
    # Ensure same size
    if foreground.shape[:2] != background.shape[:2]:
        background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
    
    # Composite
    result = (foreground.astype(np.float32) * mask_3ch + 
              background.astype(np.float32) * (1 - mask_3ch))
    
    return np.clip(result, 0, 255).astype(np.uint8)


def create_transparent_background(img: np.ndarray, mask: np.ndarray) -> Image.Image:
    """
    Create image with transparent background using mask.
    Returns PIL Image with alpha channel.
    """
    # Convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Ensure mask is single channel
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Create RGBA image
    rgba = np.dstack((rgb, mask))
    return Image.fromarray(rgba, 'RGBA')


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    """Wrap text to fit within max_width."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]
        
        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def get_default_font(size: int = 72) -> ImageFont.FreeTypeFont:
    """Get default font, fallback to PIL default if custom fonts not available."""
    try:
        # Try common system fonts
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
        
        # Fallback to default
        return ImageFont.load_default()
    except:
        return ImageFont.load_default()
