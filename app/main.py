"""
FastAPI main application for image editor backend
Production-quality endpoints with proper error handling
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
import io
import base64
import logging
from PIL import Image

from .utils import (
    base64_to_cv2, read_imagefile_as_cv2, cv2_to_base64_png, 
    cv2_to_base64_jpeg, pil_to_base64_png, ensure_max_size,
    apply_brightness_contrast, apply_saturation, apply_blur,
    apply_vignette, apply_grayscale, apply_sharpen, hex_to_bgr
)
from .segmentation import segment_person, refine_mask_with_strokes, MEDIAPIPE_AVAILABLE
from .compositing import (
    place_text_behind_person, remove_background, 
    calculate_background_text_position
)
from .advanced_editing import (
    upscale_image, remove_object_inpainting, remove_watermark,
    smart_enhance, reduce_noise, auto_correct_exposure,
    perspective_correction
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Image Editor API",
    description="Production-quality image editing API with person segmentation and compositing",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://image-editor-ai.netlify.app",  # Production Netlify frontend
        "http://localhost:3000",  # Local development
        "http://localhost:5173",  # Vite dev server
        "*"  # Allow all origins (can remove in production for security)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request size limit (12 MB)
MAX_REQUEST_SIZE = 12 * 1024 * 1024

# Pydantic models for request/response
class SegmentRequest(BaseModel):
    imageBase64: str
    method: str = "auto"
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)  # Lower default for better coverage

class SegmentResponse(BaseModel):
    maskBase64: str
    previewBase64: str
    confidence: float
    method: str
    width: int
    height: int

class RemoveBackgroundRequest(BaseModel):
    imageBase64: str
    maskBase64: str
    mode: str = Field(default="transparent", pattern="^(transparent|color|image|gradient)$")
    color: Optional[str] = None
    bgImageBase64: Optional[str] = None
    gradient: Optional[Dict[str, Any]] = None

class RemoveBackgroundResponse(BaseModel):
    resultBase64: str
    format: str

class PlaceTextBehindRequest(BaseModel):
    imageBase64: str
    maskBase64: str
    text: str
    x: int = 100
    y: int = 200
    fontSize: int = 72
    fontFamily: str = "Arial"
    fontWeight: str = "normal"
    fontStyle: str = "normal"
    fontPath: Optional[str] = None
    color: str = "#FFFFFF"
    bgColor: Optional[str] = None
    bgImageBase64: Optional[str] = None
    maxWidth: Optional[int] = None
    align: str = Field(default="left", pattern="^(left|center|right)$")
    lineSpacing: int = 10
    letterSpacing: int = 0
    autoPosition: bool = False
    # Text effects
    stroke: bool = False
    strokeColor: str = "#000000"
    strokeWidth: int = 2
    shadow: bool = False
    shadowColor: str = "#000000"
    shadowBlur: int = 10
    shadowOffsetX: int = 5
    shadowOffsetY: int = 5
    textTransform: str = "none"

class PlaceTextBehindResponse(BaseModel):
    resultBase64: str

class FilterOperation(BaseModel):
    op: str
    params: Dict[str, Any] = {}

class AdvancedProcessRequest(BaseModel):
    imageBase64: str
    ops: List[FilterOperation]

class AdvancedProcessResponse(BaseModel):
    resultBase64: str

class RefineMaskRequest(BaseModel):
    imageBase64: str
    maskBase64: str
    fgStrokesBase64: Optional[str] = None
    bgStrokesBase64: Optional[str] = None

class RefineMaskResponse(BaseModel):
    maskBase64: str


# Middleware to check request size
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_SIZE:
        raise HTTPException(status_code=413, detail="Request too large")
    return await call_next(request)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Image Editor API",
        "version": "1.0.0",
        "status": "healthy",
        "mediapipe_available": MEDIAPIPE_AVAILABLE
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mediapipe": MEDIAPIPE_AVAILABLE
    }


@app.post("/api/segment", response_model=SegmentResponse)
async def segment_endpoint(request: SegmentRequest):
    """
    Segment person from image using Mediapipe or GrabCut.
    Accepts JSON with base64 image.
    """
    try:
        # Get image from base64
        if not request.imageBase64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        img = base64_to_cv2(request.imageBase64)
        
        # Ensure reasonable size
        img = ensure_max_size(img)
        h, w = img.shape[:2]
        
        # Get parameters
        method = request.method
        threshold = request.threshold
        
        # Segment
        mask, method_used = segment_person(img, method=method, threshold=threshold)
        
        # Calculate confidence (simple metric based on mask coverage)
        mask_coverage = np.count_nonzero(mask) / (h * w)
        confidence = min(0.99, mask_coverage * 1.2)  # Scale for better UX
        
        # Convert to base64
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_b64 = cv2_to_base64_png(mask_bgr)
        preview_b64 = cv2_to_base64_png(img)
        
        logger.info(f"Segmentation successful: {method_used}, {w}x{h}, confidence: {confidence:.2f}")
        
        return SegmentResponse(
            maskBase64=mask_b64,
            previewBase64=preview_b64,
            confidence=confidence,
            method=method_used,
            width=w,
            height=h
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/api/refine-mask", response_model=RefineMaskResponse)
async def refine_mask_endpoint(request: RefineMaskRequest):
    """
    Refine segmentation mask using user-drawn foreground/background strokes.
    """
    try:
        img = base64_to_cv2(request.imageBase64)
        mask = base64_to_cv2(request.maskBase64)
        
        # Convert mask to grayscale if needed
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Get strokes if provided
        fg_strokes = None
        bg_strokes = None
        
        if request.fgStrokesBase64:
            fg_strokes = base64_to_cv2(request.fgStrokesBase64)
            if len(fg_strokes.shape) == 3:
                fg_strokes = cv2.cvtColor(fg_strokes, cv2.COLOR_BGR2GRAY)
        
        if request.bgStrokesBase64:
            bg_strokes = base64_to_cv2(request.bgStrokesBase64)
            if len(bg_strokes.shape) == 3:
                bg_strokes = cv2.cvtColor(bg_strokes, cv2.COLOR_BGR2GRAY)
        
        # Refine mask
        refined_mask = refine_mask_with_strokes(img, mask, fg_strokes, bg_strokes)
        
        # Convert to base64
        mask_bgr = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)
        mask_b64 = cv2_to_base64_png(mask_bgr)
        
        logger.info("Mask refinement successful")
        
        return RefineMaskResponse(maskBase64=mask_b64)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Mask refinement error: {e}")
        raise HTTPException(status_code=500, detail=f"Mask refinement failed: {str(e)}")


@app.post("/api/remove-bg", response_model=RemoveBackgroundResponse)
async def remove_background_endpoint(request: RemoveBackgroundRequest):
    """
    Remove or replace background.
    """
    try:
        img = base64_to_cv2(request.imageBase64)
        mask = base64_to_cv2(request.maskBase64)
        
        # Convert mask to grayscale
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Get background replacement if provided
        bg_image = None
        if request.bgImageBase64:
            bg_image = base64_to_cv2(request.bgImageBase64)
        
        # Remove/replace background
        result = remove_background(
            img, mask,
            mode=request.mode,
            replacement_color=request.color,
            replacement_image=bg_image,
            gradient=request.gradient
        )
        
        # Convert to base64
        if isinstance(result, Image.Image):
            # Transparent PNG
            result_b64 = pil_to_base64_png(result)
            format_type = "png"
        else:
            # BGR image
            result_b64 = cv2_to_base64_png(result)
            format_type = "png"
        
        logger.info(f"Background removal successful: mode={request.mode}")
        
        return RemoveBackgroundResponse(
            resultBase64=result_b64,
            format=format_type
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Background removal error: {e}")
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")


@app.post("/api/place-text-behind", response_model=PlaceTextBehindResponse)
async def place_text_behind_endpoint(request: PlaceTextBehindRequest):
    """
    Place text behind person in image.
    """
    try:
        img = base64_to_cv2(request.imageBase64)
        mask = base64_to_cv2(request.maskBase64)
        
        # Convert mask to grayscale
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Calculate position if auto
        if request.autoPosition:
            x, y = calculate_background_text_position(mask, request.text, request.fontSize, request.fontPath)
        else:
            x, y = request.x, request.y
        
        # Get background if provided
        bg_image = None
        if request.bgImageBase64:
            bg_image = base64_to_cv2(request.bgImageBase64)
        
        # Debug: Log mask statistics
        logger.info(f"Mask stats: min={mask.min()}, max={mask.max()}, mean={mask.mean():.2f}, shape={mask.shape}")
        
        # Place text
        result = place_text_behind_person(
            img, mask,
            text=request.text,
            position=(x, y),
            font_size=request.fontSize,
            font_family=request.fontFamily,
            font_weight=request.fontWeight,
            font_style=request.fontStyle,
            font_path=request.fontPath,
            color=request.color,
            bg_color=request.bgColor,
            bg_image=bg_image,
            max_width=request.maxWidth,
            align=request.align,
            line_spacing=request.lineSpacing,
            letter_spacing=request.letterSpacing,
            stroke=request.stroke,
            stroke_color=request.strokeColor,
            stroke_width=request.strokeWidth,
            shadow=request.shadow,
            shadow_color=request.shadowColor,
            shadow_blur=request.shadowBlur,
            shadow_offset_x=request.shadowOffsetX,
            shadow_offset_y=request.shadowOffsetY,
            text_transform=request.textTransform
        )
        
        # Convert to base64
        result_b64 = cv2_to_base64_png(result)
        
        logger.info(f"Text placement successful: '{request.text[:20]}...'")
        
        return PlaceTextBehindResponse(resultBase64=result_b64)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Text placement error: {e}")
        raise HTTPException(status_code=500, detail=f"Text placement failed: {str(e)}")


@app.post("/api/advanced-process", response_model=AdvancedProcessResponse)
async def advanced_process_endpoint(request: AdvancedProcessRequest):
    """
    Apply pipeline of image processing operations.
    """
    try:
        img = base64_to_cv2(request.imageBase64)
        
        # Apply operations in sequence
        for op_spec in request.ops:
            op = op_spec.op
            params = op_spec.params
            
            if op == "brightness_contrast":
                brightness = params.get("brightness", 0)
                contrast = params.get("contrast", 1.0)
                img = apply_brightness_contrast(img, brightness, contrast)
            
            elif op == "saturation":
                saturation = params.get("saturation", 1.0)
                img = apply_saturation(img, saturation)
            
            elif op == "blur":
                amount = params.get("amount", 5)
                img = apply_blur(img, amount)
            
            elif op == "vignette":
                intensity = params.get("intensity", 0.5)
                img = apply_vignette(img, intensity)
            
            elif op == "grayscale":
                img = apply_grayscale(img)
            
            elif op == "sharpen":
                amount = params.get("amount", 1.0)
                img = apply_sharpen(img, amount)
            
            else:
                logger.warning(f"Unknown operation: {op}")
        
        # Convert to base64
        result_b64 = cv2_to_base64_png(img)
        
        logger.info(f"Advanced processing successful: {len(request.ops)} operations")
        
        return AdvancedProcessResponse(resultBase64=result_b64)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Advanced processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/test-segmentation")
async def test_segmentation_endpoint(file: UploadFile = File(...)):
    """
    Test endpoint to help debug segmentation quality.
    Returns multiple versions with different thresholds.
    """
    try:
        data = await file.read()
        img = read_imagefile_as_cv2(data)
        img = ensure_max_size(img)
        
        results = {}
        
        # Test with different thresholds
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            mask, method = segment_person(img, method="auto", threshold=threshold)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            results[f"threshold_{threshold}"] = cv2_to_base64_png(mask_bgr)
        
        # Also return original
        results["original"] = cv2_to_base64_png(img)
        
        return {"results": results, "message": "Try different thresholds to find the best one"}
        
    except Exception as e:
        logger.error(f"Test segmentation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ADVANCED EDITING ENDPOINTS
# ============================================================================

class UpscaleRequest(BaseModel):
    imageBase64: str
    scaleFactor: float = Field(default=2.0, ge=1.5, le=4.0)
    method: str = Field(default="bicubic")  # bicubic, lanczos, super_resolution


class UpscaleResponse(BaseModel):
    resultBase64: str
    originalWidth: int
    originalHeight: int
    newWidth: int
    newHeight: int


@app.post("/api/upscale", response_model=UpscaleResponse)
async def upscale_image_endpoint(request: UpscaleRequest):
    """
    Upscale image using various methods.
    Methods: bicubic (fast), lanczos (high-quality), super_resolution (enhanced)
    """
    try:
        img = base64_to_cv2(request.imageBase64)
        orig_h, orig_w = img.shape[:2]
        
        # Upscale
        result = upscale_image(img, request.scaleFactor, request.method)
        new_h, new_w = result.shape[:2]
        
        # Convert to base64
        result_b64 = cv2_to_base64_png(result)
        
        logger.info(f"Upscale successful: {orig_w}x{orig_h} -> {new_w}x{new_h}, method={request.method}")
        
        return UpscaleResponse(
            resultBase64=result_b64,
            originalWidth=orig_w,
            originalHeight=orig_h,
            newWidth=new_w,
            newHeight=new_h
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upscale error: {e}")
        raise HTTPException(status_code=500, detail=f"Upscale failed: {str(e)}")


class ObjectRemovalRequest(BaseModel):
    imageBase64: str
    maskBase64: str  # White areas will be removed/inpainted
    method: str = Field(default="telea")  # telea or navier_stokes


class ObjectRemovalResponse(BaseModel):
    resultBase64: str


@app.post("/api/remove-object", response_model=ObjectRemovalResponse)
async def remove_object_endpoint(request: ObjectRemovalRequest):
    """
    Remove objects from image using inpainting.
    User draws a mask over the object to remove.
    """
    try:
        img = base64_to_cv2(request.imageBase64)
        mask = base64_to_cv2(request.maskBase64)
        
        # Convert mask to grayscale if needed
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Remove object
        result = remove_object_inpainting(img, mask, request.method)
        
        # Convert to base64
        result_b64 = cv2_to_base64_png(result)
        
        logger.info(f"Object removal successful: method={request.method}")
        
        return ObjectRemovalResponse(resultBase64=result_b64)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Object removal error: {e}")
        raise HTTPException(status_code=500, detail=f"Object removal failed: {str(e)}")


class WatermarkRemovalRequest(BaseModel):
    imageBase64: str
    maskBase64: Optional[str] = None  # Optional mask of watermark location
    autoDetect: bool = True


class WatermarkRemovalResponse(BaseModel):
    resultBase64: str


@app.post("/api/remove-watermark", response_model=WatermarkRemovalResponse)
async def remove_watermark_endpoint(request: WatermarkRemovalRequest):
    """
    Remove watermark from image.
    Can auto-detect watermark in corners or use provided mask.
    """
    try:
        img = base64_to_cv2(request.imageBase64)
        
        mask = None
        if request.maskBase64:
            mask = base64_to_cv2(request.maskBase64)
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Remove watermark
        result = remove_watermark(img, mask, request.autoDetect)
        
        # Convert to base64
        result_b64 = cv2_to_base64_png(result)
        
        logger.info(f"Watermark removal successful: auto_detect={request.autoDetect}")
        
        return WatermarkRemovalResponse(resultBase64=result_b64)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Watermark removal error: {e}")
        raise HTTPException(status_code=500, detail=f"Watermark removal failed: {str(e)}")


class SmartEnhanceRequest(BaseModel):
    imageBase64: str
    denoise: bool = True
    sharpen: bool = True
    colorBalance: bool = True


class SmartEnhanceResponse(BaseModel):
    resultBase64: str


@app.post("/api/smart-enhance", response_model=SmartEnhanceResponse)
async def smart_enhance_endpoint(request: SmartEnhanceRequest):
    """
    Apply smart enhancement to image.
    Includes denoising, sharpening, and color balance.
    """
    try:
        img = base64_to_cv2(request.imageBase64)
        
        # Apply smart enhancement
        result = smart_enhance(img, request.denoise, request.sharpen, request.colorBalance)
        
        # Convert to base64
        result_b64 = cv2_to_base64_png(result)
        
        logger.info(f"Smart enhance successful: denoise={request.denoise}, sharpen={request.sharpen}, color={request.colorBalance}")
        
        return SmartEnhanceResponse(resultBase64=result_b64)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Smart enhance error: {e}")
        raise HTTPException(status_code=500, detail=f"Smart enhance failed: {str(e)}")


class NoiseReductionRequest(BaseModel):
    imageBase64: str
    strength: int = Field(default=10, ge=1, le=30)
    method: str = Field(default="nlm")  # nlm, bilateral, gaussian


class NoiseReductionResponse(BaseModel):
    resultBase64: str


@app.post("/api/reduce-noise", response_model=NoiseReductionResponse)
async def reduce_noise_endpoint(request: NoiseReductionRequest):
    """
    Reduce noise in image.
    Methods: nlm (best quality), bilateral (preserves edges), gaussian (fastest)
    """
    try:
        img = base64_to_cv2(request.imageBase64)
        
        # Reduce noise
        result = reduce_noise(img, request.strength, request.method)
        
        # Convert to base64
        result_b64 = cv2_to_base64_png(result)
        
        logger.info(f"Noise reduction successful: strength={request.strength}, method={request.method}")
        
        return NoiseReductionResponse(resultBase64=result_b64)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Noise reduction error: {e}")
        raise HTTPException(status_code=500, detail=f"Noise reduction failed: {str(e)}")


class AutoExposureRequest(BaseModel):
    imageBase64: str
    clipLimit: float = Field(default=2.0, ge=1.0, le=4.0)


class AutoExposureResponse(BaseModel):
    resultBase64: str


@app.post("/api/auto-exposure", response_model=AutoExposureResponse)
async def auto_exposure_endpoint(request: AutoExposureRequest):
    """
    Automatically correct exposure using CLAHE.
    """
    try:
        img = base64_to_cv2(request.imageBase64)
        
        # Auto correct exposure
        result = auto_correct_exposure(img, request.clipLimit)
        
        # Convert to base64
        result_b64 = cv2_to_base64_png(result)
        
        logger.info(f"Auto exposure successful: clip_limit={request.clipLimit}")
        
        return AutoExposureResponse(resultBase64=result_b64)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Auto exposure error: {e}")
        raise HTTPException(status_code=500, detail=f"Auto exposure failed: {str(e)}")


class PerspectiveRequest(BaseModel):
    imageBase64: str
    corners: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


class PerspectiveResponse(BaseModel):
    resultBase64: str


@app.post("/api/perspective-correction", response_model=PerspectiveResponse)
async def perspective_correction_endpoint(request: PerspectiveRequest):
    """
    Correct perspective distortion.
    Provide 4 corner points: top-left, top-right, bottom-right, bottom-left
    """
    try:
        img = base64_to_cv2(request.imageBase64)
        
        # Convert corners to tuples
        corners = [(int(p[0]), int(p[1])) for p in request.corners]
        
        if len(corners) != 4:
            raise ValueError("Exactly 4 corner points required")
        
        # Apply perspective correction
        result = perspective_correction(img, corners)
        
        # Convert to base64
        result_b64 = cv2_to_base64_png(result)
        
        logger.info(f"Perspective correction successful")
        
        return PerspectiveResponse(resultBase64=result_b64)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Perspective correction error: {e}")
        raise HTTPException(status_code=500, detail=f"Perspective correction failed: {str(e)}")


# Entry point for uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

