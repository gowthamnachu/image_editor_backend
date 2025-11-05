"""
Advanced image editing features using OpenCV
- Image upscaling (high-quality with OpenCV DNN Super Resolution when models are available)
- Object removal (inpainting) with adaptive mask refinement and soft blending
- Watermark removal with improved auto-detection
- Smart enhancement (denoise, white balance, CLAHE, sharpening)
- Noise reduction (auto method selection)
"""
import numpy as np
import cv2
from typing import Tuple, Optional, List
import logging
import os

# Try to import DNN Super Resolution if available in this build of OpenCV
try:
    from cv2 import dnn_superres
    _HAS_DNN_SUPERRES = True
except Exception:  # pragma: no cover - optional module
    _HAS_DNN_SUPERRES = False

# Try to import ximgproc for guided filtering (optional)
try:
    from cv2.ximgproc import guidedFilter as _cv_guided_filter  # type: ignore[attr-defined]
    _HAS_XIMGPROC = True
except Exception:  # pragma: no cover - optional module
    _HAS_XIMGPROC = False

logger = logging.getLogger(__name__)


# ----------------------------
# Helpers
# ----------------------------

def _safe_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _preprocess_mask(mask_gray: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Ensure a clean binary mask aligned to target shape with gentle dilation.

    Returns a uint8 mask with values in {0, 255}.
    """
    if mask_gray is None:
        raise ValueError("mask_gray is None")

    if len(mask_gray.shape) == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)

    # Resize to match target if needed
    th, tw = target_shape[:2]
    if mask_gray.shape[:2] != (th, tw):
        mask_gray = cv2.resize(mask_gray, (tw, th), interpolation=cv2.INTER_NEAREST)

    # Binarize
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    # Fill small holes and gently dilate proportional to image size
    h, w = mask_bin.shape[:2]
    k = max(3, int(min(h, w) * 0.006))  # ~0.6% of min dimension
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

    kd = max(3, int(min(h, w) * 0.004))  # ~0.4%
    if kd % 2 == 0:
        kd += 1
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
    mask_dilated = cv2.dilate(mask_closed, kernel_d, iterations=1)

    return mask_dilated.astype(np.uint8)


def _soft_blend(original: np.ndarray, edited: np.ndarray, mask: np.ndarray, feather_px: Optional[int] = None) -> np.ndarray:
    """Softly blend edited region back into original using a feathered mask.

    original/edited: BGR uint8 images of same shape.
    mask: uint8 binary mask (255 = edited region).
    feather_px: optional feather radius in pixels; auto if None.
    """
    original = _safe_uint8(original)
    edited = _safe_uint8(edited)

    h, w = mask.shape[:2]
    if feather_px is None:
        feather_px = max(3, int(0.01 * min(h, w)))  # 1% of min dimension
    if feather_px % 2 == 0:
        feather_px += 1

    # Create feathered alpha in [0,1]
    alpha = (mask.astype(np.float32) / 255.0)
    alpha = cv2.GaussianBlur(alpha, (feather_px, feather_px), 0)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = alpha[..., None]

    blended = (edited.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha))
    return np.clip(blended, 0, 255).astype(np.uint8)


def _refine_mask_contours(mask: np.ndarray, min_area_ratio: float = 0.0005) -> np.ndarray:
    """Remove tiny specks and fill holes via contour processing."""
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    h, w = mask.shape[:2]
    min_area = max(1, int(min_area_ratio * h * w))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(out, [cnt], -1, 255, thickness=cv2.FILLED)
    # Fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    return out


def _distance_alpha(mask: np.ndarray, max_feather_px: Optional[int] = None) -> np.ndarray:
    """Create a soft alpha ramp inside the mask using distance transform.
    Returns float32 alpha in [0,1] with 1.0 at the center of masked regions, ramping to 0.0 at edges.
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    inv = cv2.bitwise_not(mask_bin)
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 3)
    maxd = dist.max() if dist.size else 1.0
    alpha = dist / (maxd + 1e-6)
    if max_feather_px is not None and max_feather_px > 0:
        # Limit feather extent by clipping normalized distance
        # normalize distance to pixels then clip
        alpha = np.clip(dist / float(max_feather_px), 0.0, 1.0)
    return alpha.astype(np.float32)


def _harmonize_color_to_border(filled_bgr: np.ndarray, original_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Match color/statistics of the filled region to the border ring around the mask (Lab space).
    This reduces patchiness by aligning mean/std of a, b channels and gently matching L.
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Create a thin ring around mask to sample surrounding context
    dil = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    ring = cv2.subtract(dil, m)

    # Convert to Lab
    fill_lab = cv2.cvtColor(filled_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    orig_lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Stats in filled region and border ring
    def stats(arr: np.ndarray, mask_bin: np.ndarray):
        vals = [cv2.mean(arr[:, :, c], mask=mask_bin)[0] for c in range(3)]
        stds = [np.sqrt(cv2.mean((arr[:, :, c] - vals[c]) ** 2, mask=mask_bin)[0] + 1e-6) for c in range(3)]
        return np.array(vals, dtype=np.float32), np.array(stds, dtype=np.float32)

    m_bool = m > 0
    ring_bool = ring > 0
    if not np.any(m_bool) or not np.any(ring_bool):
        return filled_bgr

    mu_fill, sd_fill = stats(fill_lab, m)
    mu_ring, sd_ring = stats(orig_lab, ring)

    out = fill_lab.copy()
    # Match a and b channels more strongly, L gently
    for c, strength in zip([0, 1, 2], [0.4, 0.9, 0.9]):
        if sd_fill[c] > 1e-3:
            norm = (out[:, :, c] - mu_fill[c]) / sd_fill[c]
        else:
            norm = out[:, :, c] * 0.0
        target = norm * sd_ring[c] + mu_ring[c]
        out[:, :, c] = out[:, :, c] * (1.0 - strength) + target * strength

    out_bgr = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    # Only apply to filled region
    out_bgr = np.where(m[..., None] > 0, out_bgr, filled_bgr)
    return out_bgr


def _estimate_noise_strength(gray: np.ndarray) -> float:
    """Estimate noise level from Laplacian variance; lower variance -> likely smoother/noisier.
    Returns a normalized strength in [0.0, 1.0].
    """
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Heuristic normalization (empirical): 100..1500 typical
    return float(np.clip((300.0 - min(var, 300.0)) / 300.0, 0.0, 1.0))


def _auto_gamma(image_bgr: np.ndarray) -> float:
    """Compute a mild gamma to push mid-tones towards ~0.5."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # median in [0,255]
    med = np.median(gray) / 255.0
    if med <= 0:
        return 1.0
    gamma = np.clip(np.log(0.5) / np.log(med), 0.7, 1.4)
    return float(gamma)


def _apply_gamma(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-3:
        return image_bgr
    inv = 1.0 / max(gamma, 1e-6)
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image_bgr, lut)


def _guided_filter_smooth(guide_bgr: np.ndarray, src_bgr: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Edge-preserving smoothing of src guided by guide.
    Uses ximgproc.guidedFilter if available, else bilateral as a fallback.
    """
    guide = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY)
    if _HAS_XIMGPROC:
        try:
            # guidedFilter expects src to be single-channel or 3-channel float
            out = []
            for c in cv2.split(src_bgr):
                c_f = c.astype(np.float32) / 255.0
                gf = _cv_guided_filter(guide.astype(np.float32) / 255.0, c_f, radius, eps)
                out.append((np.clip(gf, 0, 1) * 255).astype(np.uint8))
            return cv2.merge(out)
        except Exception:
            pass
    # Bilateral fallback
    return cv2.bilateralFilter(src_bgr, d=radius * 2 + 1, sigmaColor=80, sigmaSpace=radius * 2)


def _load_superres(method: str, scale: int):
    """Try to load a DNN super-resolution model if available.
    Looks in ./models/superres/<MODEL>_x<scale>.pb, falls back to None if not found.
    """
    if not _HAS_DNN_SUPERRES:
        return None
    model_map = {
        "edsr": ("edsr", f"EDSR_x{scale}.pb"),
        "espcn": ("espcn", f"ESPCN_x{scale}.pb"),
        "fsrcnn": ("fsrcnn", f"FSRCNN_x{scale}.pb"),
        "lapsrn": ("lapsrn", f"LapSRN_x{scale}.pb"),
    }
    key = method.lower()
    if key not in model_map:
        return None
    algo_name, filename = model_map[key]
    base_dir = os.getenv("SUPERRES_MODEL_DIR", os.path.join(os.getcwd(), "models", "superres"))
    model_path = os.path.join(base_dir, filename)
    if not os.path.isfile(model_path):
        logger.warning(f"Superres model not found at {model_path}; falling back to interpolation.")
        return None
    try:
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel(algo_name, scale)
        return sr
    except Exception as e:
        logger.warning(f"Failed to load superres model: {e}; using interpolation fallback.")
        return None


def upscale_image(
    image_bgr: np.ndarray,
    scale_factor: float = 2.0,
    method: str = "edsr"
) -> np.ndarray:
    """
    Upscale image using various methods.
    
    Args:
        image_bgr: Input image (BGR)
        scale_factor: Upscaling factor (2, 3, or 4)
        method: Upscaling method ("edsr", "espcn", "fsrcnn", "lapsrn", "bicubic")
    
    Returns:
        Upscaled image (BGR)
    """
    h, w = image_bgr.shape[:2]
    image_bgr = _safe_uint8(image_bgr)

    # If a known DNN method is requested and available, try it first
    dnn_methods = {"edsr", "espcn", "fsrcnn", "lapsrn"}
    int_scale = int(round(scale_factor))
    int_scale = int(np.clip(int_scale, 2, 4))  # common SR scales

    result = None
    if method.lower() in dnn_methods:
        sr = _load_superres(method.lower(), int_scale)
        if sr is not None:
            try:
                result = sr.upsample(image_bgr)
            except Exception as e:
                logger.warning(f"Superres upsample failed: {e}; falling back to interpolation.")
                result = None

    if result is None:
        # High-quality interpolation fallback with mild detail enhancement
        new_size = (int(w * scale_factor), int(h * scale_factor))
        interp = cv2.INTER_LANCZOS4 if method.lower() == "lanczos" else cv2.INTER_CUBIC
        result = cv2.resize(image_bgr, new_size, interpolation=interp)

        # Gentle sharpening kernel to counteract blur
        sharp_k = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]], dtype=np.float32)
        result = cv2.filter2D(result, -1, sharp_k)
        result = _safe_uint8(result)

    logger.info(f"Upscaled from {w}x{h} to {result.shape[1]}x{result.shape[0]} using {method}")
    return result


def remove_object_inpainting(
    image_bgr: np.ndarray,
    mask_gray: np.ndarray,
    method: str = "telea"
) -> np.ndarray:
    """
    Remove objects from image using inpainting.
    
    Args:
        image_bgr: Input image (BGR)
        mask_gray: Mask where white (255) = areas to remove
        method: Inpainting method ("telea" or "navier_stokes")
    
    Returns:
        Inpainted image (BGR)
    """
    # Prepare mask aligned to image
    h, w = image_bgr.shape[:2]
    base_mask = _preprocess_mask(mask_gray, (h, w))
    base_mask = _refine_mask_contours(base_mask)

    # Adaptive radius based on image size
    radius = max(3, int(0.006 * min(h, w)))  # ~0.6% of min dimension

    # Try both algorithms and pick better near the boundary
    def _inpaint_with_algo(algo_flag: int) -> np.ndarray:
        # Multi-pass: coarse downscaled + fine full-res
        scale = 0.75
        small_size = (max(16, int(w * scale)), max(16, int(h * scale)))
        small_img = cv2.resize(image_bgr, small_size, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(base_mask, small_size, interpolation=cv2.INTER_NEAREST)
        small = cv2.inpaint(small_img, small_mask, radius, algo_flag)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        return cv2.inpaint(up, base_mask, radius, algo_flag)

    cand_telea = _inpaint_with_algo(cv2.INPAINT_TELEA)
    cand_ns = _inpaint_with_algo(cv2.INPAINT_NS)

    # Evaluate which candidate blends better at boundary (lower difference ring)
    ring = cv2.dilate(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    ring = cv2.subtract(ring, base_mask)
    def _boundary_score(candidate: np.ndarray) -> float:
        diff = cv2.absdiff(candidate, image_bgr)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        return float(cv2.mean(diff_gray, mask=ring)[0])

    score_telea = _boundary_score(cand_telea)
    score_ns = _boundary_score(cand_ns)
    refined = cand_telea if score_telea <= score_ns else cand_ns

    # Optional seamlessClone to reduce seams further
    try:
        x, y, wbb, hbb = cv2.boundingRect(base_mask)
        if wbb > 0 and hbb > 0:
            center = (x + wbb // 2, y + hbb // 2)
            # Ensure mask for cloning is 3-channel
            clone_mask = base_mask
            result_clone = cv2.seamlessClone(refined, image_bgr, clone_mask, center, cv2.NORMAL_CLONE)
            refined = result_clone
    except Exception as e:
        logger.debug(f"seamlessClone fallback: {e}")

    # Soft blend to avoid halos using distance alpha
    alpha = _distance_alpha(base_mask, max_feather_px=int(0.02 * min(h, w)))
    alpha_u8 = (alpha * 255).astype(np.uint8)
    # Color/contrast harmonization to surrounding
    refined = _harmonize_color_to_border(refined, image_bgr, base_mask)
    # Edge-preserving smoothing guided by the original image
    refined = _guided_filter_smooth(image_bgr, refined, radius=max(6, int(0.01 * min(h, w))), eps=1e-3)
    # Inject a bit of original scene detail near the boundary to hide low-frequency shifts
    ring = cv2.dilate(base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    ring = cv2.subtract(ring, base_mask)
    if np.any(ring > 0):
        # High-frequency detail from original
        blur = cv2.GaussianBlur(image_bgr, (0, 0), 1.0)
        detail = cv2.addWeighted(image_bgr, 1.0, blur, -1.0, 0)
        # Feathered ring alpha
        ring_alpha = cv2.GaussianBlur((ring / 255.0).astype(np.float32), (11, 11), 0)[..., None]
        refined = np.clip(refined.astype(np.float32) + detail.astype(np.float32) * 0.15 * ring_alpha, 0, 255).astype(np.uint8)
    result = _soft_blend(image_bgr, refined, alpha_u8)

    logger.info(f"Inpainting completed (auto-picked), radius={radius}, scores telea={score_telea:.2f}, ns={score_ns:.2f}")
    return result


def remove_watermark(
    image_bgr: np.ndarray,
    mask_gray: Optional[np.ndarray] = None,
    auto_detect: bool = True
) -> np.ndarray:
    """
    Remove watermark from image.
    
    Args:
        image_bgr: Input image (BGR)
        mask_gray: Optional mask of watermark location
        auto_detect: Try to automatically detect watermark
    
    Returns:
        Image with watermark removed (BGR)
    """
    if mask_gray is None and auto_detect:
        # Improved heuristic: combine HSV constraints, gradient magnitude, MSER text-like regions, and spatial priors
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        hch, sch, vch = cv2.split(hsv)

        # Low-saturation, high-value pixels are common for gray/white watermarks
        low_sat = cv2.threshold(sch, 40, 255, cv2.THRESH_BINARY_INV)[1]
        high_val = cv2.threshold(vch, 180, 255, cv2.THRESH_BINARY)[1]
        grayish = cv2.bitwise_and(low_sat, high_val)

        # Edge/gradient map
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gradx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grady = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(gradx), 0.5,
                                                   cv2.convertScaleAbs(grady), 0.5, 0))
        _, high_grad = cv2.threshold(grad, 40, 255, cv2.THRESH_BINARY)

        # MSER to catch text-like regions (if available)
        try:
            mser = cv2.MSER_create(_delta=5, _min_area=60, _max_area=20000)
            regions, _ = mser.detectRegions(gray)
            mser_mask = np.zeros_like(gray, dtype=np.uint8)
            for p in regions:
                hull = cv2.convexHull(p.reshape(-1, 1, 2))
                cv2.drawContours(mser_mask, [hull], -1, 255, -1)
        except Exception:
            mser_mask = np.zeros_like(gray, dtype=np.uint8)

        # Spatial priors (corners + center horizontal band)
        h, w = gray.shape
        mask_prior = np.zeros_like(gray, dtype=np.uint8)
        csz = max(16, min(h, w) // 5)
        corners = [
            (0, 0, csz, csz),
            (w - csz, 0, w, csz),
            (0, h - csz, csz, h),
            (w - csz, h - csz, w, h),
        ]
        for x1, y1, x2, y2 in corners:
            mask_prior[y1:y2, x1:x2] = 255
        band = max(4, min(h, w) // 50)
        cv2.line(mask_prior, (0, h // 2), (w - 1, h // 2), 255, band)

        # Combine cues
        candidate = cv2.bitwise_and(grayish, high_grad)
        candidate = cv2.bitwise_or(candidate, mser_mask)
        candidate = cv2.bitwise_or(candidate, cv2.bitwise_and(grayish, mask_prior))

        # Morphological refinement
        k = max(3, int(min(h, w) * 0.008))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, max(3, k // 3)))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)

        mask_gray = candidate
    
    if mask_gray is not None:
        # Refine mask
        refined_mask = _preprocess_mask(mask_gray, image_bgr.shape[:2])
        refined_mask = _refine_mask_contours(refined_mask)

        # Path 1: Inpainting-based fill
        inpainted = remove_object_inpainting(image_bgr, refined_mask, method="telea")

        # Path 2: Attenuation towards edge-preserved base to reduce semi-transparent logos
        try:
            base = cv2.edgePreservingFilter(image_bgr, flags=1, sigma_s=60, sigma_r=0.4)
        except Exception:
            base = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=50, sigmaSpace=50)

        # Create distance-based alpha (strong center, soft edges)
        alpha_dist = _distance_alpha(refined_mask, max_feather_px=int(0.03 * min(image_bgr.shape[:2])))  # up to 3%
        alpha_dist = alpha_dist[..., None]

        # Reduce saturation/contrast in watermark while pulling towards base
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask_float = (refined_mask.astype(np.float32) / 255.0)[..., None]

        # Attenuate saturation and value more in center
        atten_factor = 0.6 * alpha_dist + 0.2  # 0.2..0.8
        sat = sat * (1.0 - 0.7 * atten_factor.squeeze())
        val = val * (1.0 - 0.3 * atten_factor.squeeze())
        hsv[:, :, 1] = np.clip(sat, 0, 255)
        hsv[:, :, 2] = np.clip(val, 0, 255)
        atten_color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Pull colors towards base texture inside mask
        attenuated = (atten_color.astype(np.float32) * (1.0 - 0.5 * alpha_dist) + base.astype(np.float32) * (0.5 * alpha_dist)).astype(np.uint8)
        attenuated = np.where(refined_mask[..., None] > 0, attenuated, image_bgr)

        # Combine paths: favor inpaint in center, attenuation near edges for better continuity
        mix_alpha = (alpha_dist ** 1.0)  # linear center weight
        mixed = (inpainted.astype(np.float32) * mix_alpha + attenuated.astype(np.float32) * (1.0 - mix_alpha)).astype(np.uint8)

        # Color/contrast harmonization
        mixed = _harmonize_color_to_border(mixed, image_bgr, refined_mask)
        # Edge-preserving smoothing guided by original
        mixed = _guided_filter_smooth(image_bgr, mixed, radius=max(6, int(0.01 * min(image_bgr.shape[:2]))), eps=1e-3)
        # Inject slight detail near boundary
        ring = cv2.dilate(refined_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        ring = cv2.subtract(ring, refined_mask)
        if np.any(ring > 0):
            blur = cv2.GaussianBlur(image_bgr, (0, 0), 1.0)
            detail = cv2.addWeighted(image_bgr, 1.0, blur, -1.0, 0)
            ring_alpha = cv2.GaussianBlur((ring / 255.0).astype(np.float32), (11, 11), 0)[..., None]
            mixed = np.clip(mixed.astype(np.float32) + detail.astype(np.float32) * 0.12 * ring_alpha, 0, 255).astype(np.uint8)
        # Final soft blend over original to hide seams
        result = _soft_blend(image_bgr, mixed, refined_mask)
    else:
        result = image_bgr.copy()
        logger.warning("No watermark mask provided or detected")

    return result


def smart_enhance(
    image_bgr: np.ndarray,
    denoise: bool = True,
    sharpen: bool = True,
    color_balance: bool = True
) -> np.ndarray:
    """
    Apply smart enhancement to image.
    
    Args:
        image_bgr: Input image (BGR)
        denoise: Apply denoising
        sharpen: Apply sharpening
        color_balance: Apply color balance correction
    
    Returns:
        Enhanced image (BGR)
    """
    result = _safe_uint8(image_bgr.copy())

    # Denoise (auto strength based on estimated noise)
    if denoise:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        nlevel = _estimate_noise_strength(gray)
        h = int(5 + 20 * nlevel)  # 5..25
        result = cv2.fastNlMeansDenoisingColored(result, None, h, h, 7, 21)
        logger.info(f"Applied denoising (NLM) with h~{h}")

    # Color balance + CLAHE on L channel
    if color_balance:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Gray-world correction
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        avg_a = float(np.mean(lab[:, :, 1]))
        avg_b = float(np.mean(lab[:, :, 2]))
        lab[:, :, 1] = np.clip(lab[:, :, 1] - (avg_a - 128) * (lab[:, :, 0] / 255.0), 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] - (avg_b - 128) * (lab[:, :, 0] / 255.0), 0, 255)
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        logger.info("Applied color balance + CLAHE")

    # Mild gamma to normalize mid-tones
    gamma = _auto_gamma(result)
    result = _apply_gamma(result, gamma)

    # Sharpen (unsharp mask, conservative)
    if sharpen:
        blur = cv2.GaussianBlur(result, (0, 0), 1.5)
        result = cv2.addWeighted(result, 1.25, blur, -0.25, 0)
        result = _safe_uint8(result)
        logger.info("Applied sharpening")

    return result


def reduce_noise(
    image_bgr: np.ndarray,
    strength: int = 10,
    method: str = "nlm"
) -> np.ndarray:
    """
    Reduce noise in image.
    
    Args:
        image_bgr: Input image (BGR)
        strength: Noise reduction strength (1-30)
        method: Method ("nlm", "bilateral", "gaussian")
    
    Returns:
        Denoised image (BGR)
    """
    strength = int(np.clip(strength, 1, 30))

    img = _safe_uint8(image_bgr)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    auto_noise = _estimate_noise_strength(gray)  # 0..1

    if method == "nlm":
        h = int(np.interp(strength, [1, 30], [5, 25]))
        # Adjust by auto noise estimate
        h = int(h * (0.6 + 0.8 * auto_noise))
        result = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    elif method == "bilateral":
        sigma = int(np.interp(strength, [1, 30], [20, 100]))
        result = cv2.bilateralFilter(img, d=9, sigmaColor=sigma, sigmaSpace=sigma)
    else:  # gaussian
        ksize = max(3, (strength // 3) * 2 + 1)
        result = cv2.GaussianBlur(img, (ksize, ksize), 0)

    logger.info(f"Noise reduction applied: method={method}, strength={strength}, auto={auto_noise:.2f}")
    return result


def auto_correct_exposure(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0
) -> np.ndarray:
    """
    Automatically correct exposure using CLAHE.
    
    Args:
        image_bgr: Input image (BGR)
        clip_limit: Contrast limiting (1.0-4.0)
    
    Returns:
        Exposure-corrected image (BGR)
    """
    img = _safe_uint8(image_bgr)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=float(np.clip(clip_limit, 1.0, 4.0)), tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Auto gamma after CLAHE to normalize mid-tones
    gamma = _auto_gamma(result)
    result = _apply_gamma(result, gamma)

    logger.info(f"Auto exposure correction applied with clip_limit={clip_limit}, gamma={gamma:.2f}")
    return result


def remove_red_eye(
    image_bgr: np.ndarray,
    eye_regions: List[Tuple[int, int, int, int]]
) -> np.ndarray:
    """
    Remove red eye effect from photos.
    
    Args:
        image_bgr: Input image (BGR)
        eye_regions: List of (x, y, width, height) for each eye
    
    Returns:
        Image with red eye removed (BGR)
    """
    result = _safe_uint8(image_bgr.copy())

    for x, y, w, h in eye_regions:
        x0, y0, x1, y1 = max(0, x), max(0, y), min(result.shape[1], x + w), min(result.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        eye = result[y0:y1, x0:x1]

        hsv = cv2.cvtColor(eye, cv2.COLOR_BGR2HSV)
        # Red ranges
        lower_red1 = np.array([0, 60, 60])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([168, 60, 60])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Keep highlights (very bright pixels) to avoid dull eyes
        bright = hsv[:, :, 2] > 220
        mask[bright] = 0

        # Feather mask
        k = max(3, int(min(w, h) * 0.1))
        if k % 2 == 0:
            k += 1
        mask_blur = cv2.GaussianBlur(mask, (k, k), 0)
        alpha = (mask_blur.astype(np.float32) / 255.0)[..., None]

        # Desaturate and slightly darken only masked region
        hsv_corr = hsv.copy()
        hsv_corr[:, :, 1] = (hsv_corr[:, :, 1].astype(np.float32) * (1.0 - 0.8 * alpha.squeeze())).astype(np.uint8)
        hsv_corr[:, :, 2] = (hsv_corr[:, :, 2].astype(np.float32) * (1.0 - 0.1 * alpha.squeeze())).astype(np.uint8)
        corrected = cv2.cvtColor(hsv_corr, cv2.COLOR_HSV2BGR)

        eye_out = (corrected.astype(np.float32) * alpha + eye.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        result[y0:y1, x0:x1] = eye_out

    logger.info(f"Red eye removal applied to {len(eye_regions)} regions")
    return result


def perspective_correction(
    image_bgr: np.ndarray,
    corners: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Correct perspective distortion.
    
    Args:
        image_bgr: Input image (BGR)
        corners: Four corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                 in order: top-left, top-right, bottom-right, bottom-left
    
    Returns:
        Perspective-corrected image (BGR)
    """
    if len(corners) != 4:
        logger.error("Perspective correction requires exactly 4 corners")
        return image_bgr
    
    # Convert to numpy array
    src_points = np.float32(corners)

    # Calculate width and height of output
    width_top = np.linalg.norm(src_points[1] - src_points[0])
    width_bottom = np.linalg.norm(src_points[2] - src_points[3])
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(src_points[3] - src_points[0])
    height_right = np.linalg.norm(src_points[2] - src_points[1])
    max_height = int(max(height_left, height_right))

    # Avoid degenerate output size
    max_width = max(10, max_width)
    max_height = max(10, max_height)

    # Destination points
    dst_points = np.float32([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ])

    # Get perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply transformation with high-quality interpolation
    result = cv2.warpPerspective(image_bgr, matrix, (max_width, max_height), flags=cv2.INTER_LANCZOS4)

    logger.info(f"Perspective correction applied: {max_width}x{max_height}")
    return result
