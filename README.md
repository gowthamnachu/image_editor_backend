# Image Editor Backend

Production-quality FastAPI backend for image editing with AI-powered person segmentation.

## Features

- **Person Segmentation**: Mediapipe SelfieSegmentation with GrabCut fallback
- **Text Behind Person**: Place text on background behind detected person with antialiasing
- **Background Removal**: Transparent PNG, solid color, gradient, or custom image replacement
- **Filters**: Brightness, contrast, saturation, blur, vignette, grayscale, sharpen
- **Mask Refinement**: User-drawn strokes to refine segmentation
- **Stateless**: No database, no persistent storage

## Requirements

- Python 3.8+
- pip

## Installation

```powershell
cd backend
pip install -r requirements.txt
```

## Running the Server

### Development Mode

```powershell
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Production Mode

```powershell
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

For Windows production deployment, use:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
```

## API Endpoints

### Health Check

```
GET /health
```

Returns server status and Mediapipe availability.

### Segment Person

```
POST /api/segment
```

**Multipart File Upload:**
```
Content-Type: multipart/form-data
file: <image file>
```

**JSON Request:**
```json
{
  "imageBase64": "base64_encoded_image",
  "method": "auto",  // "auto", "mediapipe", or "grabcut"
  "threshold": 0.5
}
```

**Response:**
```json
{
  "maskBase64": "base64_encoded_mask",
  "previewBase64": "base64_encoded_preview",
  "confidence": 0.99,
  "method": "mediapipe",
  "width": 1920,
  "height": 1080
}
```

### Refine Mask

```
POST /api/refine-mask
```

**Request:**
```json
{
  "imageBase64": "base64_encoded_image",
  "maskBase64": "base64_encoded_mask",
  "fgStrokesBase64": "base64_encoded_foreground_strokes",  // optional
  "bgStrokesBase64": "base64_encoded_background_strokes"   // optional
}
```

### Remove Background

```
POST /api/remove-bg
```

**Request:**
```json
{
  "imageBase64": "base64_encoded_image",
  "maskBase64": "base64_encoded_mask",
  "mode": "transparent",  // "transparent", "color", "image", "gradient"
  "color": "#FFFFFF",     // for "color" mode
  "bgImageBase64": "...", // for "image" mode
  "gradient": {           // for "gradient" mode
    "start_color": "#000000",
    "end_color": "#FFFFFF",
    "direction": "vertical"
  }
}
```

### Place Text Behind Person

```
POST /api/place-text-behind
```

**Request:**
```json
{
  "imageBase64": "base64_encoded_image",
  "maskBase64": "base64_encoded_mask",
  "text": "Hello World",
  "x": 100,
  "y": 200,
  "fontSize": 72,
  "color": "#FFFFFF",
  "bgColor": "#000000",    // optional
  "bgImageBase64": "...",  // optional
  "align": "left",         // "left", "center", "right"
  "lineSpacing": 10,
  "autoPosition": false
}
```

### Advanced Processing

```
POST /api/advanced-process
```

**Request:**
```json
{
  "imageBase64": "base64_encoded_image",
  "ops": [
    {
      "op": "brightness_contrast",
      "params": {
        "brightness": 10,
        "contrast": 1.2
      }
    },
    {
      "op": "saturation",
      "params": {
        "saturation": 1.3
      }
    },
    {
      "op": "blur",
      "params": {
        "amount": 5
      }
    }
  ]
}
```

**Supported operations:**
- `brightness_contrast`: brightness (-100 to 100), contrast (0.5 to 3.0)
- `saturation`: saturation (0.0 to 2.0)
- `blur`: amount (0 to 20)
- `vignette`: intensity (0.0 to 1.0)
- `grayscale`: no params
- `sharpen`: amount (0.0 to 2.0)

## Architecture

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app and endpoints
│   ├── segmentation.py   # Mediapipe & GrabCut algorithms
│   ├── compositing.py    # Text placement & background ops
│   └── utils.py          # Image utilities and filters
└── requirements.txt
```

## Key Algorithms

### Person Segmentation

1. **Primary**: Mediapipe SelfieSegmentation (model_selection=1)
   - Fast and accurate for person detection
   - Returns float mask (0..1) converted to 8-bit
   - Applied morphological operations for cleanup
   - Gaussian blur for edge feathering

2. **Fallback**: OpenCV GrabCut
   - Automatic bounding box detection via edge detection
   - 5 iterations for refinement
   - Morphological operations for mask cleanup

### Text Behind Person

1. Convert mask to alpha channel with feathering
2. Create background layer (solid color, gradient, or original background)
3. Render text on background using PIL ImageDraw with antialiasing
4. Composite: `person * mask + background_with_text * (1 - mask)`
5. Result preserves person in foreground with text behind

### Background Removal

- **Transparent**: Create RGBA with mask as alpha channel
- **Color/Gradient/Image**: Composite foreground with custom background using mask

## Performance

- Images auto-scaled to max 3840px width
- Request size limit: 12MB
- Typical segmentation time: <1s for 1280px wide image (with Mediapipe)
- GrabCut fallback: 2-5s depending on image size

## Error Handling

All endpoints return proper HTTP status codes:
- `200`: Success
- `400`: Invalid request (bad image data, missing parameters)
- `413`: Request too large (>12MB)
- `500`: Server error (segmentation failed, etc.)

Error responses include detail message:
```json
{
  "detail": "Error description"
}
```

## Testing

Run a quick test:

```powershell
cd backend
python -c "import mediapipe; print('Mediapipe OK')"
python -c "import cv2; print('OpenCV OK')"
```

## Deployment

For production:

1. Use `uvicorn` with multiple workers (Unix) or single worker (Windows)
2. Set up reverse proxy (nginx) for SSL and load balancing
3. Configure CORS for specific origins in `main.py`
4. Set request size limits in reverse proxy
5. Monitor logs and set up health check endpoint polling

## Troubleshooting

**Mediapipe not available**: Backend will automatically fall back to GrabCut

**Font errors**: Will use system default font if specified font not found

**Memory issues**: Reduce max image size in `utils.py` `ensure_max_size()`

**Slow performance**: Use Mediapipe for fast segmentation; consider running on GPU

## License

MIT
