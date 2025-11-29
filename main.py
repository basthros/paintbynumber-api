"""
Paint by Number API v2.0
Professional-grade paint-by-number template generator

Improvements over v1:
- Higher resolution output (2400px vs 800px)
- SVG vector output for crisp printing
- Color key generation
- Better region processing
- Overlay view support
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import numpy as np
import cv2
import json
import base64
from io import BytesIO
from typing import Optional
import logging

# Import our processor
from paint_processor import PaintByNumberProcessor, HAS_SVGWRITE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Paint by Number API",
    description="Professional-grade paint-by-number template generator",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration - update with your domains
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8080",
    "https://paint-by-number-new.onrender.com",
    "capacitor://localhost",
    "http://localhost",
    # Add your production domains here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor with production settings
processor = PaintByNumberProcessor(
    min_region_pixels=100,
    target_max_regions=200,
    line_thickness=2,
    output_size=2400,  # High res for printing
    min_font_size=10,
    max_font_size=20
)


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Paint by Number API",
        "version": "2.0.0",
        "features": {
            "svg_output": HAS_SVGWRITE,
            "max_resolution": processor.output_size
        }
    }


@app.get("/health")
async def health():
    """Alternative health check endpoint."""
    return {"status": "healthy"}


def validate_palette(palette_str: str) -> list:
    """Validate and parse palette JSON."""
    try:
        palette = json.loads(palette_str)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid palette JSON: {str(e)}"
        )
    
    if not isinstance(palette, list):
        raise HTTPException(
            status_code=400,
            detail="Palette must be a JSON array"
        )
    
    if len(palette) < 2:
        raise HTTPException(
            status_code=400,
            detail="Palette must have at least 2 colors"
        )
    
    # Validate each color
    for i, color in enumerate(palette):
        if not isinstance(color, dict):
            raise HTTPException(
                status_code=400,
                detail=f"Color {i} must be an object"
            )
        
        if 'id' not in color:
            raise HTTPException(
                status_code=400,
                detail=f"Color {i} missing 'id' field"
            )
        
        if 'rgb' not in color:
            raise HTTPException(
                status_code=400,
                detail=f"Color {i} missing 'rgb' field"
            )
        
        rgb = color['rgb']
        if not isinstance(rgb, list) or len(rgb) != 3:
            raise HTTPException(
                status_code=400,
                detail=f"Color {i} 'rgb' must be array of 3 integers"
            )
        
        if not all(isinstance(v, (int, float)) and 0 <= v <= 255 for v in rgb):
            raise HTTPException(
                status_code=400,
                detail=f"Color {i} RGB values must be 0-255"
            )
        
        # Ensure RGB values are integers
        color['rgb'] = [int(v) for v in rgb]
    
    return palette


def encode_image_to_base64(image: np.ndarray, format: str = 'png') -> str:
    """Encode numpy image to base64 string."""
    # Ensure image is BGR for cv2.imencode
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    if format.lower() == 'png':
        success, buffer = cv2.imencode('.png', image_bgr)
        mime = 'image/png'
    elif format.lower() in ['jpg', 'jpeg']:
        success, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        mime = 'image/jpeg'
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    if not success:
        raise RuntimeError("Failed to encode image")
    
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:{mime};base64,{b64}"


@app.post("/generate")
async def generate_paint_by_number(
    file: UploadFile = File(..., description="Image file (PNG, JPEG, WEBP)"),
    palette: str = Form(..., description="JSON array of colors"),
    threshold: int = Form(default=50, ge=1, le=100, description="Complexity 1-100")
):
    """
    Generate paint-by-number preview and template from an uploaded image.
    
    Parameters:
    - **file**: Image file (PNG, JPEG, WEBP, max 10MB)
    - **palette**: JSON array of colors. Each color: `{"id": "1", "rgb": [255, 0, 0], "note": "Red"}`
    - **threshold**: Complexity level 1-100 (higher = more detail/regions)
    
    Returns:
    - **preview**: Base64 PNG of colored preview
    - **template**: Base64 PNG of line art with numbers
    - **template_svg**: Base64 SVG of template (if available)
    - **color_key**: Base64 PNG of color legend
    - **dimensions**: Width and height
    - **region_count**: Number of paintable regions
    - **colors_used**: Number of colors from palette actually used
    """
    # Validate file type
    valid_types = ["image/png", "image/jpeg", "image/webp", "image/jpg"]
    if file.content_type not in valid_types:
        raise HTTPException(
            status_code=415, 
            detail=f"Unsupported file type: {file.content_type}. Use PNG, JPEG, or WEBP."
        )
    
    # Read file
    contents = await file.read()
    
    # Check file size (10MB limit)
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413, 
            detail="File too large. Maximum size is 10MB."
        )
    
    # Decode image
    try:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid image file: {str(e)}"
        )
    
    # Validate and parse palette
    palette_data = validate_palette(palette)
    
    # Process image
    try:
        logger.info(f"Processing image {file.filename}: {image.shape}, {len(palette_data)} colors, complexity={threshold}")
        result = processor.process(image, palette_data, threshold)
        logger.info(f"Generated {len(result.regions)} regions using {len(set(r.color_id for r in result.regions))} colors")
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Processing error: {str(e)}"
        )
    
    # Generate outputs
    response_data = {
        "success": True,
        "dimensions": {
            "width": result.dimensions[0],
            "height": result.dimensions[1]
        },
        "region_count": len(result.regions),
        "colors_used": len(set(r.color_id for r in result.regions))
    }
    
    # Preview image (colored)
    try:
        response_data["preview"] = encode_image_to_base64(result.preview_image, 'png')
    except Exception as e:
        logger.error(f"Preview encoding error: {e}")
        raise HTTPException(status_code=500, detail="Failed to encode preview")
    
    # Template image (line art with numbers)
    try:
        template_img = processor.generate_template_image(result, show_numbers=True)
        response_data["template"] = encode_image_to_base64(template_img, 'png')
    except Exception as e:
        logger.error(f"Template encoding error: {e}")
        raise HTTPException(status_code=500, detail="Failed to encode template")
    
    # SVG template (if available)
    if HAS_SVGWRITE:
        try:
            svg_str = processor.generate_template_svg(result, show_numbers=True)
            svg_b64 = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
            response_data["template_svg"] = f"data:image/svg+xml;base64,{svg_b64}"
            
            # Color key SVG
            key_svg = processor.generate_color_key_svg(result)
            key_b64 = base64.b64encode(key_svg.encode('utf-8')).decode('utf-8')
            response_data["color_key"] = f"data:image/svg+xml;base64,{key_b64}"
        except Exception as e:
            logger.warning(f"SVG generation error: {e}")
            # Non-fatal - continue without SVG
    
    # Generate PNG color key as fallback
    if "color_key" not in response_data:
        try:
            # Create simple color key image
            colors_used = list(set(r.color_id for r in result.regions))
            palette_in_use = [pc for pc in result.palette_used if pc.id in colors_used]
            
            swatch_size = 40
            padding = 10
            key_height = len(palette_in_use) * (swatch_size + padding) + padding
            key_width = 300
            
            key_img = np.ones((key_height, key_width, 3), dtype=np.uint8) * 255
            
            for i, pc in enumerate(palette_in_use):
                y = padding + i * (swatch_size + padding)
                # Draw swatch
                cv2.rectangle(
                    key_img,
                    (padding, y),
                    (padding + swatch_size, y + swatch_size),
                    pc.rgb[::-1],  # RGB to BGR
                    -1
                )
                cv2.rectangle(
                    key_img,
                    (padding, y),
                    (padding + swatch_size, y + swatch_size),
                    (0, 0, 0),
                    1
                )
                # Draw text
                label = f"{pc.id}: {pc.name}" if pc.name else f"Color {pc.id}"
                cv2.putText(
                    key_img,
                    label,
                    (padding * 2 + swatch_size, y + swatch_size // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
            
            response_data["color_key"] = encode_image_to_base64(
                cv2.cvtColor(key_img, cv2.COLOR_BGR2RGB), 'png'
            )
        except Exception as e:
            logger.warning(f"Color key generation error: {e}")
    
    return JSONResponse(response_data)


@app.post("/generate/svg")
async def generate_svg_only(
    file: UploadFile = File(...),
    palette: str = Form(...),
    threshold: int = Form(default=50, ge=1, le=100),
    show_numbers: bool = Form(default=True),
    fill_regions: bool = Form(default=False)
):
    """
    Generate SVG template only (for direct download).
    
    Returns raw SVG content with appropriate headers.
    """
    if not HAS_SVGWRITE:
        raise HTTPException(
            status_code=501,
            detail="SVG generation not available (svgwrite not installed)"
        )
    
    # Validate inputs
    valid_types = ["image/png", "image/jpeg", "image/webp", "image/jpg"]
    if file.content_type not in valid_types:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    
    try:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    palette_data = validate_palette(palette)
    
    try:
        result = processor.process(image, palette_data, threshold)
        svg_content = processor.generate_template_svg(
            result, 
            show_numbers=show_numbers,
            fill_regions=fill_regions
        )
    except Exception as e:
        logger.error(f"SVG generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    return Response(
        content=svg_content,
        media_type="image/svg+xml",
        headers={
            "Content-Disposition": "attachment; filename=paint-by-number.svg"
        }
    )


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    palette: str = Form(...)
):
    """
    Analyze an image against a palette without generating full output.
    
    Useful for quick feedback on palette coverage.
    """
    valid_types = ["image/png", "image/jpeg", "image/webp", "image/jpg"]
    if file.content_type not in valid_types:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    
    try:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    palette_data = validate_palette(palette)
    
    # Quick analysis - resize smaller for speed
    h, w = image.shape[:2]
    if max(h, w) > 400:
        scale = 400 / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    
    # Convert to LAB
    from skimage import color as skcolor
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = skcolor.rgb2lab(image_rgb.astype(np.float64) / 255.0)
    
    # Analyze color distribution
    pixels = image_lab.reshape(-1, 3)
    
    # For each palette color, find how many pixels are closest to it
    from scipy.spatial import cKDTree
    palette_lab = []
    for p in palette_data:
        rgb = np.array([[p['rgb']]], dtype=np.float64) / 255.0
        lab = skcolor.rgb2lab(rgb)[0, 0]
        palette_lab.append(lab)
    
    tree = cKDTree(palette_lab)
    distances, indices = tree.query(pixels, k=1)
    
    # Calculate coverage stats
    coverage = {}
    for i, p in enumerate(palette_data):
        count = np.sum(indices == i)
        coverage[p['id']] = {
            "pixel_count": int(count),
            "percentage": round(count / len(pixels) * 100, 2),
            "avg_distance": round(float(distances[indices == i].mean()) if count > 0 else 0, 2)
        }
    
    # Calculate overall fit
    avg_distance = float(distances.mean())
    max_distance = float(distances.max())
    
    return {
        "success": True,
        "image_dimensions": {"width": w, "height": h},
        "palette_coverage": coverage,
        "quality_metrics": {
            "average_color_distance": round(avg_distance, 2),
            "max_color_distance": round(max_distance, 2),
            "palette_fit_score": round(max(0, 100 - avg_distance * 2), 1)
        },
        "recommendations": _generate_recommendations(coverage, avg_distance)
    }


def _generate_recommendations(coverage: dict, avg_distance: float) -> list:
    """Generate recommendations based on analysis."""
    recs = []
    
    # Check for unused colors
    unused = [cid for cid, stats in coverage.items() if stats['percentage'] < 1]
    if unused:
        recs.append({
            "type": "unused_colors",
            "message": f"Colors {', '.join(unused)} cover less than 1% of the image. Consider removing them.",
            "severity": "info"
        })
    
    # Check for dominant colors
    dominant = [cid for cid, stats in coverage.items() if stats['percentage'] > 50]
    if dominant:
        recs.append({
            "type": "dominant_color",
            "message": f"Color {dominant[0]} dominates the image ({coverage[dominant[0]]['percentage']}%). Results may be less detailed.",
            "severity": "info"
        })
    
    # Check overall fit
    if avg_distance > 30:
        recs.append({
            "type": "poor_palette_fit",
            "message": "The palette doesn't match the image colors well. Consider adding colors closer to the image's actual colors.",
            "severity": "warning"
        })
    elif avg_distance > 20:
        recs.append({
            "type": "moderate_palette_fit",
            "message": "Some image colors don't have close palette matches. Results may show some color shifting.",
            "severity": "info"
        })
    
    return recs


# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
