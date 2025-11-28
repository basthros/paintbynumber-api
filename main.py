from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse  # ADD FileResponse here
from fastapi.staticfiles import StaticFiles  # ADD this line
import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.color import rgb2lab
import base64
import json
from typing import List, Dict
import io
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path 

app = FastAPI(title="Paint by Number API", version="1.0.0")

# CORS configuration for Capacitor mobile apps
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "development":
    origins = ["*"]
else:
    origins = [
        "https://yourdomain.com",
        "capacitor://localhost",  # iOS
        "http://localhost",        # Android
        "ionic://localhost",       # Alternative format
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
    expose_headers=["Content-Disposition"],
    max_age=3600,
)


def validate_image(file: UploadFile):
    """Validate image file type and size."""
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"]
    
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}"
        )


def decode_image(file_content: bytes) -> np.ndarray:
    """Decode image bytes to numpy array."""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(file_content, np.uint8)
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        return img
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image: {str(e)}"
        )


def resize_image(img: np.ndarray, max_dimension: int = 800) -> np.ndarray:
    """Resize image to max dimension while maintaining aspect ratio."""
    height, width = img.shape[:2]
    
    if max(height, width) <= max_dimension:
        return img
    
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized


def map_to_custom_palette(img_rgb: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    """
    Map image to custom palette using LAB color space and KD-tree.
    This provides perceptually accurate color matching.
    """
    # Convert to LAB color space (perceptually uniform)
    img_lab = rgb2lab(img_rgb / 255.0)
    palette_lab = rgb2lab(palette_rgb.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(palette_lab)
    
    # Query nearest colors for all pixels
    pixels = img_lab.reshape(-1, 3)
    distances, indices = tree.query(pixels, k=1)
    
    # Map to palette colors
    quantized = palette_rgb[indices].reshape(img_rgb.shape)
    
    return quantized.astype(np.uint8)


def simplify_regions(img: np.ndarray, threshold: int) -> np.ndarray:
    """
    Apply median blur and morphological operations to simplify regions.
    Higher threshold = more simplification = fewer, larger regions.
    """
    # Calculate kernel size based on threshold (must be odd)
    kernel_size = max(3, min(15, 3 + (threshold // 20) * 2))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Median blur for noise reduction while preserving edges
    blurred = cv2.medianBlur(img, kernel_size)
    
    # Morphological operations for region simplification
    morph_kernel_size = max(3, min(9, 3 + (threshold // 30) * 2))
    if morph_kernel_size % 2 == 0:
        morph_kernel_size += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    
    # Opening: remove small speckles
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Closing: fill small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return closed


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Automatic Canny edge detection using image statistics.
    No manual threshold tuning required!
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Compute automatic thresholds from median
    v = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, lower, upper, apertureSize=3, L2gradient=True)
    
    return edges


def generate_line_art(img: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Generate clean black-and-white line art template."""
    # Post-process edges for clean templates
    edge_kernel = np.ones((2, 2), np.uint8)
    
    # Close gaps in edges
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel)
    
    # Remove small segments
    edges_clean = cv2.morphologyEx(edges_closed, cv2.MORPH_OPEN, edge_kernel)
    
    # Create white canvas
    line_art = np.ones(img.shape[:2], dtype=np.uint8) * 255
    
    # Draw black edges
    line_art[edges_clean > 0] = 0
    
    # Convert to 3-channel for consistency
    line_art_color = cv2.cvtColor(line_art, cv2.COLOR_GRAY2BGR)
    
    return line_art_color


def encode_image_to_base64(img: np.ndarray, format: str = 'jpeg') -> str:
    """Encode numpy array image to base64 string."""
    # Convert BGR to RGB for proper encoding
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Encode to JPEG bytes
    success, buffer = cv2.imencode(f'.{format}', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    if not success:
        raise ValueError("Failed to encode image")
    
    # Convert to base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/{format};base64,{img_base64}"


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Paint by Number API",
        "version": "1.0.0"
    }


@app.post("/generate")
async def generate_paint_by_number(
    file: UploadFile = File(...),
    palette: str = Form(...),
    threshold: int = Form(50)
):
    """
    Generate paint-by-number preview and template.
    
    Args:
        file: Target image to convert
        palette: JSON string of custom colors [{'id': '1', 'rgb': [r, g, b], 'note': 'Recipe'}]
        threshold: Region simplification level (10-150)
    
    Returns:
        JSON with base64-encoded preview and template images
    """
    try:
        # Validate inputs
        validate_image(file)
        
        if threshold < 10 or threshold > 150:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Threshold must be between 10 and 150"
            )
        
        # Parse palette
        try:
            palette_data = json.loads(palette)
            if not isinstance(palette_data, list) or len(palette_data) == 0:
                raise ValueError("Palette must be a non-empty list")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid palette format: {str(e)}"
            )
        
        # Extract RGB values from palette
        palette_rgb = np.array([color['rgb'] for color in palette_data], dtype=np.uint8)
        
        # Read and decode image
        file_content = await file.read()
        img = decode_image(file_content)
        
        # Resize for faster processing
        img = resize_image(img, max_dimension=800)
        
        # Convert BGR to RGB (OpenCV uses BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Step 1: Map to custom palette using LAB color space
        quantized = map_to_custom_palette(img_rgb, palette_rgb)
        
        # Step 2: Simplify regions based on threshold
        simplified = simplify_regions(quantized, threshold)
        
        # Step 3: Generate edge detection for line art
        edges = auto_canny(simplified, sigma=0.33)
        
        # Step 4: Create line art template
        template = generate_line_art(simplified, edges)
        
        # Convert back to BGR for OpenCV encoding
        preview_bgr = cv2.cvtColor(simplified, cv2.COLOR_RGB2BGR)
        
        # Encode to base64
        preview_base64 = encode_image_to_base64(preview_bgr)
        template_base64 = encode_image_to_base64(template)
        
        return JSONResponse(content={
            "success": True,
            "preview": preview_base64,
            "template": template_base64,
            "dimensions": {
                "width": img.shape[1],
                "height": img.shape[0]
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# Serve static files (React app)
static_dir = Path(__file__).parent / "dist"
if static_dir.exists():
    # Serve static assets
    app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")
    
    # Catch-all route for React Router (must be AFTER all API routes)
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        """Serve React app for all non-API routes."""
        file_path = static_dir / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        # For React Router, return index.html
        return FileResponse(static_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
