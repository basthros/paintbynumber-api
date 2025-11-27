from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
        "*",  # Allow all for combined service
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
    Higher threshold = MORE detail (smaller blur kernels).
    Lower threshold = LESS detail (larger blur kernels, simpler).
    """
    # Invert threshold: 10->150, 150->10, so higher values = more detail
    inverted_threshold = 160 - threshold

    # Calculate kernel size based on inverted threshold (must be odd)
    # Lower detail (high inverted_threshold) = larger kernels
    kernel_size = max(3, min(15, 3 + (inverted_threshold // 20) * 2))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Median blur for noise reduction while preserving edges
    blurred = cv2.medianBlur(img, kernel_size)

    # Morphological operations for region simplification
    morph_kernel_size = max(3, min(9, 3 + (inverted_threshold // 30) * 2))
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


def find_region_label_position(mask: np.ndarray) -> tuple:
    """
    Find optimal position for placing a label in a region.
    Uses centroid with fallback to ensure position is within region.
    """
    # Find all points in the region
    points = np.column_stack(np.where(mask > 0))

    if len(points) == 0:
        return None

    # Calculate centroid
    centroid_y, centroid_x = points.mean(axis=0).astype(int)

    # Verify centroid is within region
    if mask[centroid_y, centroid_x]:
        return (centroid_x, centroid_y)

    # Fallback: use the first point in the region
    return (points[0, 1], points[0, 0])


def calculate_font_scale(region_area: int, base_scale: float = 0.6) -> tuple:
    """
    Calculate font scale and thickness based on region size.
    Returns (font_scale, thickness)
    """
    # Scale font with square root of area
    scale_factor = np.sqrt(region_area / 500)  # 500 is reference area
    font_scale = max(0.3, min(base_scale * scale_factor, 1.2))
    thickness = max(1, int(font_scale * 2))

    return (font_scale, thickness)


def generate_numbered_template(
    simplified: np.ndarray,
    palette_data: List[Dict],
    palette_rgb: np.ndarray
) -> np.ndarray:
    """
    Generate paint-by-number template with numbered regions and color legend.

    This creates a clean black-and-white template with:
    - Region boundaries in black
    - Region numbers placed optimally
    - Color legend showing number -> color mapping
    """
    height, width = simplified.shape[:2]

    # Create white canvas
    template = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Create a mapping from RGB colors to palette IDs
    color_to_id = {}
    for idx, color_info in enumerate(palette_data):
        rgb_tuple = tuple(color_info['rgb'])
        color_to_id[rgb_tuple] = idx + 1  # 1-indexed for user-friendliness

    # Create labels image for connected components
    # Convert RGB to single channel by creating unique ID for each color
    simplified_flat = simplified[:, :, 0] * 65536 + simplified[:, :, 1] * 256 + simplified[:, :, 2]

    # Find regions for each color
    processed_regions = set()
    region_info = []  # Store region info for legend

    for palette_idx, palette_color in enumerate(palette_rgb):
        # Create mask for this specific color
        color_mask = np.all(simplified == palette_color, axis=2).astype(np.uint8)

        if color_mask.sum() == 0:
            continue

        # Find connected components for this color
        num_labels, labels = cv2.connectedComponents(color_mask)

        for region_id in range(1, num_labels):  # Skip background (0)
            # Create mask for this specific region
            region_mask = (labels == region_id).astype(np.uint8)
            region_area = region_mask.sum()

            # Skip very small regions (less than 50 pixels)
            if region_area < 50:
                continue

            # Find contours for this region
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                continue

            # Draw black boundary
            cv2.drawContours(template, contours, -1, (0, 0, 0), thickness=2)

            # Find optimal label position
            label_pos = find_region_label_position(region_mask)

            if label_pos is None:
                continue

            # Get the palette ID for this color
            color_tuple = tuple(palette_color.tolist())
            palette_id = color_to_id.get(color_tuple, palette_idx + 1)

            # Calculate font scale based on region size
            font_scale, thickness = calculate_font_scale(region_area)

            # Prepare text
            text = str(palette_id)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Get text size to center it
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Adjust position to center text
            text_x = label_pos[0] - text_width // 2
            text_y = label_pos[1] + text_height // 2

            # Ensure text is within bounds
            text_x = max(5, min(text_x, width - text_width - 5))
            text_y = max(text_height + 5, min(text_y, height - 5))

            # Draw white background for better visibility
            bg_margin = 3
            cv2.rectangle(
                template,
                (text_x - bg_margin, text_y - text_height - bg_margin),
                (text_x + text_width + bg_margin, text_y + baseline + bg_margin),
                (255, 255, 255),
                -1
            )

            # Draw number
            cv2.putText(
                template,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

            # Store region info
            if palette_id not in [r['id'] for r in region_info]:
                region_info.append({
                    'id': palette_id,
                    'color': palette_color,
                    'used': True
                })

    # Add color legend to the template
    template_with_legend = add_color_legend(template, palette_data, palette_rgb)

    return template_with_legend


def add_color_legend(
    template: np.ndarray,
    palette_data: List[Dict],
    palette_rgb: np.ndarray
) -> np.ndarray:
    """
    Add a color legend to the bottom of the template.
    Shows number -> color mapping for easy reference.
    """
    height, width = template.shape[:2]

    # Calculate legend dimensions
    legend_height = min(200, max(100, len(palette_data) * 25))

    # Create expanded canvas
    canvas = np.ones((height + legend_height + 40, width, 3), dtype=np.uint8) * 255

    # Copy template to top portion
    canvas[:height, :] = template

    # Draw legend title
    title_y = height + 30
    cv2.putText(
        canvas,
        "Paint-by-Number Color Guide",
        (20, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # Calculate layout for legend items
    items_per_row = max(1, width // 180)  # Each item needs ~180px

    for idx, (color_info, color_rgb) in enumerate(zip(palette_data, palette_rgb)):
        row = idx // items_per_row
        col = idx % items_per_row

        x = 20 + col * 180
        y = title_y + 35 + row * 30

        # Ensure we don't overflow
        if y > height + legend_height + 20:
            break

        # Draw color swatch
        swatch_size = 20
        cv2.rectangle(
            canvas,
            (x, y - swatch_size),
            (x + swatch_size, y),
            tuple(int(c) for c in color_rgb.tolist()),
            -1
        )
        cv2.rectangle(
            canvas,
            (x, y - swatch_size),
            (x + swatch_size, y),
            (0, 0, 0),
            1
        )

        # Draw number and note
        text = f"{idx + 1}: {color_info.get('note', 'Color')} "
        cv2.putText(
            canvas,
            text,
            (x + swatch_size + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return canvas


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


# API Routes
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Paint by Number API",
        "version": "1.0.0"
    }


@app.post("/api/generate")
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

        # Step 0: Apply bilateral filter to preserve edges while smoothing
        # This helps create cleaner regions before color mapping
        img_smoothed = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)

        # Step 1: Map to custom palette using LAB color space
        quantized = map_to_custom_palette(img_smoothed, palette_rgb)

        # Step 2: Simplify regions based on threshold
        simplified = simplify_regions(quantized, threshold)

        # Step 3: Generate numbered template with regions and legend
        template = generate_numbered_template(simplified, palette_data, palette_rgb)
        
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


# Serve React Static Files
# Check if dist folder exists (production)
dist_path = Path(__file__).parent / "dist"
if dist_path.exists():
    # Mount static assets
    app.mount("/assets", StaticFiles(directory=str(dist_path / "assets")), name="assets")
    
    # Serve index.html for root and all non-API routes
    @app.get("/")
    async def serve_root():
        """Serve React app"""
        return FileResponse(str(dist_path / "index.html"))
    
    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        """Catch-all route to serve React app for client-side routing"""
        # Don't serve index.html for API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        
        # Check if file exists in dist
        file_path = dist_path / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        
        # Otherwise serve index.html for React Router
        return FileResponse(str(dist_path / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
