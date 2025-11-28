from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


def pole_of_inaccessibility(mask: np.ndarray, precision: float = 1.0) -> tuple:
    """
    Find the pole of inaccessibility - the point inside a region that is
    farthest from any edge. This is the optimal position for placing text.

    Based on the polylabel algorithm by Mapbox.
    """
    from scipy.ndimage import distance_transform_edt

    # Use distance transform to find point farthest from edges
    # This is faster than the full polylabel algorithm and works well for our use case
    distance_map = distance_transform_edt(mask)

    # Find the point with maximum distance
    max_dist_idx = np.unravel_index(np.argmax(distance_map), distance_map.shape)

    # Return as (x, y) coordinates
    return (max_dist_idx[1], max_dist_idx[0])


def calculate_adaptive_font_scale(region_area: int, text_length: int = 1) -> tuple:
    """
    Calculate font scale adaptively based on region size and text length.
    Returns (font_scale, thickness, should_place)
    """
    # Minimum area required for text placement
    min_area_per_char = 100  # pixels per character
    required_area = min_area_per_char * text_length

    if region_area < required_area:
        # Region too small for reliable text placement
        return (0, 0, False)

    # Calculate base font scale from area
    # Use cube root for more gradual scaling
    scale_factor = np.power(region_area / 500, 0.35)

    # Adjust for text length
    length_penalty = 1.0 / (1.0 + (text_length - 1) * 0.15)

    font_scale = max(0.25, min(scale_factor * length_penalty * 0.5, 0.9))
    thickness = max(1, int(font_scale * 2.5))

    return (font_scale, thickness, True)


def text_fits_in_region(mask: np.ndarray, pos: tuple, text: str,
                        font_scale: float, thickness: int) -> bool:
    """
    Check if text will fit inside the region without going outside or
    overlapping boundaries too much.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # Create a buffer zone around the text
    buffer = 4
    text_bbox_width = text_width + buffer * 2
    text_bbox_height = text_height + baseline + buffer * 2

    # Check if the bounding box fits within the region
    x, y = pos
    half_width = text_bbox_width // 2
    half_height = text_bbox_height // 2

    # Extract region around text position
    y_min = max(0, y - half_height)
    y_max = min(mask.shape[0], y + half_height)
    x_min = max(0, x - half_width)
    x_max = min(mask.shape[1], x + half_width)

    if y_max <= y_min or x_max <= x_min:
        return False

    region_crop = mask[y_min:y_max, x_min:x_max]

    # At least 80% of the text bounding box should be inside the region
    fill_ratio = region_crop.sum() / (region_crop.size + 1e-6)

    return fill_ratio > 0.8


def draw_outlined_text(img: np.ndarray, text: str, pos: tuple,
                      font_scale: float, thickness: int,
                      text_color=(0, 0, 0), outline_color=(255, 255, 255)):
    """
    Draw text with an outline for better visibility without background boxes.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw outline (thicker, white)
    outline_thickness = thickness + 2
    cv2.putText(img, text, pos, font, font_scale, outline_color,
                outline_thickness, cv2.LINE_AA)

    # Draw main text (thinner, black)
    cv2.putText(img, text, pos, font, font_scale, text_color,
                thickness, cv2.LINE_AA)


def should_add_multiple_numbers(region_area: int, region_shape: tuple) -> int:
    """
    Determine how many numbers should be placed in a region based on its size.
    Returns the number of labels to place (1-3).
    """
    # Very small regions: 1 number
    if region_area < 1000:
        return 1

    # Medium regions: 1 number
    elif region_area < 3000:
        return 1

    # Large regions: potentially 2-3 numbers for easier painting
    elif region_area < 6000:
        return 2

    else:
        return min(3, max(2, region_area // 3000))


def find_multiple_label_positions(mask: np.ndarray, num_labels: int) -> List[tuple]:
    """
    Find multiple optimal positions for placing labels in large regions.
    Uses k-means clustering on the distance transform.
    """
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial.distance import cdist

    if num_labels == 1:
        return [pole_of_inaccessibility(mask)]

    # Get distance transform
    distance_map = distance_transform_edt(mask)

    # Find all points with good distance from edges
    threshold = max(3, distance_map.max() * 0.3)
    good_points = np.column_stack(np.where(distance_map > threshold))

    if len(good_points) < num_labels:
        # Not enough good points, just return the best one
        return [pole_of_inaccessibility(mask)]

    # Sample points to reduce computation
    if len(good_points) > 500:
        indices = np.random.choice(len(good_points), 500, replace=False)
        good_points = good_points[indices]

    # Simple k-means clustering to spread labels out
    # Initialize centers
    centers_idx = np.linspace(0, len(good_points) - 1, num_labels, dtype=int)
    centers = good_points[centers_idx]

    # Run a few iterations of k-means
    for _ in range(5):
        # Assign points to nearest center
        distances = cdist(good_points, centers)
        assignments = np.argmin(distances, axis=1)

        # Update centers
        new_centers = []
        for i in range(num_labels):
            cluster_points = good_points[assignments == i]
            if len(cluster_points) > 0:
                # Weight by distance transform value
                cluster_dist_values = distance_map[
                    cluster_points[:, 0], cluster_points[:, 1]
                ]
                weights = cluster_dist_values / (cluster_dist_values.sum() + 1e-6)
                weighted_center = np.average(cluster_points, axis=0, weights=weights)
                new_centers.append(weighted_center.astype(int))
            else:
                new_centers.append(centers[i])

        centers = np.array(new_centers)

    # Convert to (x, y) format
    positions = [(int(center[1]), int(center[0])) for center in centers]

    return positions


def generate_numbered_template(
    simplified: np.ndarray,
    palette_data: List[Dict],
    palette_rgb: np.ndarray
) -> np.ndarray:
    """
    Generate professional paint-by-number template with advanced algorithms.

    Features:
    - Pole of inaccessibility for optimal number placement
    - Adaptive numbering (1-3 numbers per region based on size)
    - Smart font sizing that guarantees fit
    - Outlined text instead of background boxes
    - Thin 1px boundaries
    - Proper color legend
    """
    height, width = simplified.shape[:2]

    # Create white canvas
    template = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Create a mapping from RGB colors to palette IDs
    color_to_id = {}
    for idx, color_info in enumerate(palette_data):
        rgb_tuple = tuple(color_info['rgb'])
        color_to_id[rgb_tuple] = idx + 1  # 1-indexed for user-friendliness

    # Track which colors are actually used
    colors_used = set()

    # Process each color in the palette
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

            # Skip very small regions
            if region_area < 80:
                continue

            # Find contours for this region
            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                continue

            # Draw thin boundary (1px)
            cv2.drawContours(template, contours, -1, (0, 0, 0), thickness=1)

            # Get the palette ID for this color
            color_tuple = tuple(palette_color.tolist())
            palette_id = color_to_id.get(color_tuple, palette_idx + 1)
            colors_used.add(palette_id)

            # Prepare text
            text = str(palette_id)

            # Determine how many numbers to place
            num_numbers = should_add_multiple_numbers(region_area, region_mask.shape)

            # Calculate adaptive font scale
            font_scale, thickness, should_place = calculate_adaptive_font_scale(
                region_area, len(text)
            )

            if not should_place:
                continue

            # Find optimal position(s)
            positions = find_multiple_label_positions(region_mask, num_numbers)

            # Place numbers
            for pos in positions:
                # Verify text fits
                if not text_fits_in_region(region_mask, pos, text, font_scale, thickness):
                    # Try reducing font size
                    font_scale *= 0.7
                    thickness = max(1, thickness - 1)

                    if not text_fits_in_region(region_mask, pos, text, font_scale, thickness):
                        continue  # Skip this position

                # Calculate text position (adjust for baseline)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, thickness
                )

                text_x = pos[0] - text_width // 2
                text_y = pos[1] + text_height // 2

                # Draw outlined text
                draw_outlined_text(
                    template, text, (text_x, text_y),
                    font_scale, thickness
                )

    # Add color legend with only used colors
    template_with_legend = add_color_legend(
        template, palette_data, palette_rgb, colors_used
    )

    return template_with_legend


def add_color_legend(
    template: np.ndarray,
    palette_data: List[Dict],
    palette_rgb: np.ndarray,
    colors_used: set = None
) -> np.ndarray:
    """
    Add a color legend to the bottom of the template.
    Shows number -> color mapping with CORRECT colors from palette.
    Only shows colors that are actually used in the image.
    """
    height, width = template.shape[:2]

    # Filter to only used colors if provided
    if colors_used:
        display_items = [(idx + 1, palette_data[idx], palette_rgb[idx])
                        for idx in range(len(palette_data))
                        if (idx + 1) in colors_used]
    else:
        display_items = [(idx + 1, palette_data[idx], palette_rgb[idx])
                        for idx in range(len(palette_data))]

    if len(display_items) == 0:
        return template

    # Calculate legend dimensions
    legend_height = min(250, max(120, len(display_items) * 25))

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
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # Calculate layout for legend items
    items_per_row = max(1, width // 200)  # Each item needs ~200px

    for display_idx, (palette_id, color_info, color_rgb) in enumerate(display_items):
        row = display_idx // items_per_row
        col = display_idx % items_per_row

        x = 20 + col * 200
        y = title_y + 40 + row * 32

        # Ensure we don't overflow
        if y > height + legend_height + 20:
            break

        # Draw color swatch (use exact RGB from palette)
        swatch_size = 24
        # Convert palette RGB to BGR for OpenCV
        bgr_color = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

        cv2.rectangle(
            canvas,
            (x, y - swatch_size),
            (x + swatch_size, y),
            bgr_color,
            -1
        )
        # Black border around swatch
        cv2.rectangle(
            canvas,
            (x, y - swatch_size),
            (x + swatch_size, y),
            (0, 0, 0),
            2
        )

        # Draw number and note
        note_text = color_info.get('note', 'Color').strip()
        if not note_text:
            note_text = 'Color'

        text = f"{palette_id}: {note_text[:20]}"  # Truncate long notes
        cv2.putText(
            canvas,
            text,
            (x + swatch_size + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
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


@app.get("/")
async def root():
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

from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Serve static files (React app)
static_dir = Path(__file__).parent / "dist"
if static_dir.exists():
    # Serve static assets
    app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")
    
    # Catch-all route for React Router (must be AFTER all API routes)
    from fastapi.responses import FileResponse
    
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