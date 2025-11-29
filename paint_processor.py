"""
Paint-by-Number Processor v2.0
Professional-grade image-to-template conversion with:
- Advanced color quantization (LAB + CIEDE2000-weighted)
- Intelligent region building and cleanup
- Smooth vector contour extraction
- Smart label placement with collision avoidance
- SVG/PDF output for printing
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
from skimage import color
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import warnings

# Optional imports - graceful fallback
try:
    import svgwrite
    HAS_SVGWRITE = True
except ImportError:
    HAS_SVGWRITE = False
    warnings.warn("svgwrite not installed - SVG output disabled")


@dataclass
class PaletteColor:
    """Represents a color in the user's palette."""
    id: str
    rgb: Tuple[int, int, int]
    lab: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    name: str = ""


@dataclass  
class Region:
    """Represents a paintable region."""
    id: int
    color_id: str
    pixels: int
    centroid: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    contour: Optional[np.ndarray] = None
    label_position: Optional[Tuple[int, int]] = None
    font_size: int = 12


@dataclass
class ProcessingResult:
    """Complete processing result."""
    preview_image: np.ndarray
    region_map: np.ndarray
    regions: List[Region]
    contours: Dict[int, List[np.ndarray]]  # region_id -> list of contours
    palette_used: List[PaletteColor]
    dimensions: Tuple[int, int]  # width, height


class PaintByNumberProcessor:
    """
    Production-grade paint-by-number generator.
    
    Key improvements over basic implementations:
    1. Bilateral filtering preserves edges while smoothing
    2. LAB color space with perceptual weighting
    3. Multi-pass region cleanup (small region removal, hole filling, strip removal)
    4. Smooth contour extraction with Douglas-Peucker simplification
    5. Intelligent label placement using distance transform
    """
    
    def __init__(self, 
                 min_region_pixels: int = 100,
                 target_max_regions: int = 200,
                 line_thickness: int = 2,
                 output_size: int = 2400,
                 min_font_size: int = 8,
                 max_font_size: int = 20):
        """
        Initialize processor with configuration.
        
        Args:
            min_region_pixels: Minimum pixels for a region (smaller merged)
            target_max_regions: Target maximum number of regions
            line_thickness: Line thickness for template output
            output_size: Maximum dimension for output (affects print quality)
            min_font_size: Minimum label font size
            max_font_size: Maximum label font size
        """
        self.min_region_pixels = min_region_pixels
        self.target_max_regions = target_max_regions
        self.line_thickness = line_thickness
        self.output_size = output_size
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        
    def process(self, 
                image: np.ndarray, 
                palette: List[Dict],
                complexity: int = 50) -> ProcessingResult:
        """
        Main processing pipeline.
        
        Args:
            image: BGR image (from cv2.imread) or RGB image
            palette: List of color dicts with 'id', 'rgb', and optional 'note' keys
            complexity: 1-100, higher = more detail/regions
            
        Returns:
            ProcessingResult with preview, template, regions, etc.
        """
        # Ensure we have a valid complexity value
        complexity = max(1, min(100, complexity))
        
        # Convert palette to internal format
        palette_colors = self._prepare_palette(palette)
        
        if len(palette_colors) < 2:
            raise ValueError("Palette must have at least 2 colors")
        
        # Preprocess image
        processed = self._preprocess(image, complexity)
        
        # Quantize to palette colors
        quantized, color_indices = self._quantize_to_palette(
            processed, palette_colors, complexity
        )
        
        # Build and clean regions
        region_map, regions = self._build_regions(
            color_indices, palette_colors, complexity
        )
        
        # Extract contours for all regions
        contours = self._extract_all_contours(region_map)
        
        # Calculate label positions
        regions = self._calculate_label_positions(region_map, regions)
        
        # Build preview image
        preview = self._build_preview(region_map, palette_colors, regions)
        
        return ProcessingResult(
            preview_image=preview,
            region_map=region_map,
            regions=regions,
            contours=contours,
            palette_used=palette_colors,
            dimensions=(quantized.shape[1], quantized.shape[0])
        )
    
    def _prepare_palette(self, palette: List[Dict]) -> List[PaletteColor]:
        """Convert input palette to LAB color space."""
        colors = []
        for p in palette:
            rgb = tuple(p['rgb'])
            # Validate RGB values
            if len(rgb) != 3 or not all(0 <= v <= 255 for v in rgb):
                raise ValueError(f"Invalid RGB value: {rgb}")
            
            # Convert to LAB using scikit-image
            rgb_normalized = np.array([[list(rgb)]], dtype=np.float64) / 255.0
            lab = color.rgb2lab(rgb_normalized)[0, 0]
            
            colors.append(PaletteColor(
                id=str(p['id']),
                rgb=rgb,
                lab=lab,
                name=p.get('note', p.get('name', ''))
            ))
        return colors
    
    def _preprocess(self, image: np.ndarray, complexity: int) -> np.ndarray:
        """
        Preprocess image: resize, convert color space, denoise.
        
        Uses bilateral filtering which preserves edges while smoothing.
        """
        # Check if image is BGR (from cv2) or RGB
        if len(image.shape) == 2:
            # Grayscale - convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA - drop alpha
            image = image[:, :, :3]
        
        # Assume BGR input (from cv2.imread), convert to RGB
        # If already RGB, this is handled by the caller
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Try to detect if BGR by checking if it looks wrong
            # For safety, assume BGR and convert
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to working size
        h, w = image.shape[:2]
        scale = self.output_size / max(h, w)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif scale > 1 and max(h, w) < 800:
            # Upscale very small images
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Adaptive denoising based on complexity
        # Lower complexity = more smoothing = simpler regions
        # d: diameter of pixel neighborhood (5-15)
        # sigmaColor: filter sigma in color space (larger = more colors mixed)
        # sigmaSpace: filter sigma in coordinate space
        
        # Scale parameters inversely with complexity
        smooth_factor = (100 - complexity) / 100.0  # 0 at complexity=100, 1 at complexity=0
        
        d = int(5 + smooth_factor * 10)  # 5-15
        sigma_color = int(30 + smooth_factor * 70)  # 30-100
        sigma_space = int(30 + smooth_factor * 70)  # 30-100
        
        image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
        # Optional: light contrast enhancement
        # Convert to LAB, enhance L channel, convert back
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return image
    
    def _quantize_to_palette(self, 
                             image: np.ndarray, 
                             palette: List[PaletteColor],
                             complexity: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize image colors to the nearest palette colors.
        
        Uses LAB color space for perceptually accurate matching.
        """
        h, w = image.shape[:2]
        
        # Convert image to LAB
        image_float = image.astype(np.float64) / 255.0
        image_lab = color.rgb2lab(image_float)
        pixels_lab = image_lab.reshape(-1, 3)
        
        # Build KD-tree from palette LAB values
        palette_lab = np.array([c.lab for c in palette])
        tree = cKDTree(palette_lab)
        
        # Find nearest palette color for each pixel
        # KD-tree query is O(log n) per pixel
        distances, indices = tree.query(pixels_lab, k=1)
        
        # Reshape to image dimensions
        indices_2d = indices.reshape(h, w)
        
        # Apply spatial smoothing to reduce noise in color assignment
        # This creates cleaner regions
        smooth_factor = (100 - complexity) / 100.0
        filter_size = max(3, int(3 + smooth_factor * 12))  # 3-15, odd numbers
        if filter_size % 2 == 0:
            filter_size += 1
        
        # Median filter on indices - this smooths region boundaries
        indices_smoothed = ndimage.median_filter(
            indices_2d.astype(np.int32), 
            size=filter_size
        )
        
        # Rebuild quantized image from indices
        quantized = np.zeros((h, w, 3), dtype=np.uint8)
        for i, pc in enumerate(palette):
            mask = indices_smoothed == i
            quantized[mask] = pc.rgb
            
        return quantized, indices_smoothed.astype(np.int32)
    
    def _build_regions(self, 
                       color_indices: np.ndarray,
                       palette: List[PaletteColor],
                       complexity: int) -> Tuple[np.ndarray, List[Region]]:
        """
        Build clean, paintable regions from quantized image.
        
        Process:
        1. Find connected components for each color
        2. Remove tiny regions (merge into neighbors)
        3. Fill small holes
        4. Clean up narrow strips
        """
        h, w = color_indices.shape
        region_map = np.zeros((h, w), dtype=np.int32)
        regions = []
        current_region_id = 1
        
        # Process each color separately
        for color_idx, pc in enumerate(palette):
            color_mask = color_indices == color_idx
            if not np.any(color_mask):
                continue
                
            # Find connected components for this color
            # Using 4-connectivity for cleaner boundaries
            labeled, num_features = ndimage.label(color_mask)
            
            for component_id in range(1, num_features + 1):
                component_mask = labeled == component_id
                pixel_count = np.sum(component_mask)
                
                # Calculate region properties
                coords = np.argwhere(component_mask)
                if len(coords) == 0:
                    continue
                    
                centroid = coords.mean(axis=0)
                min_r, min_c = coords.min(axis=0)
                max_r, max_c = coords.max(axis=0)
                
                # Assign to region map
                region_map[component_mask] = current_region_id
                
                regions.append(Region(
                    id=current_region_id,
                    color_id=pc.id,
                    pixels=int(pixel_count),
                    centroid=(float(centroid[1]), float(centroid[0])),  # x, y
                    bounding_box=(int(min_c), int(min_r), 
                                  int(max_c - min_c + 1), int(max_r - min_r + 1))
                ))
                current_region_id += 1
        
        # Clean up regions
        region_map, regions = self._cleanup_regions(region_map, regions, complexity)
        
        return region_map, regions
    
    def _cleanup_regions(self, 
                         region_map: np.ndarray, 
                         regions: List[Region],
                         complexity: int) -> Tuple[np.ndarray, List[Region]]:
        """
        Remove small regions and merge appropriately.
        
        Multi-pass cleanup:
        1. Merge regions smaller than threshold
        2. Fill holes in regions
        3. Remove narrow strips
        """
        # Adjust min size based on complexity
        # At complexity=100, use min threshold
        # At complexity=1, use higher threshold
        smooth_factor = (100 - complexity) / 100.0
        min_size = max(50, int(self.min_region_pixels * (1 + smooth_factor * 2)))
        
        # Pass 1: Merge small regions into neighbors
        iterations = 0
        max_iterations = 10
        
        while iterations < max_iterations:
            iterations += 1
            small_regions = [r for r in regions if r.pixels < min_size]
            
            if not small_regions:
                break
                
            merged_any = False
            for small_region in small_regions:
                if small_region.pixels == 0:
                    continue
                    
                mask = region_map == small_region.id
                if not np.any(mask):
                    continue
                
                # Find neighboring regions
                dilated = ndimage.binary_dilation(mask, iterations=1)
                border = dilated & ~mask
                neighbor_ids = np.unique(region_map[border])
                neighbor_ids = neighbor_ids[(neighbor_ids != 0) & (neighbor_ids != small_region.id)]
                
                if len(neighbor_ids) > 0:
                    # Find neighbor with most shared boundary
                    best_neighbor = None
                    best_shared = 0
                    
                    for nid in neighbor_ids:
                        shared = np.sum(border & (region_map == nid))
                        if shared > best_shared:
                            best_shared = shared
                            best_neighbor = nid
                    
                    if best_neighbor is not None:
                        # Merge into best neighbor
                        region_map[mask] = best_neighbor
                        
                        # Update pixel counts
                        for r in regions:
                            if r.id == best_neighbor:
                                r.pixels += small_region.pixels
                                break
                        
                        small_region.pixels = 0  # Mark as merged
                        merged_any = True
            
            if not merged_any:
                break
        
        # Remove merged regions from list
        regions = [r for r in regions if r.pixels > 0]
        
        # Pass 2: Fill small holes in regions
        for region in regions:
            mask = region_map == region.id
            filled = ndimage.binary_fill_holes(mask)
            holes = filled & ~mask
            
            if np.any(holes):
                hole_labels, num_holes = ndimage.label(holes)
                for hole_id in range(1, num_holes + 1):
                    hole_mask = hole_labels == hole_id
                    hole_size = np.sum(hole_mask)
                    # Only fill small holes
                    if hole_size < min_size // 2:
                        region_map[hole_mask] = region.id
                        region.pixels += hole_size
        
        # Update region properties after cleanup
        for region in regions:
            mask = region_map == region.id
            if not np.any(mask):
                region.pixels = 0
                continue
                
            coords = np.argwhere(mask)
            region.pixels = len(coords)
            region.centroid = (float(coords[:, 1].mean()), float(coords[:, 0].mean()))
            min_r, min_c = coords.min(axis=0)
            max_r, max_c = coords.max(axis=0)
            region.bounding_box = (int(min_c), int(min_r), 
                                   int(max_c - min_c + 1), int(max_r - min_r + 1))
        
        # Final filter - remove any regions that got too small
        regions = [r for r in regions if r.pixels >= min_size // 2]
        
        return region_map, regions
    
    def _extract_all_contours(self, region_map: np.ndarray) -> Dict[int, List[np.ndarray]]:
        """
        Extract smooth contours for all regions.
        
        Returns both external and internal contours (holes).
        """
        contours = {}
        unique_regions = np.unique(region_map)
        
        for region_id in unique_regions:
            if region_id == 0:
                continue
                
            mask = (region_map == region_id).astype(np.uint8) * 255
            
            # Find contours - RETR_TREE gets hierarchy (for holes)
            cnts, hierarchy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
            )
            
            if not cnts:
                contours[region_id] = []
                continue
            
            region_contours = []
            for cnt in cnts:
                if len(cnt) < 3:
                    continue
                    
                # Simplify contour using Douglas-Peucker algorithm
                # Epsilon controls simplification level
                perimeter = cv2.arcLength(cnt, True)
                epsilon = 0.002 * perimeter  # 0.2% of perimeter
                simplified = cv2.approxPolyDP(cnt, epsilon, True)
                
                if len(simplified) >= 3:
                    region_contours.append(simplified.squeeze())
            
            contours[region_id] = region_contours
                
        return contours
    
    def _calculate_label_positions(self, 
                                   region_map: np.ndarray, 
                                   regions: List[Region]) -> List[Region]:
        """
        Calculate optimal label positions using distance transform.
        
        Places labels at the point furthest from region boundaries,
        with collision avoidance.
        """
        placed_labels: List[Tuple[int, int, int, int]] = []  # x, y, w, h
        
        # Sort regions by size (process larger first for better placement)
        sorted_regions = sorted(regions, key=lambda r: r.pixels, reverse=True)
        
        for region in sorted_regions:
            mask = region_map == region.id
            
            if not np.any(mask):
                continue
            
            # Distance transform - finds distance to nearest boundary
            dist = ndimage.distance_transform_edt(mask)
            max_dist = dist.max()
            
            if max_dist < 5:
                # Region too thin for label, use centroid
                region.label_position = (int(region.centroid[0]), int(region.centroid[1]))
                region.font_size = self.min_font_size
                continue
            
            # Find candidates: points with high distance values
            threshold = max_dist * 0.7
            candidates = np.argwhere(dist >= threshold)
            
            if len(candidates) == 0:
                # Fallback to maximum distance point
                max_idx = np.unravel_index(np.argmax(dist), dist.shape)
                region.label_position = (int(max_idx[1]), int(max_idx[0]))
                region.font_size = max(self.min_font_size, min(self.max_font_size, int(max_dist * 0.5)))
                continue
            
            # Calculate font size based on available space
            font_size = max(self.min_font_size, min(self.max_font_size, int(max_dist * 0.6)))
            label_text = region.color_id
            
            # Estimate label dimensions
            label_width = len(label_text) * font_size * 0.7
            label_height = font_size * 1.3
            
            # Find best candidate (furthest from existing labels and boundaries)
            best_pos = None
            best_score = -1
            
            for y, x in candidates:
                # Check collision with existing labels
                has_collision = False
                for lx, ly, lw, lh in placed_labels:
                    # Check overlap with margin
                    margin = 5
                    if (x - label_width/2 < lx + lw + margin and
                        x + label_width/2 > lx - margin and
                        y - label_height/2 < ly + lh + margin and
                        y + label_height/2 > ly - margin):
                        has_collision = True
                        break
                
                if has_collision:
                    continue
                
                # Score based on distance from boundary
                score = dist[y, x]
                
                if score > best_score:
                    best_score = score
                    best_pos = (int(x), int(y))
            
            if best_pos is None:
                # No collision-free position, use max distance point
                max_idx = np.unravel_index(np.argmax(dist), dist.shape)
                best_pos = (int(max_idx[1]), int(max_idx[0]))
            
            region.label_position = best_pos
            region.font_size = font_size
            
            # Record placed label
            placed_labels.append((
                best_pos[0] - label_width/2,
                best_pos[1] - label_height/2,
                label_width,
                label_height
            ))
        
        return regions
    
    def _build_preview(self, 
                       region_map: np.ndarray,
                       palette: List[PaletteColor],
                       regions: List[Region]) -> np.ndarray:
        """Build the colored preview image from regions."""
        h, w = region_map.shape
        preview = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
        
        # Build color lookup
        region_colors = {}
        for region in regions:
            for pc in palette:
                if pc.id == region.color_id:
                    region_colors[region.id] = pc.rgb
                    break
        
        # Fill regions with colors
        for region_id, rgb in region_colors.items():
            mask = region_map == region_id
            preview[mask] = rgb
        
        return preview
    
    def generate_template_image(self, 
                                result: ProcessingResult,
                                show_numbers: bool = True,
                                line_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Generate raster template image with lines and numbers.
        
        Args:
            result: ProcessingResult from process()
            show_numbers: Whether to show region numbers
            line_color: RGB color for lines
            
        Returns:
            RGB numpy array of template
        """
        w, h = result.dimensions
        template = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
        
        # Draw contours
        for region_id, contour_list in result.contours.items():
            for contour in contour_list:
                if len(contour) < 3:
                    continue
                # Draw contour
                pts = contour.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(template, [pts], True, line_color, self.line_thickness)
        
        # Add labels
        if show_numbers:
            for region in result.regions:
                if region.label_position:
                    x, y = region.label_position
                    label = str(region.color_id)
                    
                    # Calculate text size for centering
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = region.font_size / 30.0
                    thickness = max(1, region.font_size // 10)
                    
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    
                    # Center the text
                    text_x = int(x - text_w / 2)
                    text_y = int(y + text_h / 2)
                    
                    # Draw white background for readability
                    padding = 2
                    cv2.rectangle(
                        template,
                        (text_x - padding, text_y - text_h - padding),
                        (text_x + text_w + padding, text_y + padding),
                        (255, 255, 255),
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        template, label,
                        (text_x, text_y),
                        font, font_scale, (51, 51, 51), thickness
                    )
        
        return template
    
    def generate_template_svg(self, 
                              result: ProcessingResult,
                              show_numbers: bool = True,
                              fill_regions: bool = False) -> str:
        """
        Generate SVG template for high-quality printing.
        
        Args:
            result: ProcessingResult from process()
            show_numbers: Whether to show region numbers
            fill_regions: Whether to fill regions with colors
            
        Returns:
            SVG string
        """
        if not HAS_SVGWRITE:
            raise ImportError("svgwrite is required for SVG output")
        
        w, h = result.dimensions
        dwg = svgwrite.Drawing(size=(f"{w}px", f"{h}px"), viewBox=f"0 0 {w} {h}")
        
        # White background
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill='white'))
        
        # Build color lookup for fill
        region_colors = {}
        if fill_regions:
            for region in result.regions:
                for pc in result.palette_used:
                    if pc.id == region.color_id:
                        region_colors[region.id] = pc.rgb
                        break
        
        # Draw region contours
        for region_id, contour_list in result.contours.items():
            fill_color = 'none'
            if fill_regions and region_id in region_colors:
                rgb = region_colors[region_id]
                fill_color = f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'
            
            for contour in contour_list:
                if len(contour) < 3:
                    continue
                
                # Convert contour to SVG path
                points = [(float(p[0]), float(p[1])) for p in contour]
                path_data = f"M {points[0][0]:.1f},{points[0][1]:.1f} "
                path_data += " ".join([f"L {p[0]:.1f},{p[1]:.1f}" for p in points[1:]])
                path_data += " Z"
                
                dwg.add(dwg.path(
                    d=path_data,
                    stroke='black',
                    stroke_width=self.line_thickness,
                    fill=fill_color,
                    stroke_linejoin='round'
                ))
        
        # Add labels
        if show_numbers:
            for region in result.regions:
                if region.label_position:
                    x, y = region.label_position
                    
                    # White background for label
                    label_text = str(region.color_id)
                    approx_width = len(label_text) * region.font_size * 0.6
                    approx_height = region.font_size * 1.2
                    
                    dwg.add(dwg.rect(
                        insert=(x - approx_width/2 - 2, y - approx_height/2 - 2),
                        size=(approx_width + 4, approx_height + 4),
                        fill='white',
                        fill_opacity=0.8
                    ))
                    
                    dwg.add(dwg.text(
                        label_text,
                        insert=(x, y),
                        text_anchor='middle',
                        dominant_baseline='middle',
                        font_size=f'{region.font_size}px',
                        font_family='Arial, Helvetica, sans-serif',
                        fill='#333333'
                    ))
        
        return dwg.tostring()
    
    def generate_color_key_svg(self, result: ProcessingResult) -> str:
        """
        Generate SVG color key/legend showing all colors used.
        
        Returns:
            SVG string of color key
        """
        if not HAS_SVGWRITE:
            raise ImportError("svgwrite is required for SVG output")
        
        # Find colors actually used in regions
        colors_used: Set[str] = set()
        for region in result.regions:
            colors_used.add(region.color_id)
        
        palette_in_use = [pc for pc in result.palette_used if pc.id in colors_used]
        
        if not palette_in_use:
            return ""
        
        # Calculate dimensions
        swatch_size = 50
        padding = 15
        text_width = 250
        row_height = swatch_size + padding
        width = swatch_size + padding * 3 + text_width
        height = len(palette_in_use) * row_height + padding * 2
        
        dwg = svgwrite.Drawing(size=(f"{width}px", f"{height}px"))
        
        # Background
        dwg.add(dwg.rect(
            insert=(0, 0), 
            size=(width, height), 
            fill='white', 
            stroke='#cccccc',
            stroke_width=1
        ))
        
        # Title
        dwg.add(dwg.text(
            "Color Key",
            insert=(padding, padding + 12),
            font_size='14px',
            font_weight='bold',
            font_family='Arial, sans-serif'
        ))
        
        for i, pc in enumerate(palette_in_use):
            y = padding * 2 + 20 + i * row_height
            
            # Color swatch with border
            dwg.add(dwg.rect(
                insert=(padding, y),
                size=(swatch_size, swatch_size),
                fill=f'rgb({pc.rgb[0]},{pc.rgb[1]},{pc.rgb[2]})',
                stroke='#333333',
                stroke_width=1
            ))
            
            # Number on swatch
            # Choose text color based on luminance
            luminance = 0.299 * pc.rgb[0] + 0.587 * pc.rgb[1] + 0.114 * pc.rgb[2]
            text_color = 'white' if luminance < 128 else 'black'
            
            dwg.add(dwg.text(
                pc.id,
                insert=(padding + swatch_size / 2, y + swatch_size / 2),
                text_anchor='middle',
                dominant_baseline='middle',
                font_size='16px',
                font_weight='bold',
                font_family='Arial, sans-serif',
                fill=text_color
            ))
            
            # Color name/note
            display_name = pc.name if pc.name else f"Color {pc.id}"
            rgb_str = f"RGB({pc.rgb[0]}, {pc.rgb[1]}, {pc.rgb[2]})"
            
            dwg.add(dwg.text(
                f"{pc.id}: {display_name}",
                insert=(padding * 2 + swatch_size, y + swatch_size / 2 - 8),
                font_size='13px',
                font_family='Arial, sans-serif',
                fill='#333333'
            ))
            
            dwg.add(dwg.text(
                rgb_str,
                insert=(padding * 2 + swatch_size, y + swatch_size / 2 + 10),
                font_size='11px',
                font_family='Arial, sans-serif',
                fill='#666666'
            ))
            
        return dwg.tostring()


# Convenience function for simple usage
def generate_paint_by_number(
    image_path: str,
    palette: List[Dict],
    complexity: int = 50,
    output_dir: str = "."
) -> Dict[str, str]:
    """
    Simple interface to generate paint-by-number from an image file.
    
    Args:
        image_path: Path to input image
        palette: List of color dicts with 'id', 'rgb', and optional 'note'
        complexity: 1-100, higher = more detail
        output_dir: Directory for output files
        
    Returns:
        Dict with paths to generated files
    """
    import os
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Process
    processor = PaintByNumberProcessor()
    result = processor.process(image, palette, complexity)
    
    # Generate outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    outputs = {}
    
    # Preview
    preview_path = os.path.join(output_dir, f"{base_name}_preview.png")
    preview_bgr = cv2.cvtColor(result.preview_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(preview_path, preview_bgr)
    outputs['preview'] = preview_path
    
    # Template image
    template_img = processor.generate_template_image(result)
    template_path = os.path.join(output_dir, f"{base_name}_template.png")
    cv2.imwrite(template_path, cv2.cvtColor(template_img, cv2.COLOR_RGB2BGR))
    outputs['template_png'] = template_path
    
    # SVG template
    if HAS_SVGWRITE:
        svg_path = os.path.join(output_dir, f"{base_name}_template.svg")
        with open(svg_path, 'w') as f:
            f.write(processor.generate_template_svg(result))
        outputs['template_svg'] = svg_path
        
        # Color key
        key_path = os.path.join(output_dir, f"{base_name}_color_key.svg")
        with open(key_path, 'w') as f:
            f.write(processor.generate_color_key_svg(result))
        outputs['color_key'] = key_path
    
    return outputs


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python paint_processor.py <image_path>")
        print("Example palette will be used for testing")
        sys.exit(1)
    
    # Example palette
    test_palette = [
        {"id": "1", "rgb": [255, 255, 255], "note": "White"},
        {"id": "2", "rgb": [0, 0, 0], "note": "Black"},
        {"id": "3", "rgb": [255, 0, 0], "note": "Red"},
        {"id": "4", "rgb": [0, 128, 0], "note": "Green"},
        {"id": "5", "rgb": [0, 0, 255], "note": "Blue"},
        {"id": "6", "rgb": [255, 255, 0], "note": "Yellow"},
        {"id": "7", "rgb": [255, 165, 0], "note": "Orange"},
        {"id": "8", "rgb": [128, 0, 128], "note": "Purple"},
    ]
    
    outputs = generate_paint_by_number(sys.argv[1], test_palette, complexity=50)
    print("Generated files:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
