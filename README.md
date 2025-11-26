# Paint by Number API (Backend)

FastAPI-based backend for processing images into paint-by-number templates using custom color palettes.

## Features

- **Custom Palette Matching**: Uses LAB color space and KD-tree for perceptually accurate color matching
- **Smart Region Simplification**: Adaptive median blur and morphological operations
- **Auto-Edge Detection**: Zero-parameter Canny edge detection
- **Clean Line Art Generation**: Production-ready templates for printing
- **Mobile-Optimized**: Supports Capacitor CORS origins and compressed responses

## Tech Stack

- **FastAPI**: Modern async web framework
- **OpenCV**: Image processing and edge detection
- **scikit-image**: LAB color space conversion
- **scipy**: KD-tree for fast nearest-neighbor search
- **Pillow**: Advanced image manipulation

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy environment configuration:
```bash
cp .env.example .env
```

## Running the Server

### Development Mode

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Production Mode

```bash
ENVIRONMENT=production uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc

## API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "online",
  "service": "Paint by Number API",
  "version": "1.0.0"
}
```

### `POST /generate`
Generate paint-by-number preview and template from an uploaded image.

**Parameters:**
- `file` (multipart/form-data): Image file (PNG, JPEG, WEBP)
- `palette` (form field): JSON string of custom colors
- `threshold` (form field): Region simplification level (10-150)

**Palette Format:**
```json
[
  {"id": "1", "rgb": [255, 0, 0], "note": "Cadmium Red"},
  {"id": "2", "rgb": [0, 0, 255], "note": "Ultramarine Blue"}
]
```

**Response:**
```json
{
  "success": true,
  "preview": "data:image/jpeg;base64,...",
  "template": "data:image/jpeg;base64,...",
  "dimensions": {
    "width": 800,
    "height": 600
  }
}
```

## Algorithm Details

### Color Matching Pipeline

1. **Image Preprocessing**: Resize to max 800px dimension
2. **LAB Conversion**: Convert both image and palette to LAB color space
3. **KD-Tree Search**: O(log n) nearest-neighbor lookup for each pixel
4. **Region Simplification**: Median blur + morphological operations based on threshold
5. **Edge Detection**: Auto-Canny with adaptive thresholding
6. **Line Art Generation**: Morphological refinement for clean templates

### Why LAB Color Space?

RGB Euclidean distance doesn't match human perception. LAB provides perceptually uniform color differences where equal distances represent equal perceived color changes.

### Performance Optimization

- KD-tree provides O(log n) color lookup vs O(n) naive iteration
- Median blur preserves edges better than Gaussian while removing speckles
- Auto-Canny eliminates manual threshold tuning across different images

## Configuration

### CORS Settings

For development, all origins are allowed. For production, configure specific origins in `main.py`:

```python
origins = [
    "https://yourdomain.com",
    "capacitor://localhost",  # iOS
    "http://localhost",        # Android
]
```

### Image Size Limits

- Max file size: 10MB
- Processing resize: 800px max dimension
- Supported formats: PNG, JPEG, WEBP

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid palette, threshold out of range)
- `413`: File too large
- `415`: Unsupported file type
- `500`: Internal server error

## Testing with cURL

```bash
# Test health endpoint
curl http://localhost:8000/

# Test image generation
curl -X POST http://localhost:8000/generate \
  -F "file=@test_image.jpg" \
  -F 'palette=[{"id":"1","rgb":[255,0,0],"note":"Red"},{"id":"2","rgb":[0,0,255],"note":"Blue"}]' \
  -F "threshold=50"
```

## Development

### Adding New Features

The codebase is modular:
- `validate_image()`: Input validation
- `map_to_custom_palette()`: Core color matching algorithm
- `simplify_regions()`: Region simplification
- `auto_canny()`: Edge detection
- `generate_line_art()`: Template generation

### Testing

Create test images in a `test_images/` directory and use the interactive docs at `/docs` for quick testing.

## Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Platforms

- **Render**: Connect GitHub repo, set build command to `pip install -r requirements.txt`
- **Railway**: Auto-detects Python and requirements.txt
- **Google Cloud Run**: Use the provided Dockerfile
- **AWS Lambda**: Use Mangum adapter for serverless deployment

## Troubleshooting

### Issue: "Failed to decode image"
- Ensure the uploaded file is a valid image format
- Check file isn't corrupted
- Try re-encoding with different quality settings

### Issue: "Palette must be a non-empty list"
- Verify JSON format is correct
- Ensure RGB values are arrays of 3 integers [0-255]

### Issue: CORS errors from mobile app
- Add Capacitor origins to CORS configuration
- Verify ENVIRONMENT variable is set correctly

## License

MIT

## Contributing

Pull requests welcome! Please ensure code follows PEP 8 style guidelines.
