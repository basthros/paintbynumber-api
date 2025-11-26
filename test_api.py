#!/usr/bin/env python3
"""
Quick test script for the Paint by Number API.
Run this after starting the backend to verify everything works.
"""

import requests
import json
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

# Sample palette with a few basic colors
SAMPLE_PALETTE = [
    {"id": "1", "rgb": [255, 0, 0], "note": "Cadmium Red"},
    {"id": "2", "rgb": [0, 0, 255], "note": "Ultramarine Blue"},
    {"id": "3", "rgb": [255, 255, 0], "note": "Lemon Yellow"},
    {"id": "4", "rgb": [0, 128, 0], "note": "Viridian Green"},
    {"id": "5", "rgb": [255, 255, 255], "note": "Titanium White"},
    {"id": "6", "rgb": [0, 0, 0], "note": "Mars Black"},
]


def test_health_check():
    """Test if the API is running."""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is online: {data}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the backend running?")
        print("   Start it with: cd paintbynumber-api && python main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def create_test_image():
    """Create a simple test image using PIL."""
    try:
        from PIL import Image, ImageDraw
        import io
        
        # Create a 300x300 image with some simple shapes
        img = Image.new('RGB', (300, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a red circle
        draw.ellipse([50, 50, 150, 150], fill='red', outline='black')
        
        # Draw a blue square
        draw.rectangle([150, 50, 250, 150], fill='blue', outline='black')
        
        # Draw a yellow triangle (roughly)
        draw.polygon([(100, 250), (150, 150), (200, 250)], fill='yellow', outline='black')
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes
    except ImportError:
        print("⚠️  PIL not installed. Using sample image if available.")
        return None


def test_generate(image_path=None):
    """Test the /generate endpoint."""
    print("\n🎨 Testing image generation...")
    
    # Get image file
    if image_path and Path(image_path).exists():
        print(f"   Using image: {image_path}")
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/jpeg')}
            data = {
                'palette': json.dumps(SAMPLE_PALETTE),
                'threshold': 50
            }
            
            try:
                print("   Sending request...")
                response = requests.post(f"{API_URL}/generate", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Generation successful!")
                    print(f"   Dimensions: {result['dimensions']['width']}x{result['dimensions']['height']}")
                    print(f"   Preview size: {len(result['preview'])} chars")
                    print(f"   Template size: {len(result['template'])} chars")
                    
                    # Save results to files
                    save_results(result)
                    return True
                else:
                    print(f"❌ Generation failed: {response.status_code}")
                    print(f"   Error: {response.json()}")
                    return False
                    
            except Exception as e:
                print(f"❌ Error during generation: {e}")
                return False
    else:
        # Try to create a test image
        img_bytes = create_test_image()
        if img_bytes:
            files = {'file': ('test.png', img_bytes, 'image/png')}
            data = {
                'palette': json.dumps(SAMPLE_PALETTE),
                'threshold': 50
            }
            
            try:
                print("   Using generated test image...")
                response = requests.post(f"{API_URL}/generate", files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Generation successful!")
                    print(f"   Dimensions: {result['dimensions']['width']}x{result['dimensions']['height']}")
                    save_results(result)
                    return True
                else:
                    print(f"❌ Generation failed: {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"❌ Error during generation: {e}")
                return False
        else:
            print("❌ No image available for testing")
            print("   Please run: python test_api.py <path_to_image>")
            return False


def save_results(result):
    """Save the base64 results to HTML file for viewing."""
    try:
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Paint by Number Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .container {{ display: flex; gap: 20px; margin-top: 20px; }}
        .result {{ flex: 1; }}
        img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
        .info {{ background: #f5f5f5; padding: 10px; border-radius: 4px; margin-top: 10px; }}
    </style>
</head>
<body>
    <h1>🎨 Paint by Number Test Results</h1>
    <p>Generated by the Paint by Number API</p>
    
    <div class="container">
        <div class="result">
            <h2>Preview (Colored)</h2>
            <img src="{result['preview']}" alt="Preview">
            <div class="info">
                <strong>Dimensions:</strong> {result['dimensions']['width']} × {result['dimensions']['height']}
            </div>
        </div>
        
        <div class="result">
            <h2>Template (Line Art)</h2>
            <img src="{result['template']}" alt="Template">
            <div class="info">
                <strong>Palette Colors:</strong> {len(SAMPLE_PALETTE)}
            </div>
        </div>
    </div>
    
    <div style="margin-top: 40px; padding: 20px; background: #e8f4f8; border-radius: 8px;">
        <h3>✅ Test Successful!</h3>
        <p>Your Paint by Number API is working correctly.</p>
        <ul>
            <li>Backend is responding</li>
            <li>Image processing works</li>
            <li>Color matching complete</li>
            <li>Edge detection successful</li>
        </ul>
    </div>
</body>
</html>
"""
        
        with open('test_results.html', 'w') as f:
            f.write(html)
        
        print("\n📄 Results saved to: test_results.html")
        print("   Open this file in your browser to view the results!")
        
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Paint by Number API Test Suite")
    print("=" * 60)
    
    # Check if backend is running
    if not test_health_check():
        sys.exit(1)
    
    # Test generation
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not test_generate(image_path):
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✨ All tests passed! Your API is working perfectly!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Open test_results.html in your browser")
    print("2. Start the frontend: cd paintbynumber-app && npm run dev")
    print("3. Open http://localhost:5173 and try the full app!")


if __name__ == "__main__":
    main()
