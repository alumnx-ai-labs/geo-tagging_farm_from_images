"""
Improved test with better GPS parsing
"""

import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import re
import sys

def extract_gps_coordinates(text):
    """
    More robust GPS extraction
    Handles various formats and OCR errors
    """
    # Clean the text
    text = text.replace(',', '.')
    text = text.replace(':', '.')
    
    # Try multiple patterns
    patterns = [
        # Standard: 15.97036, 79.27803
        r'(\d{1,2}\.\d{4,6})[,\s]+(\d{1,2}\.\d{4,6})',
        # Without decimal: 15969, 79278 -> 15.969, 79.278
        r'(\d{4,5})[,\s]+(\d{4,5})',
        # OCR errors: 15969,.018 79278,.18
        r'(\d{4,5})[,\.\s]+(\d{4,5})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            lat_str = match.group(1)
            lon_str = match.group(2)
            
            # Convert to proper decimal format
            lat = float(lat_str)
            lon = float(lon_str)
            
            # If numbers are too large, insert decimal
            if lat > 100:
                lat = lat / 1000  # 15969 -> 15.969
            if lon > 100:
                lon = lon / 1000  # 79278 -> 79.278
            
            # Validate coordinates (India region)
            if 8 < lat < 37 and 68 < lon < 98:
                return lat, lon
    
    return None, None

def test_model(image_path, model_path="./models/gps-trocr-final"):
    print(f"\nTesting: {image_path}")
    print("-"*60)
    
    from pathlib import Path
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return None
    
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    try:
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        crop = img.crop((0, int(height * 0.80), width, height))
    except Exception as e:
        print(f"Error reading image: {e}")
        return None
    
    print("Generating prediction...")
    pixel_values = processor(crop, return_tensors="pt").pixel_values.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"\nRaw prediction: {predicted_text}")
    
    # Extract GPS with improved parsing
    lat, lon = extract_gps_coordinates(predicted_text)
    
    if lat and lon:
        print(f"\n" + "="*60)
        print("Extracted GPS Coordinates:")
        print("="*60)
        print(f"Latitude:  {lat:.6f}")
        print(f"Longitude: {lon:.6f}")
        
        maps_url = f"https://www.google.com/maps?q={lat},{lon}"
        print(f"\nGoogle Maps: {maps_url}")
        print("="*60)
    else:
        print("\n✗ Could not extract valid GPS coordinates")
    
    print("-"*60)
    return predicted_text

def main():
    test_image = sys.argv[1] if len(sys.argv) > 1 else "data/images/WhatsApp Image 2025-12-29 at 12.02.42.jpeg"
    result = test_model(test_image)
    
    if result:
        print("\n✓ Test completed!")

if __name__ == "__main__":
    main()