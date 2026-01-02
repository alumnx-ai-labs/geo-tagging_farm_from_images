"""
Step 4: Test the trained model
"""

import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import re
import sys

def test_model(image_path, model_path="./models/gps-trocr-final"):
    """Test model on an image"""
    
    print(f"\nTesting: {image_path}")
    print("-"*60)
    
    # Check if image exists
    from pathlib import Path
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return None
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Make sure the model exists at: {model_path}")
        return None
    
    # Load image using PIL
    try:
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        
        # Crop bottom 20% where GPS text appears
        crop_box = (0, int(height * 0.80), width, height)
        crop = img.crop(crop_box)
        
    except Exception as e:
        print(f"Error reading image: {e}")
        return None
    
    # Process
    print("Generating prediction...")
    pixel_values = processor(crop, return_tensors="pt").pixel_values.to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    
    # Decode
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"\nPredicted text: {predicted_text}")
    
    # Parse GPS coordinates
    coords = re.findall(r'(\d+\.\d+)', predicted_text)
    
    if len(coords) >= 2:
        print(f"\n" + "="*60)
        print("Extracted GPS Coordinates:")
        print("="*60)
        print(f"Latitude:  {coords[0]}")
        print(f"Longitude: {coords[1]}")
        
        if len(coords) >= 3:
            print(f"Altitude:  {coords[2]}m")
        
        # Extract direction
        direction = re.search(r'(\d{1,3})°', predicted_text)
        if direction:
            print(f"Direction: {direction.group(1)}°")
        
        # Google Maps link
        maps_url = f"https://www.google.com/maps?q={coords[0]},{coords[1]}"
        print(f"\nGoogle Maps: {maps_url}")
        print("="*60)
    else:
        print("\n✗ Could not extract GPS coordinates from prediction")
    
    print("-"*60)
    
    return predicted_text

def main():
    """Main function"""
    
    # Default test image
    test_image = "data/images/WhatsApp_Image_2025-12-28_at_13_38_07.jpeg"
    
    # Check if user provided image path
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    # Test the model
    result = test_model(test_image)
    
    if result:
        print("\n✓ Test completed successfully!")
    else:
        print("\n✗ Test failed")

if __name__ == "__main__":
    main()