"""
Step 4: Test the trained model
"""

import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import cv2
import re

def test_model(image_path, model_path="./models/gps-trocr-final"):
    """Test model on an image"""
    
    print(f"\nTesting: {image_path}")
    print("-"*60)
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load image
    img = cv2.imread(image_path)
    height = img.shape[0]
    
    # Crop bottom region
    crop = img[int(height * 0.80):, :]
    
    # Convert to PIL
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    
    # Process
    pixel_values = processor(crop_pil, return_tensors="pt").pixel_values.to(device)
    
    # Generate
    print("Generating prediction...")
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    
    # Decode
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"\nPredicted: {predicted_text}")
    
    # Parse GPS
    coords = re.findall(r'(\d+\.\d+)', predicted_text)
    if len(coords) >= 2:
        print(f"\nExtracted GPS:")
        print(f"  Latitude:  {coords[0]}")
        print(f"  Longitude: {coords[1]}")
        
        # Google Maps link
        maps_url = f"https://www.google.com/maps?q={coords[0]},{coords[1]}"
        print(f"  Google Maps: {maps_url}")
    else:
        print("\n✗ Could not extract GPS coordinates")
    
    print("-"*60)
    
    return predicted_text

if __name__ == "__main__":
    # Test on one of your images
    test_image = "data/images/WhatsApp_Image_2025-12-28_at_13_38_07.jpeg"
    test_model(test_image)