"""
Auto-annotate images using Google Vision API
Reads: data/annotations.json (template)
Fills: GPS data using Google Vision API
Saves: data/annotations_filled.json
"""

import json
import requests
import base64
import re
from pathlib import Path

def extract_text_from_image(image_path, api_key):
    """
    Extract text from image using Google Vision API
    
    Args:
        image_path: Path to image file
        api_key: Google Cloud Vision API key
        
    Returns:
        Extracted text or None
    """
    try:
        # Read and encode image
        with open(image_path, 'rb') as image_file:
            image_content = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare API request
        url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'
        
        payload = {
            'requests': [{
                'image': {
                    'content': image_content
                },
                'features': [{
                    'type': 'TEXT_DETECTION'
                }]
            }]
        }
        
        # Make API request
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            print(f"    API Error: {response.status_code}")
            return None
        
        result = response.json()
        
        # Extract text
        if 'responses' in result and len(result['responses']) > 0:
            response_data = result['responses'][0]
            if 'textAnnotations' in response_data and len(response_data['textAnnotations']) > 0:
                # First annotation contains all detected text
                return response_data['textAnnotations'][0]['description']
        
        return None
        
    except Exception as e:
        print(f"    Error: {e}")
        return None


def parse_gps_from_text(text):
    """
    Parse GPS coordinates and metadata from extracted text
    
    Args:
        text: Extracted text from image
        
    Returns:
        Dictionary with GPS data or None
    """
    if not text:
        return None
    
    # Extract latitude and longitude
    # Patterns to match: 15.97036, 79.27803 or 15.97036,79.27803
    coord_patterns = [
        r'(\d{1,3}\.\d{4,})\s*,\s*(\d{1,3}\.\d{4,})',  # With spaces
        r'(\d{1,3}\.\d{4,}),(\d{1,3}\.\d{4,})',         # Without spaces
    ]
    
    lat, lon = None, None
    for pattern in coord_patterns:
        match = re.search(pattern, text)
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            break
    
    if lat is None or lon is None:
        return None
    
    # Extract altitude (e.g., 90.8m or 90.8 m)
    altitude = 0.0
    alt_match = re.search(r'([\d.]+)\s*m', text, re.IGNORECASE)
    if alt_match:
        altitude = float(alt_match.group(1))
    
    # Extract direction (e.g., 259° or 259 °)
    direction = 0
    dir_match = re.search(r'(\d{1,3})\s*°', text)
    if dir_match:
        direction = int(dir_match.group(1))
    
    # Extract date/time (e.g., Dec 28, 2025 13:35:07)
    date_text = ''
    date_patterns = [
        r'(\w{3}\s+\d{1,2},?\s+\d{4}\s+\d{1,2}:\d{2}:\d{2})',  # Dec 28, 2025 13:35:07
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{4}\s+\d{1,2}:\d{2}:\d{2})', # 28-12-2025 13:35:07
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            date_text = match.group(1)
            break
    
    # Construct GPS text line
    # Try to extract the exact GPS line from the text
    gps_text = f"{lat}, {lon}"
    if altitude > 0:
        gps_text += f", {altitude}m"
    if direction > 0:
        gps_text += f", {direction}°"
    
    return {
        'gps_text': gps_text,
        'date_text': date_text,
        'latitude': lat,
        'longitude': lon,
        'altitude': altitude,
        'direction': direction,
        'extracted_text': text.strip()
    }


def auto_annotate(api_key, template_file='data/annotations.json', output_file='data/annotations_filled.json'):
    """
    Auto-annotate all images using Google Vision API
    
    Args:
        api_key: Google Cloud Vision API key
        template_file: Path to template JSON file
        output_file: Path to save filled annotations
    """
    
    print("="*70)
    print("Auto-Annotation with Google Vision API")
    print("="*70)
    
    # Load template
    print(f"\nLoading template: {template_file}")
    try:
        with open(template_file, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Template file not found: {template_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in template file")
        return
    
    print(f"Found {len(annotations)} images in template")
    
    # Process each image
    print("\n" + "-"*70)
    print("Processing images...")
    print("-"*70)
    
    success_count = 0
    failed_count = 0
    
    for i, item in enumerate(annotations):
        image_path = item['image_path']
        filename = item['filename']
        
        print(f"\n[{i+1}/{len(annotations)}] {filename}")
        
        # Check if image exists
        if not Path(image_path).exists():
            print(f"    ✗ Image not found: {image_path}")
            failed_count += 1
            continue
        
        # Extract text using Vision API
        print(f"    Calling Vision API...")
        extracted_text = extract_text_from_image(image_path, api_key)
        
        if not extracted_text:
            print(f"    ✗ No text detected")
            failed_count += 1
            continue
        
        # Parse GPS data
        gps_data = parse_gps_from_text(extracted_text)
        
        if not gps_data:
            print(f"    ✗ Could not parse GPS coordinates")
            print(f"    Extracted: {extracted_text[:100]}")
            failed_count += 1
            continue
        
        # Update annotation
        item['gps_text'] = gps_data['gps_text']
        item['date_text'] = gps_data['date_text']
        item['latitude'] = gps_data['latitude']
        item['longitude'] = gps_data['longitude']
        
        # Optional: store additional fields if they exist
        if 'altitude' not in item or item.get('altitude', 0) == 0:
            item['altitude'] = gps_data['altitude']
        if 'direction' not in item or item.get('direction', 0) == 0:
            item['direction'] = gps_data['direction']
        
        success_count += 1
        print(f"    ✓ GPS: {gps_data['latitude']}, {gps_data['longitude']}")
        if gps_data['altitude'] > 0:
            print(f"    ✓ Altitude: {gps_data['altitude']}m")
        if gps_data['direction'] > 0:
            print(f"    ✓ Direction: {gps_data['direction']}°")
        if gps_data['date_text']:
            print(f"    ✓ Date: {gps_data['date_text']}")
    
    # Save filled annotations
    print("\n" + "="*70)
    print(f"Saving annotations to: {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("Auto-Annotation Complete!")
    print("="*70)
    print(f"Total images:    {len(annotations)}")
    print(f"Successfully annotated: {success_count}")
    print(f"Failed:          {failed_count}")
    print(f"Success rate:    {(success_count/len(annotations)*100):.1f}%")
    print(f"\nOutput file:     {output_file}")
    print("\nNext step: python 2_prepare_data.py")
    print("="*70)


def main():
    """Main function"""
    
    print("\n" + "="*70)
    print("Google Vision API Auto-Annotation Script")
    print("="*70)
    
    # Get API key from user
    print("\nPlease enter your Google Cloud Vision API key:")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("\nError: API key is required!")
        print("\nTo get an API key:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create/select a project")
        print("3. Enable Cloud Vision API")
        print("4. Go to 'APIs & Services' > 'Credentials'")
        print("5. Create API Key")
        return
    
    # Verify template file exists
    template_file = 'data/annotations.json'
    if not Path(template_file).exists():
        print(f"\nError: Template file not found: {template_file}")
        print("Please make sure data/annotations.json exists")
        return
    
    # Run auto-annotation
    auto_annotate(
        api_key=api_key,
        template_file='data/annotations.json',
        output_file='data/annotations_filled.json'
    )


if __name__ == "__main__":
    main()