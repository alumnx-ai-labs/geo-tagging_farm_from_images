"""
Step 1: Create annotation template
This creates a JSON file you'll fill manually
"""

import json
from pathlib import Path

def create_template():
    """Create empty annotation template"""
    
    # Find all images
    image_dir = Path('E:\HP\Alumnx\AgriGPT_Project\Latitude_Longitude\Clone\gps_finetuning\data\imagesmkdir -p gps_finetuning/data/images')
    images = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg')))
    
    if not images:
        print("No images found in data/images/")
        return
    
    print(f"Found {len(images)} images")
    
    # Create template
    annotations = []
    
    for i, img_path in enumerate(images):
        entry = {
            'image_id': i,
            'filename': img_path.name,
            'image_path': str(img_path),
            
            # FILL THESE MANUALLY by looking at each image:
            'gps_text': '',      # Example: "15.97036, 79.27803, 90.8m, 259°"
            'date_text': '',     # Example: "Dec 28, 2025 13:35:07"
            'latitude': 0.0,     # Example: 15.97036
            'longitude': 0.0,    # Example: 79.27803
        }
        annotations.append(entry)
    
    # Save template
    output_file = 'data/annotations.json'
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\n✓ Created template: {output_file}")
    print(f"Total images: {len(annotations)}")
    print("\nNext step: Fill in the GPS data for each image in annotations.json")
    
    return annotations

if __name__ == "__main__":
    create_template()