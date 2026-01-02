"""
Step 2: Prepare data for training (Fixed for path issues)
- Fixes file paths (Windows vs Linux)
- Crops GPS regions from images
- Creates train/val split
"""

import json
import cv2
from pathlib import Path
import random
import os

def fix_path(path_str):
    """
    Fix path to work on current OS
    Converts Windows paths to Unix and vice versa
    """
    # Convert to Path object which handles OS differences
    path = Path(path_str)
    
    # If path doesn't exist, try to find the image in data/images/
    if not path.exists():
        # Get just the filename
        filename = path.name
        # Try in data/images/
        alt_path = Path('data/images') / filename
        if alt_path.exists():
            return str(alt_path)
        
        # Try without data/ prefix
        alt_path2 = Path('images') / filename
        if alt_path2.exists():
            return str(alt_path2)
    
    return str(path)

def prepare_data():
    """Crop GPS regions and split dataset"""
    
    print("="*70)
    print("Preparing Training Data")
    print("="*70)
    
    # Load annotations
    annotations_file = 'data/annotations.json'
    print(f"\nLoading annotations from: {annotations_file}")
    
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {annotations_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file")
        return
    
    print(f"Found {len(annotations)} annotations")
    
    # Create crops directory
    crops_dir = Path('data/crops')
    crops_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each image
    print("\n" + "-"*70)
    print("Cropping GPS regions from images...")
    print("-"*70)
    
    processed = []
    skipped = 0
    
    for i, item in enumerate(annotations):
        filename = item.get('filename', f"image_{i}")
        
        print(f"\n[{i+1}/{len(annotations)}] {filename}")
        
        # Check if GPS text exists
        if not item.get('gps_text'):
            print(f"     No GPS text - skipping")
            skipped += 1
            continue
        
        # Check if latitude/longitude exist
        if item.get('latitude', 0) == 0 or item.get('longitude', 0) == 0:
            print(f"     Invalid GPS coordinates - skipping")
            skipped += 1
            continue
        
        # Fix image path
        original_path = item.get('image_path', '')
        fixed_path = fix_path(original_path)
        
        # Check if image exists
        if not Path(fixed_path).exists():
            print(f"     Image not found")
            print(f"       Original path: {original_path}")
            print(f"       Fixed path: {fixed_path}")
            skipped += 1
            continue
        
        # Read image
        img = cv2.imread(fixed_path)
        if img is None:
            print(f"     Failed to read image: {fixed_path}")
            skipped += 1
            continue
        
        height, width = img.shape[:2]
        
        # Crop bottom 20% of image (where GPS watermark typically appears)
        crop_y_start = int(height * 0.80)
        crop = img[crop_y_start:, :]
        
        # Save cropped image
        crop_filename = f"crop_{item['image_id']:03d}.jpg"
        crop_path = crops_dir / crop_filename
        cv2.imwrite(str(crop_path), crop)
        
        # Combine GPS text and date text for training
        full_text = item['gps_text']
        if item.get('date_text'):
            full_text = f"{item['gps_text']}\n{item['date_text']}"
        
        # Add to processed list
        processed.append({
            'image_id': item['image_id'],
            'crop_path': str(crop_path),
            'full_text': full_text,
            'gps_text': item['gps_text'],
            'date_text': item.get('date_text', ''),
            'latitude': item['latitude'],
            'longitude': item['longitude']
        })
        
        print(f"     Cropped and saved")
        print(f"    GPS: {item['latitude']}, {item['longitude']}")
    
    # Summary
    print("\n" + "="*70)
    print("Cropping Summary")
    print("="*70)
    print(f"Total annotations: {len(annotations)}")
    print(f"Successfully processed: {len(processed)}")
    print(f"Skipped: {skipped}")
    
    if len(processed) == 0:
        print("\n✗ No valid data to process!")
        print("Please check that your images are in data/images/ directory")
        return
    
    # Create train/val split (90% train, 10% validation)
    print("\n" + "-"*70)
    print("Creating train/validation split...")
    print("-"*70)
    
    # Shuffle data
    random.seed(42)  # For reproducibility
    random.shuffle(processed)
    
    # Split
    split_idx = int(0.9 * len(processed))
    train_data = processed[:split_idx]
    val_data = processed[split_idx:]
    
    # Save train annotations
    train_file = 'data/train_annotations.json'
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    # Save validation annotations
    val_file = 'data/val_annotations.json'
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"\nTrain file: {train_file}")
    print(f"Val file: {val_file}")
    
    # Final summary
    print("\n" + "="*70)
    print(" Data Preparation Complete!")
    print("="*70)
    print(f"Cropped images saved to: {crops_dir}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"\nNext step: python 3_train.py")
    print("="*70)

if __name__ == "__main__":
    prepare_data()