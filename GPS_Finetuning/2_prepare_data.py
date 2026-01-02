"""
Step 2: Prepare data for training
Crops GPS regions and creates train/val split
"""

import json
import cv2
from pathlib import Path
import random

def prepare_data():
    """Crop GPS regions and split dataset"""
    
    # Load annotations
    with open('data/annotations.json', 'r') as f:
        annotations = json.load(f)
    
    print(f"Processing {len(annotations)} images...")
    
    processed = []
    skipped = 0
    
    for item in annotations:
        # Skip if no GPS text
        if not item.get('gps_text'):
            skipped += 1
            continue
        
        # Read image
        img = cv2.imread(item['image_path'])
        if img is None:
            print(f"✗ Failed to read: {item['filename']}")
            skipped += 1
            continue
        
        height, width = img.shape[:2]
        
        # Crop bottom 20% (where GPS text appears)
        crop_y_start = int(height * 0.80)
        crop = img[crop_y_start:, :]
        
        # Save crop
        crop_filename = f"crop_{item['image_id']:03d}.jpg"
        crop_path = f"data/crops/{crop_filename}"
        cv2.imwrite(crop_path, crop)
        
        # Combine GPS and date text
        full_text = item['gps_text']
        if item.get('date_text'):
            full_text = f"{item['gps_text']}\n{item['date_text']}"
        
        # Add to processed list
        processed.append({
            'image_id': item['image_id'],
            'crop_path': crop_path,
            'full_text': full_text,
            'latitude': item.get('latitude', 0.0),
            'longitude': item.get('longitude', 0.0)
        })
        
        print(f"✓ {item['filename']}")
    
    print(f"\n✓ Processed: {len(processed)}")
    print(f"✗ Skipped: {skipped}")
    
    if len(processed) == 0:
        print("\nNo valid data! Please fill in GPS text in annotations.json")
        return
    
    # Shuffle and split (90% train, 10% val)
    random.seed(42)
    random.shuffle(processed)
    
    split_idx = int(0.9 * len(processed))
    train_data = processed[:split_idx]
    val_data = processed[split_idx:]
    
    # Save splits
    with open('data/train_annotations.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('data/val_annotations.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print("\n✓ Dataset ready for training!")

if __name__ == "__main__":
    prepare_data()