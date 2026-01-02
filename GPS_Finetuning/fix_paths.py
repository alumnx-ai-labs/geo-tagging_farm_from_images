"""
Fix Windows paths to Linux paths in annotations.json
"""

import json

print("Fixing paths in annotations.json...")

# Load annotations
with open('data/annotations.json', 'r') as f:
    annotations = json.load(f)

print(f"Found {len(annotations)} annotations")

# Fix each path
fixed_count = 0
for item in annotations:
    filename = item['filename']
    
    # Change from Windows path to Linux path
    old_path = item['image_path']
    new_path = f"data/images/{filename}"
    
    item['image_path'] = new_path
    fixed_count += 1

# Save fixed annotations
with open('data/annotations.json', 'w') as f:
    json.dump(annotations, f, indent=2)

print(f"\n✓ Fixed {fixed_count} paths")
print("✓ Saved to data/annotations.json")
print("\nNow run: python 2_prepare_data.py")