import cv2
import re
import pytesseract
import os
import pandas as pd
from pathlib import Path
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image_advanced(roi):
    """
    Advanced preprocessing pipeline for better OCR accuracy
    """
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Sharpen
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
    
    # Binary threshold (Otsu's method)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if background is dark
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    
    # Upscale significantly for OCR
    upscaled = cv2.resize(binary, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    # Light morphology to connect broken characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    morphed = cv2.morphologyEx(upscaled, cv2.MORPH_CLOSE, kernel)
    
    return morphed

def smart_fix_coordinate(coord_str, is_latitude=True):
    """
    Intelligently fix a coordinate based on expected patterns
    
    For this dataset:
    - Latitude should be 15.96XXX to 15.97XXX 
    - Longitude should be 79.27XXX to 79.28XXX
    """
    if not coord_str or coord_str == 'None':
        return None
    
    try:
        # Remove any spaces
        coord_str = coord_str.strip()
        
        # Parse current value
        parts = coord_str.split('.')
        if len(parts) != 2:
            return None
        
        whole_part = parts[0]
        decimal_part = parts[1]
        
        if is_latitude:
            # Expected pattern: 15.96XXX or 15.97XXX
            # Common OCR errors for latitude:
            # - 15 misread as: 18, 19, 45, 75, 95, 16, 13, 5
            # - First digit 1 misread as: 4, 7, 9, 8, 3, 5
            
            fixed_whole = whole_part
            
            # Fix patterns where first digit should be 1
            if len(whole_part) == 2:
                second_digit = whole_part[1]
                
                # If it's X5 where X is wrong, should be 15
                if second_digit == '5':
                    if whole_part in ['45', '75', '95', '85', '35', '65', '05']:
                        fixed_whole = '15'
                
                # If it's X6 where X is wrong, should be 16 (less common)
                elif second_digit == '6':
                    if whole_part in ['46', '76', '96', '86', '36', '66', '06']:
                        fixed_whole = '16'
                
                # Common specific misreads
                elif whole_part in ['18', '19', '13', '10', '17']:
                    fixed_whole = '15'
            
            # Single digit case (rare but possible)
            elif len(whole_part) == 1:
                if whole_part in ['5', '6']:
                    fixed_whole = '1' + whole_part
            
            # Three digit case (extra digit OCR error)
            elif len(whole_part) == 3:
                # Could be 015, 115, etc -> 15
                if whole_part.endswith('5'):
                    fixed_whole = '15'
                elif whole_part.endswith('6'):
                    fixed_whole = '16'
            
            # Check decimal part - should start with 96 or 97
            if len(decimal_part) >= 2:
                first_two = decimal_part[:2]
                # Common patterns: 96XXX or 97XXX
                if first_two not in ['96', '97']:
                    # If it's 9XXXX, likely correct pattern
                    if decimal_part[0] == '9':
                        pass  # Keep as is
                    # If completely wrong, try to infer
                    else:
                        # Most common is 97
                        if '7' in decimal_part[:3]:
                            decimal_part = '97' + decimal_part[2:]
                        elif '6' in decimal_part[:3]:
                            decimal_part = '96' + decimal_part[2:]
            
            # Ensure 5 decimal places
            if len(decimal_part) < 5:
                decimal_part = decimal_part.ljust(5, '0')
            elif len(decimal_part) > 5:
                decimal_part = decimal_part[:5]
            
            result = f"{fixed_whole}.{decimal_part}"
            
        else:  # Longitude
            # Expected pattern: 79.27XXX or 79.28XXX
            # Common OCR errors for longitude:
            # - 79 misread as: 70, 74, 75, 78, 29, 19, 49, 69, 89, 99
            
            fixed_whole = whole_part
            
            # Fix patterns where it should be 79
            if len(whole_part) == 2:
                # If second digit is 9, first should be 7
                if whole_part[1] == '9':
                    if whole_part[0] != '7':
                        fixed_whole = '79'
                
                # Common misreads
                elif whole_part in ['70', '74', '75', '78', '29', '19', '49', '69', '89', '99', '73', '76', '77']:
                    fixed_whole = '79'
                
                # If starts with 2-6, likely should be 79
                elif whole_part[0] in ['2', '3', '4', '5', '6']:
                    if whole_part[1] in ['7', '8', '9']:
                        fixed_whole = '79'
            
            # Check decimal part - should start with 27 or 28
            if len(decimal_part) >= 2:
                first_two = decimal_part[:2]
                if first_two not in ['27', '28']:
                    # Try to fix
                    if '7' in decimal_part[:3] or '8' in decimal_part[:3]:
                        if '8' in decimal_part[:3]:
                            decimal_part = '28' + decimal_part[2:]
                        else:
                            decimal_part = '27' + decimal_part[2:]
                    else:
                        # Default to most common
                        decimal_part = '27' + decimal_part[2:]
            
            # Ensure 5 decimal places
            if len(decimal_part) < 5:
                decimal_part = decimal_part.ljust(5, '0')
            elif len(decimal_part) > 5:
                decimal_part = decimal_part[:5]
            
            result = f"{fixed_whole}.{decimal_part}"
        
        # Final validation
        result_float = float(result)
        
        if is_latitude:
            if 14 <= result_float <= 17:  # Broader range for latitude
                return result
        else:
            if 78 <= result_float <= 81:  # Broader range for longitude
                return result
        
        return None
        
    except Exception as e:
        return None

def extract_coordinates(image_path):
    """
    Extract latitude and longitude with advanced preprocessing and error correction
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        
        if img is None:
            return {
                'filename': os.path.basename(image_path),
                'status': 'Failed to load',
                'latitude': None,
                'longitude': None,
                'raw_ocr': '',
                'cleaned_ocr': ''
            }
        
        h, w, _ = img.shape
        
        # Try multiple ROI strategies
        rois = [
            img[int(h * 0.88):h, 0:w],           # Bottom 12%
            img[int(h * 0.85):h, 0:w],           # Bottom 15%
            img[int(h * 0.90):h, 0:w],           # Bottom 10%
            img[int(h * 0.92):h, 0:w],           # Bottom 8%
        ]
        
        all_results = []
        
        for roi_idx, roi in enumerate(rois):
            # Method 1: Advanced preprocessing
            processed1 = preprocess_image_advanced(roi)
            
            # Method 2: HSV masking (original approach)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_white = (0, 0, 180)
            upper_white = (180, 60, 255)
            mask = cv2.inRange(hsv, lower_white, upper_white)
            mask = cv2.resize(mask, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            processed2 = cv2.dilate(mask, kernel, iterations=1)
            
            # Try both processed images
            for method_idx, processed in enumerate([processed1, processed2]):
                # OCR with multiple configs
                configs = [
                    r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789., ',
                    r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789., ',
                    r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789., ',
                    r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789., ',
                ]
                
                for config in configs:
                    text = pytesseract.image_to_string(processed, config=config)
                    
                    # Clean up text
                    raw_text = text.replace('\n', ' ').replace('\r', ' ')
                    raw_text = re.sub(r'\s+', ' ', raw_text).strip()
                    
                    if not raw_text:
                        continue
                    
                    # Extract coordinates
                    result = extract_coords_from_text(raw_text)
                    
                    if result['latitude'] and result['longitude']:
                        all_results.append(result)
        
        # Select best result
        if all_results:
            # Sort by confidence (valid coordinates first)
            valid_results = [r for r in all_results if r['status'] == 'Success']
            
            if valid_results:
                return valid_results[0]
        
        # If all methods failed, return failure with best raw text
        return {
            'filename': os.path.basename(image_path),
            'status': 'Failed',
            'latitude': None,
            'longitude': None,
            'raw_ocr': all_results[0]['raw_ocr'] if all_results else '',
            'cleaned_ocr': ''
        }
        
    except Exception as e:
        return {
            'filename': os.path.basename(image_path),
            'status': f'Error: {str(e)}',
            'latitude': None,
            'longitude': None,
            'raw_ocr': '',
            'cleaned_ocr': ''
        }

def extract_coords_from_text(raw_text):
    """
    Extract coordinates from raw OCR text with aggressive cleaning
    """
    # === AGGRESSIVE CLEANING ===
    text = raw_text
    
    # Remove leading/trailing junk
    text = re.sub(r'(?:^|\s)\.+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    
    # Fix common OCR patterns
    text = re.sub(r'0\.(\d{2}\.\d+)', r'\1', text)  # Remove leading 0.
    text = re.sub(r'(\d{2}),(\d{5})', r'\1.\2', text)  # Comma to dot
    text = re.sub(r'(\d{2}),\.(\d{5})', r'\1.\2', text)  # Fix comma-dot
    text = re.sub(r'\.(\d{2}\.\d{4,6})', r'\1', text)  # Remove leading dot
    
    # Fix stuck numbers (no dots): 7927808 -> 79.27808
    text = re.sub(r'(?:^|\s|,)([7-8]\d)(\d{5})(?=\s|,|$)', r' \1.\2', text)
    
    # Split stuck lat/lon pairs: 15.9704279.27806 -> 15.97042 79.27806
    text = re.sub(r'(\d{2}\.\d{5})(\d{2}\.\d{5})', r'\1 \2', text)
    
    # Handle completely stuck: 1597042792780 6 -> 15.97042 79.27806
    text = re.sub(r'(?:^|\s)(\d{2})(\d{5})(\d{2})(\d{5})', r' \1.\2 \3.\4', text)
    
    # Clean spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # === EXTRACT ALL COORDINATE-LIKE PATTERNS ===
    # Find all XX.XXXXX patterns (2 digits, dot, 4-6 digits)
    all_numbers = re.findall(r'\d{2}\.\d{4,6}', text)
    
    latitude = None
    longitude = None
    
    # Try to find valid lat/lon pairs
    for i in range(len(all_numbers)):
        for j in range(i, len(all_numbers)):
            lat_candidate = all_numbers[i]
            lon_candidate = all_numbers[j] if j < len(all_numbers) else None
            
            if not lon_candidate:
                continue
            
            # Apply smart fixes
            fixed_lat = smart_fix_coordinate(lat_candidate, is_latitude=True)
            fixed_lon = smart_fix_coordinate(lon_candidate, is_latitude=False)
            
            if fixed_lat and fixed_lon:
                # Validate
                try:
                    lat_val = float(fixed_lat)
                    lon_val = float(fixed_lon)
                    
                    # Check if in valid range
                    if 14 <= lat_val <= 17 and 78 <= lon_val <= 81:
                        latitude = fixed_lat
                        longitude = fixed_lon
                        break
                except ValueError:
                    continue
        
        if latitude and longitude:
            break
    
    # If still not found, try extracting any numbers and fixing them
    if not latitude or not longitude:
        # Look for patterns like: XX.XX XXX or XX XX XXX
        all_coords = re.findall(r'\d+\.?\d*', text)
        
        # Try to construct coordinates from parts
        for i in range(len(all_coords) - 1):
            # Try to build latitude
            lat_candidate = all_coords[i]
            if '.' not in lat_candidate and len(lat_candidate) >= 7:
                # Could be stuck: 1597042
                lat_candidate = lat_candidate[:2] + '.' + lat_candidate[2:7]
            
            lon_candidate = all_coords[i + 1]
            if '.' not in lon_candidate and len(lon_candidate) >= 7:
                # Could be stuck: 7927806
                lon_candidate = lon_candidate[:2] + '.' + lon_candidate[2:7]
            
            # Apply smart fixes
            fixed_lat = smart_fix_coordinate(lat_candidate, is_latitude=True)
            fixed_lon = smart_fix_coordinate(lon_candidate, is_latitude=False)
            
            if fixed_lat and fixed_lon:
                try:
                    lat_val = float(fixed_lat)
                    lon_val = float(fixed_lon)
                    
                    if 14 <= lat_val <= 17 and 78 <= lon_val <= 81:
                        latitude = fixed_lat
                        longitude = fixed_lon
                        break
                except ValueError:
                    continue
    
    status = 'Success' if latitude and longitude else 'Failed'
    
    return {
        'status': status,
        'latitude': latitude,
        'longitude': longitude,
        'raw_ocr': raw_text[:150],
        'cleaned_ocr': text[:150]
    }

def process_images_for_coordinates(folder_path, output_csv='coordinates_extracted.csv'):
    """
    Process all images and extract coordinates
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    folder = Path(folder_path)
    image_files = sorted([f for f in folder.iterdir() 
                         if f.suffix.lower() in image_extensions])
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return None
    
    print(f"Processing {len(image_files)} images for GPS coordinates...")
    print("="*70)
    
    results = []
    success_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        result = extract_coordinates(str(img_path))
        results.append(result)
        
        if result['status'] == 'Success':
            success_count += 1
            status_mark = '[OK]'
        else:
            status_mark = '[FAIL]'
        
        # Progress display
        lat_display = result['latitude'] if result['latitude'] else 'N/A'
        lon_display = result['longitude'] if result['longitude'] else 'N/A'
        print(f"{status_mark} [{i:3d}/{len(image_files)}] {img_path.name[:50]:50s} | Lat: {lat_display:10s} | Lon: {lon_display:10s}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = folder / output_csv
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print(f"Results saved to: {output_path}")
    print("="*70)
    
    # Statistics
    success_rate = (success_count / len(image_files)) * 100
    print(f"\nSUMMARY:")
    print(f"  Total images:           {len(image_files)}")
    print(f"  Successful extractions: {success_count} ({success_rate:.1f}%)")
    print(f"  Failed extractions:     {len(image_files) - success_count}")
    
    # Show failed cases with details
    failed_df = df[df['status'] != 'Success']
    if not failed_df.empty:
        print(f"\n{len(failed_df)} images need manual review:")
        print("-"*70)
        for idx, row in failed_df.iterrows():
            print(f"  [{idx+2}] {row['filename']}")
            if row['raw_ocr']:
                print(f"      OCR: {row['raw_ocr'][:80]}")
    
    return df

def verify_and_fix_coordinates(csv_path):
    """
    Final verification and correction pass
    """
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("COORDINATE VALIDATION & AUTO-FIX")
    print("="*70)
    
    fixed_count = 0
    
    for idx, row in df.iterrows():
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            continue
        
        lat_str = str(row['latitude'])
        lon_str = str(row['longitude'])
        
        # Apply smart fixes
        fixed_lat = smart_fix_coordinate(lat_str, is_latitude=True)
        fixed_lon = smart_fix_coordinate(lon_str, is_latitude=False)
        
        if fixed_lat and fixed_lon:
            if fixed_lat != lat_str or fixed_lon != lon_str:
                df.at[idx, 'latitude'] = fixed_lat
                df.at[idx, 'longitude'] = fixed_lon
                df.at[idx, 'status'] = 'Success (Auto-fixed)'
                fixed_count += 1
                print(f"Fixed row {idx+2}: {lat_str}, {lon_str} -> {fixed_lat}, {fixed_lon}")
    
    if fixed_count > 0:
        # Save updated CSV
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n[AUTO-FIXED] {fixed_count} coordinates corrected")
    else:
        print("\nNo additional corrections needed")
    
    # Final statistics
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    valid = df[(df['latitude'] >= 14) & (df['latitude'] <= 17) & 
               (df['longitude'] >= 78) & (df['longitude'] <= 81)]
    
    print(f"\nFINAL RESULTS:")
    print(f"  Valid coordinates: {len(valid)}/{len(df)} ({len(valid)/len(df)*100:.1f}%)")
    print(f"  Latitude range: {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
    print(f"  Longitude range: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
    
    # Show any remaining issues
    invalid = df[(df['latitude'].notna()) & (df['longitude'].notna()) &
                 ~((df['latitude'] >= 14) & (df['latitude'] <= 17) & 
                   (df['longitude'] >= 78) & (df['longitude'] <= 81))]
    
    if not invalid.empty:
        print(f"\n[WARNING] {len(invalid)} coordinates still out of range:")
        for idx, row in invalid.iterrows():
            print(f"  Row {idx+2}: {row['filename']}")
            print(f"    Current: Lat={row['latitude']}, Lon={row['longitude']}")

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    folder_path = r"E:\HP\Alumnx\AgriGPT_Project\Latitude_Longitude\Plant_Images"
    
    print("ADVANCED GPS COORDINATE EXTRACTOR v2.1")
    print("="*70 + "\n")
    
    # Process images
    df = process_images_for_coordinates(folder_path, output_csv='coordinates_final.csv')
    
    if df is not None:
        # Verify and auto-fix
        verify_and_fix_coordinates(folder_path + r'\coordinates_final.csv')
        
        print("\n" + "="*70)
        print("[COMPLETE] Check 'coordinates_final.csv' for results")
        print("="*70)