import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re

# Configuration
IMAGE_PATH = "sample_2.jpg"  # Change this to your image path

ROI_X0, ROI_Y0 = 0.10, 0.36  # Adjust ROI as needed for your image
ROI_X1, ROI_Y1 = 0.95, 0.70

# Store call numbers with their positions
call_number_registry = {}

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

def crop_roi(img):
    h, w = img.shape[:2]
    x0 = int(w * ROI_X0)
    y0 = int(h * ROI_Y0)
    x1 = int(w * ROI_X1)
    y1 = int(h * ROI_Y1)
    roi = img[y0:y1, x0:x1].copy()
    return roi, (x0, y0, x1, y1)

def preprocess_for_ocr(roi_bgr):
    """Enhanced preprocessing for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    # Adaptive thresholding for better text extraction
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                cv2.THRESH_BINARY, 11, 2)
    
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def box_area(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))

def box_center(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (sum(xs)/4, sum(ys)/4)

def box_bounds(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)

def extract_call_number(text):
    """Extract potential Library of Congress or Dewey Decimal call numbers."""
    text = str(text).strip()
    
    # Remove common noise words
    noise_words = ['the', 'of', 'and', 'in', 'to', 'a', 'is', 'for', 'by']
    words = text.split()
    filtered = [w for w in words if w.lower() not in noise_words]
    text = ' '.join(filtered) if filtered else text
    
    # LC pattern: 1-3 letters followed by digits
    lc_match = re.search(r'[A-Z]{1,3}\s*\d+(?:\.\d+)?(?:\s*\.?[A-Z]\d+)?', text, re.IGNORECASE)
    if lc_match:
        return lc_match.group(0).replace(' ', '').upper()
    
    # Dewey pattern: 3 digits possibly followed by decimal
    dewey_match = re.search(r'\d{3}(?:\.\d+)?', text)
    if dewey_match:
        return dewey_match.group(0)
    
    # Any alphanumeric that looks like a call number
    alphanum_match = re.search(r'[A-Z]+\d+(?:\.\d+)?', text, re.IGNORECASE)
    if alphanum_match:
        return alphanum_match.group(0).upper()
    
    # Simplified patterns for short codes
    short_match = re.search(r'[A-Z]\d+', text, re.IGNORECASE)
    if short_match:
        return short_match.group(0).upper()
    
    # If contains digits, might be a call number
    if re.search(r'\d', text):
        return text.strip().upper()
    
    return None

def cluster_detections_by_book(detections):
    """
    Group detections into book clusters based on X position.
    Multiple labels on same spine should be grouped together.
    """
    if not detections:
        return []
    
    x_clusters = []
    
    for box, text, conf in detections:
        cx, cy = box_center(box)
        x_min, y_min, x_max, y_max = box_bounds(box)
        width = x_max - x_min
        placed = False
        
        # Try to place in existing X cluster
        for cluster in x_clusters:
            cluster_cx = cluster["x_pos"]
            cluster_x_min = cluster["x_min"]
            cluster_x_max = cluster["x_max"]
            
            # Check if boxes overlap horizontally or are very close
            horizontal_overlap = not (x_max < cluster_x_min - 50 or x_min > cluster_x_max + 50)
            close_enough = abs(cx - cluster_cx) < 250
            
            if horizontal_overlap or close_enough:
                cluster["items"].append((box, text, conf, cx, cy))
                cluster["x_pos"] = sum(item[3] for item in cluster["items"]) / len(cluster["items"])
                cluster["x_min"] = min(cluster_x_min, x_min)
                cluster["x_max"] = max(cluster_x_max, x_max)
                cluster["all_boxes"].append(box)
                placed = True
                break
        
        if not placed:
            x_clusters.append({
                "x_pos": cx,
                "x_min": x_min,
                "x_max": x_max,
                "items": [(box, text, conf, cx, cy)],
                "all_boxes": [box]
            })
    
    # Sort clusters by X position (left to right)
    x_clusters.sort(key=lambda c: c["x_pos"])
    
    return x_clusters

def build_call_number_from_cluster(cluster_items):
    """
    Extract the best call number from a cluster of detected text.
    """
    # First, try to find the best call number candidate
    best_call_number = None
    best_confidence = 0
    
    for box, text, conf, cx, cy in cluster_items:
        call_num = extract_call_number(text)
        if call_num and conf > best_confidence:
            best_call_number = call_num
            best_confidence = conf
    
    if best_call_number:
        return best_call_number
    
    # Otherwise, combine text and try again
    sorted_items = sorted(cluster_items, key=lambda item: (item[4], item[3]))
    text_parts = [item[1] for item in sorted_items]
    combined_text = " ".join(text_parts)
    
    call_number = extract_call_number(combined_text)
    
    if not call_number:
        # Return the most likely call number text
        for box, text, conf, cx, cy in sorted(cluster_items, key=lambda x: x[2], reverse=True):
            if extract_call_number(text):
                return text.strip().upper()
        # Last resort: return highest confidence text
        longest = max(cluster_items, key=lambda item: item[2])
        return longest[1].strip().upper()
    
    return call_number

def main():
    # Load image
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file '{IMAGE_PATH}' not found!")
        print("Please place your iPhone image in the same directory as this script.")
        return
    
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Could not load image from '{IMAGE_PATH}'")
        return
    
    print(f"Loaded image: {IMAGE_PATH}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Crop ROI
    roi, (x0, y0, x1, y1) = crop_roi(img)
    print(f"ROI size: {roi.shape[1]}x{roi.shape[0]}")
    
    # Preprocess
    print("Preprocessing image for OCR...")
    roi_proc = preprocess_for_ocr(roi)
    
    # Run OCR
    print("Running OCR (this may take a moment)...")
    result = ocr.ocr(roi_proc, cls=True)
    
    # Process results
    new_results = []
    
    if result and len(result) > 0:
        lines = result[0] if isinstance(result[0], list) or (result[0] is None) else result
        if lines is None:
            lines = []
        
        print(f"\nFound {len(lines)} text detections")
        
        for line in lines:
            if line is None or len(line) < 2:
                continue
            
            box = line[0]
            txt = line[1]
            
            if box is None or txt is None:
                continue
            
            if not isinstance(txt, (list, tuple)) or len(txt) < 2:
                continue
            
            text, conf = txt[0], txt[1]
            
            if text is None or len(str(text).strip()) == 0:
                continue
            
            # Filtering
            area = box_area(box)
            roi_area = roi.shape[0] * roi.shape[1]
            
            if area < 0.0003 * roi_area:
                continue
            
            if area > 0.20 * roi_area:
                continue
            
            # Aspect ratio check
            x_min, y_min, x_max, y_max = box_bounds(box)
            width = x_max - x_min
            height = y_max - y_min
            
            # Skip overly vertical text
            if height > width * 2.0:
                continue
            
            new_results.append((box, str(text).strip(), float(conf) if conf is not None else 0.0))
            print(f"  - '{text}' (conf: {conf:.2f})")
    
    # Cluster by book
    print(f"\n{len(new_results)} detections after filtering")
    clusters = cluster_detections_by_book(new_results)
    print(f"Detected {len(clusters)} books\n")
    
    # Extract call numbers
    call_numbers = []
    for i, cluster in enumerate(clusters):
        call_num = build_call_number_from_cluster(cluster["items"])
        x_pos = cluster["x_pos"]
        conf_avg = sum(item[2] for item in cluster["items"]) / len(cluster["items"])
        call_numbers.append((call_num, x_pos, conf_avg))
        print(f"Book {i+1}: {call_num} (x={int(x_pos)}, avg_conf={conf_avg:.2f})")
        print(f"  Contains {len(cluster['items'])} text detections:")
        for box, text, conf, cx, cy in cluster["items"]:
            print(f"    - '{text}' (conf: {conf:.2f})")
    
    # Create visualization
    disp = img.copy()
    cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 3)
    
    # Draw clusters with different colors
    colors = [(255,0,0), (0,255,255), (255,0,255), (0,165,255), (255,255,0), (128,0,128)]
    
    for idx, cluster in enumerate(clusters):
        color = colors[idx % len(colors)]
        for box in cluster["all_boxes"]:
            pts = np.array([[int(p[0] + x0), int(p[1] + y0)] for p in box], dtype=np.int32)
            cv2.polylines(disp, [pts], isClosed=True, color=color, thickness=3)
    
    # Display call numbers on image
    y_offset = 60
    cv2.putText(disp, "Shelf Order (Left to Right):", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    for i, (call_num, x_pos, conf) in enumerate(call_numbers):
        y_offset += 45
        display_text = f"{i+1}. {call_num}"
        cv2.putText(disp, display_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
    
    cv2.putText(disp, f"Books Detected: {len(clusters)}", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    # Save and show results
    output_path = "result.jpg"
    cv2.imwrite(output_path, disp)
    print(f"\nResult saved to: {output_path}")
    
    # Display
    scale_percent = 30  # Resize for display
    width = int(disp.shape[1] * scale_percent / 100)
    height = int(disp.shape[0] * scale_percent / 100)
    resized = cv2.resize(disp, (width, height), interpolation=cv2.INTER_AREA)
    
    cv2.imshow("iPhone Image Analysis", resized)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SHELF ORDER (Left to Right)")
    print("="*60)
    for i, (call_num, x_pos, conf) in enumerate(call_numbers):
        print(f"{i+1}. {call_num}")
    print("="*60)

if __name__ == "__main__":
    main()