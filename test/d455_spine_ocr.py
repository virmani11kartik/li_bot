import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from paddleocr import PaddleOCR
import re

# RealSense Configuration
COLOR_W, COLOR_H, FPS = 1280, 720, 30

# ROI Configuration
ROI_X0, ROI_Y0 = 0.10, 0.48
ROI_X1, ROI_Y1 = 0.70, 0.80

# OCR Configuration
OCR_PERIOD_S = 0.50  # Run OCR every 0.5 seconds

# Store call numbers with their positions
call_number_registry = {}

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)

profile = pipeline.start(config)

# Warm up camera
for _ in range(15):
    pipeline.wait_for_frames()

last_ocr_t = 0.0
cached_results = []
cached_clusters = []
cached_call_numbers = []

t0 = time.time()
frame_count = 0
disp_fps = 0

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
    Build complete call number by reading all stacked labels top to bottom.
    Example: MATH PHYS QC 794.6 S85 B58 2013
    """
    # Sort by Y position (top to bottom), then X position (left to right)
    sorted_items = sorted(cluster_items, key=lambda item: (item[4], item[3]))
    
    # Combine all text parts in order
    text_parts = []
    for box, text, conf, cx, cy in sorted_items:
        cleaned = text.strip().upper()
        if cleaned and len(cleaned) > 0:
            text_parts.append(cleaned)
    
    # Join all parts with spaces
    full_call_number = " ".join(text_parts)
    
    return full_call_number if full_call_number else "UNKNOWN"

try:
    print("Starting RealSense D455 Book Shelf Reading...")
    print("Press 'q' or ESC to quit")
    print("Press 's' to save current shelf order")
    print("Press 'c' to capture current frame as image")
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        roi, (x0, y0, x1, y1) = crop_roi(img)
        now = time.time()
        
        # Run OCR periodically
        if (now - last_ocr_t) >= OCR_PERIOD_S:
            print(f"\n[{time.strftime('%H:%M:%S')}] Running OCR...")
            
            # Preprocess
            roi_proc = preprocess_for_ocr(roi)
            
            # Run OCR
            result = ocr.ocr(roi_proc, cls=True)
            
            # Process results
            new_results = []
            
            if result and len(result) > 0:
                lines = result[0] if isinstance(result[0], list) or (result[0] is None) else result
                if lines is None:
                    lines = []
                
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
            
            # Cluster by book
            clusters = cluster_detections_by_book(new_results)
            
            # Extract call numbers
            call_numbers = []
            for i, cluster in enumerate(clusters):
                call_num = build_call_number_from_cluster(cluster["items"])
                x_pos = cluster["x_pos"]
                conf_avg = sum(item[2] for item in cluster["items"]) / len(cluster["items"])
                call_numbers.append((call_num, x_pos, conf_avg))
            
            # Cache results
            cached_results = new_results
            cached_clusters = clusters
            cached_call_numbers = call_numbers
            last_ocr_t = now
            
            # Print to console
            print(f"Detected {len(clusters)} books:")
            for i, (call_num, x_pos, conf) in enumerate(call_numbers):
                print(f"  {i+1}. {call_num}")

        # Create visualization
        disp = img.copy()
        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 3)
        
        # Draw clusters with different colors
        colors = [(255,0,0), (0,255,255), (255,0,255), (0,165,255), (255,255,0), (128,0,128)]
        
        for idx, cluster in enumerate(cached_clusters):
            color = colors[idx % len(colors)]
            for box in cluster["all_boxes"]:
                pts = np.array([[int(p[0] + x0), int(p[1] + y0)] for p in box], dtype=np.int32)
                cv2.polylines(disp, [pts], isClosed=True, color=color, thickness=3)
        
        # Display call numbers on image
        y_offset = 50
        cv2.putText(disp, "Shelf Order (Left to Right):", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        for i, (call_num, x_pos, conf) in enumerate(cached_call_numbers):
            y_offset += 35
            display_text = f"{i+1}. {call_num}"
            cv2.putText(disp, display_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Calculate and display FPS
        frame_count += 1
        if (now - t0) >= 1.0:
            disp_fps = frame_count / (now - t0)
            t0 = now
            frame_count = 0
        
        cv2.putText(disp, f"FPS: {disp_fps:.1f} | Books: {len(cached_clusters)}", 
                   (COLOR_W - 350, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("D455 Book Shelf Reading", disp)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q' to quit
            break
        elif key == ord('s'):  # 's' to save current order
            print("\n" + "="*60)
            print("SAVED SHELF ORDER (Left to Right)")
            print("="*60)
            for i, (call_num, x_pos, conf) in enumerate(cached_call_numbers):
                print(f"{i+1}. {call_num}")
            print("="*60 + "\n")
        elif key == ord('c'):  # 'c' to capture frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            capture_path = f"capture_{timestamp}.jpg"
            cv2.imwrite(capture_path, disp)
            print(f"\nFrame captured: {capture_path}\n")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    
    # Final output
    print("\n" + "="*60)
    print("FINAL SHELF ORDER (Left to Right)")
    print("="*60)
    for i, (call_num, x_pos, conf) in enumerate(cached_call_numbers):
        print(f"{i+1}. {call_num}")
    print("="*60)