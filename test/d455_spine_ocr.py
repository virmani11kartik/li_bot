import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from paddleocr import PaddleOCR
import re

COLOR_W, COLOR_H, FPS = 1280, 720, 30

ROI_X0, ROI_Y0 = 0.15, 0.46
ROI_X1, ROI_Y1 = 0.72, 0.72

OCR_PERIOD_S = 0.50

# X-position threshold for grouping text on same label
LABEL_X_THRESHOLD = 60  # pixels - text on same horizontal label
# Y-position threshold for separating different books
BOOK_Y_THRESHOLD = 100  # pixels - different books vertically

# Store call numbers with their positions
call_number_registry = {}  # {y_position: {"call_number": str, "timestamp": float}}

ocr = PaddleOCR(use_angle_cls=True, lang='en')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS)

profile = pipeline.start(config)

for _ in range(15):
    pipeline.wait_for_frames()

last_ocr_t = 0.0
cached_results = []

t0 = time.time()
frame_count = 0
disp_fps = 0

def crop_roi(img):
    h,w = img.shape[:2]
    x0=int(w*ROI_X0)
    y0=int(h*ROI_Y0)
    x1=int(w*ROI_X1)
    y1=int(h*ROI_Y1)
    roi=img[y0:y1, x0:x1].copy()
    return roi, (x0, y0, x1, y1)

def preprocess_for_ocr(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def rotate_if_needed(img):
    return img, None

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
    """
    Extract potential Library of Congress or Dewey Decimal call numbers.
    """
    text = str(text).strip()
    
    # Remove common noise words
    noise_words = ['the', 'of', 'and', 'in', 'to', 'a', 'is', 'for']
    words = text.split()
    filtered = [w for w in words if w.lower() not in noise_words]
    text = ' '.join(filtered) if filtered else text
    
    # LC pattern: 1-3 letters followed by digits
    lc_match = re.search(r'[A-Z]{1,3}\s*\d+(?:\.\d+)?(?:\s*[A-Z]\d+)?', text)
    if lc_match:
        return lc_match.group(0).replace(' ', '')
    
    # Dewey pattern: 3 digits possibly followed by decimal
    dewey_match = re.search(r'\d{3}(?:\.\d+)?', text)
    if dewey_match:
        return dewey_match.group(0)
    
    # Any alphanumeric that looks like a call number
    if re.search(r'[A-Z]+\d+', text):
        match = re.search(r'[A-Z]+\d+(?:\.\d+)?', text)
        if match:
            return match.group(0)
    
    # If contains digits, might be a call number
    if re.search(r'\d', text):
        return text.strip()
    
    return None

def cluster_detections_by_book(detections):
    """
    Group detections into book clusters.
    Books are arranged left-to-right, so we cluster by X position (horizontal position).
    Within each book, the label text runs horizontally (left to right on the label).
    Multiple labels on same spine should be grouped together.
    """
    if not detections:
        return []
    
    # First pass: group by approximate X position (left-to-right books)
    # Use a more generous threshold since labels on same book spine can vary in X position
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
            # This groups multiple stickers on the same vertical book spine
            horizontal_overlap = not (x_max < cluster_x_min - 30 or x_min > cluster_x_max + 30)
            close_enough = abs(cx - cluster_cx) < 200
            
            if horizontal_overlap or close_enough:
                cluster["items"].append((box, text, conf, cx, cy))
                # Update cluster bounds
                cluster["x_pos"] = sum(item[3] for item in cluster["items"]) / len(cluster["items"])
                cluster["x_min"] = min(cluster_x_min, x_min)
                cluster["x_max"] = max(cluster_x_max, x_max)
                placed = True
                break
        
        if not placed:
            x_clusters.append({
                "x_pos": cx,
                "x_min": x_min,
                "x_max": x_max,
                "items": [(box, text, conf, cx, cy)]
            })
    
    # Sort clusters by X position (left to right)
    x_clusters.sort(key=lambda c: c["x_pos"])
    
    return x_clusters

def build_call_number_from_cluster(cluster_items):
    """
    Combine text items from a cluster to form complete call number.
    Since a book spine may have multiple stacked labels (vertically stacked),
    we prioritize the most call-number-like text.
    """
    # First, try to find the best call number candidate from individual items
    best_call_number = None
    best_confidence = 0
    
    for box, text, conf, cx, cy in cluster_items:
        call_num = extract_call_number(text)
        if call_num and conf > best_confidence:
            best_call_number = call_num
            best_confidence = conf
    
    # If we found a good call number from a single label, use it
    if best_call_number:
        return best_call_number
    
    # Otherwise, combine all text sorted by Y position (top to bottom)
    # then by X position (left to right)
    sorted_items = sorted(cluster_items, key=lambda item: (item[4], item[3]))
    
    # Combine text
    text_parts = [item[1] for item in sorted_items]
    combined_text = " ".join(text_parts)
    
    # Try to extract call number from combined text
    call_number = extract_call_number(combined_text)
    
    # If no standard call number found, return the first meaningful text
    if not call_number:
        # Return the longest text item as it's likely the call number
        longest = max(cluster_items, key=lambda item: len(item[1]))
        return longest[1].strip()
    
    return call_number

def update_call_number_registry(clusters, current_time):
    """
    Update the global registry with detected call numbers and their positions.
    """
    global call_number_registry
    
    for cluster in clusters:
        x_pos = cluster["x_pos"]
        call_number = build_call_number_from_cluster(cluster["items"])
        
        if call_number and len(call_number) > 0:
            # Use X position as key (rounded for stability)
            key = round(x_pos / 20) * 20
            
            call_number_registry[key] = {
                "call_number": call_number,
                "timestamp": current_time,
                "x_pos": x_pos,
                "confidence": sum(item[2] for item in cluster["items"]) / len(cluster["items"])
            }
    
    # Clean up old entries (older than 5 seconds)
    keys_to_remove = [k for k, v in call_number_registry.items() 
                      if current_time - v["timestamp"] > 5.0]
    for k in keys_to_remove:
        del call_number_registry[k]

def get_sorted_call_numbers():
    """
    Return call numbers sorted left to right (increasing X position).
    """
    sorted_items = sorted(call_number_registry.items(), key=lambda x: x[1]["x_pos"])
    return [(item[1]["call_number"], item[1]["x_pos"], item[1]["confidence"]) 
            for item in sorted_items]

def check_shelf_order(call_numbers):
    """
    Check if call numbers are in proper library order.
    Returns (is_correct, issues)
    """
    if len(call_numbers) < 2:
        return True, []
    
    issues = []
    for i in range(len(call_numbers) - 1):
        current = call_numbers[i]
        next_num = call_numbers[i + 1]
        
        # Simple alphabetical/numerical comparison
        if current > next_num:
            issues.append(f"Position {i+1} ({current}) should come after position {i+2} ({next_num})")
    
    return len(issues) == 0, issues

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        roi, (x0,y0,x1,y1) = crop_roi(img)
        now = time.time()
        
        if (now - last_ocr_t) >= OCR_PERIOD_S:
            roi_proc = preprocess_for_ocr(roi)
            roi_rot, rot_flag = rotate_if_needed(roi_proc)

            result = ocr.ocr(roi_rot, cls=True)
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

                    # Filtering - adjust for horizontal call number labels
                    area = box_area(box)
                    roi_area = roi.shape[0] * roi.shape[1]
                    
                    # Filter out very small noise
                    if area < 0.0005 * roi_area:
                        continue
                    
                    # Filter out very large regions (likely book titles)
                    if area > 0.15 * roi_area:
                        continue
                    
                    # Check aspect ratio - call number labels are usually wider than tall
                    x_min, y_min, x_max, y_max = box_bounds(box)
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Skip if too vertical (likely spine text, not call number label)
                    if height > width * 1.5:
                        continue

                    new_results.append((box, str(text).strip(), float(conf) if conf is not None else 0.0))

            # Cluster by book (X position) and update registry
            clusters = cluster_detections_by_book(new_results)
            update_call_number_registry(clusters, now)
            
            cached_results = new_results
            last_ocr_t = now

        # Visualization
        disp = img.copy()
        cv2.rectangle(disp, (x0,y0), (x1,y1), (0,255,0), 2)
        
        # Draw detected text boxes with different colors for each cluster
        clusters = cluster_detections_by_book(cached_results)
        colors = [(255,0,0), (0,255,255), (255,0,255), (0,165,255), (255,255,0), (128,0,128)]
        
        for idx, cluster in enumerate(clusters):
            color = colors[idx % len(colors)]
            for box, text, conf, cx, cy in cluster["items"]:
                pts = np.array([[int(p[0] + x0), int(p[1] + y0)] for p in box], dtype=np.int32)
                cv2.polylines(disp, [pts], isClosed=True, color=color, thickness=2)
        
        # Display sorted call numbers
        sorted_calls = get_sorted_call_numbers()
        y_offset = 60
        cv2.putText(disp, "Shelf Order (Left to Right):", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        call_nums_only = [cn for cn, _, _ in sorted_calls]
        is_correct, issues = check_shelf_order(call_nums_only)
        
        for i, (call_num, x_pos, conf) in enumerate(sorted_calls):
            y_offset += 30
            display_text = f"{i+1}. {call_num}"
            color = (0, 255, 0) if is_correct else (0, 165, 255)
            cv2.putText(disp, display_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show order status
        if len(sorted_calls) > 1:
            y_offset += 40
            status = "✓ CORRECT ORDER" if is_correct else "✗ ORDER ISSUES"
            status_color = (0, 255, 0) if is_correct else (0, 0, 255)
            cv2.putText(disp, status, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        frame_count += 1
        if (now - t0) >= 1.0:
            disp_fps = frame_count / (now - t0)
            t0 = now
            frame_count = 0
        
        cv2.putText(disp, f"FPS: {disp_fps:.1f} | Books: {len(sorted_calls)}",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("Book Shelf Reading (Left to Right)", disp)
        
        # Print to console periodically
        if sorted_calls and frame_count % 60 == 0:
            print("\n=== Current Shelf Order (Left to Right) ===")
            for i, (call_num, x_pos, conf) in enumerate(sorted_calls):
                print(f"{i+1}. {call_num} (x={int(x_pos)}, conf={conf:.2f})")
            if not is_correct:
                print("\nISSUES DETECTED:")
                for issue in issues:
                    print(f"  - {issue}")
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):  # Press 's' to save current order
            print("\n" + "="*50)
            print("SAVED SHELF ORDER (Left to Right)")
            print("="*50)
            for i, (call_num, x_pos, conf) in enumerate(sorted_calls):
                print(f"{i+1}. {call_num}")
            if not is_correct:
                print("\n⚠ ORDER ISSUES:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("\n✓ Books are in correct order")
            print("="*50 + "\n")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    
    # Final output
    print("\n" + "="*50)
    print("FINAL SHELF ORDER (Left to Right)")
    print("="*50)
    sorted_calls = get_sorted_call_numbers()
    for i, (call_num, x_pos, conf) in enumerate(sorted_calls):
        print(f"{i+1}. {call_num}")
    
    call_nums_only = [cn for cn, _, _ in sorted_calls]
    is_correct, issues = check_shelf_order(call_nums_only)
    if not is_correct:
        print("\n⚠ ORDER ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ Books are in correct order")
    print("="*50)