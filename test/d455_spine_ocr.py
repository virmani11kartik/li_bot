import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from paddleocr import PaddleOCR

COLOR_W, COLOR_H, FPS = 1280, 720, 30

ROI_X0, ROI_Y0 = 0.20, 0.46
ROI_X1, ROI_Y1 = 0.70, 0.72

OCR_PERIOD_S = 0.50

TOP_K_BOXES = 8

ROTATE_90 = False

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

    # Boost local contrast
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

    # Mild sharpening
    blur1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    blur2 = cv2.GaussianBlur(gray, (0, 0), 2.0)
    sharp = cv2.addWeighted(blur1, 1.6, blur2, -0.6, 0)

    # Adaptive threshold -> crisp text on label
    bw = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7
    )

    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

def rotate_if_needed(img):
    if not ROTATE_90:
        return img, None
    rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return rot, "cw90"

def unrotate_box(box, rot_shape, orig_shape):
    Ho, Wo = orig_shape
    converted = []
    for (xp, yp) in box:
        x = yp
        y = Ho - 1 - xp
        converted.append([float(x), float(y)])
    return converted

def box_area(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        roi, (x0,y0,x1,y1)=crop_roi(img)
        now = time.time()
        if (now - last_ocr_t) >= OCR_PERIOD_S:
            roi_proc = preprocess_for_ocr(roi)
            roi_rot, rot_flag = rotate_if_needed(roi_proc)

            result = ocr.ocr(roi_rot, cls=True)
            print("RAW OCR RESULT:", type(result), "len:", 0 if result is None else len(result))
            new_results = []

            # ---- Robust handling: PaddleOCR may return [] or [None] when nothing is found ----
            if result and len(result) > 0:
                # PaddleOCR returns [lines] where lines can be None if no detections
                lines = result[0] if isinstance(result[0], list) or (result[0] is None) else result
                if lines is None:
                    lines = []

                for line in lines:
                    # line should look like: [box, (text, conf)]
                    if line is None or len(line) < 2:
                        continue

                    box = line[0]
                    txt = line[1]

                    if box is None or txt is None:
                        continue

                    # txt should look like (text, conf)
                    if not isinstance(txt, (list, tuple)) or len(txt) < 2:
                        continue

                    text, conf = txt[0], txt[1]

                    if text is None or len(str(text).strip()) == 0:
                        continue

                    # If rotated, convert box back to original ROI coords
                    if rot_flag == "cw90":
                        Hr, Wr = roi_rot.shape[:2]
                        Ho, Wo = roi.shape[:2]
                        box = unrotate_box(box, (Hr, Wr), (Ho, Wo))

                    # --- Sticker-like box filtering ---
                    area = box_area(box)
                    roi_area = roi.shape[0] * roi.shape[1]

                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)

                    # reject too tiny (noise)
                    if area < 0.002 * roi_area:
                        continue

                    # reject too big (title regions / large patches)
                    if area > 0.08 * roi_area:
                        continue

                    # sticker text boxes tend to be wider than tall (horizontal)
                    if w < 1.2 * h:
                        continue

                    new_results.append((box, (text, float(conf) if conf is not None else 0.0)))

                # Optional: sort by box area (largest first)
                new_results.sort(key=lambda t: box_area(t[0]), reverse=True)
                # Optional: keep top K
                # new_results = new_results[:TOP_K_BOXES]

            # Even if nothing detected, just cache empty list (no crash)
            cached_results = new_results
            last_ocr_t = now

        disp = img.copy()
        cv2.rectangle(disp, (x0,y0), (x1,y1), (0,255,0), 2)
        for box, (text, conf) in cached_results:
            pts = np.array([[int(p[0] + x0), int(p[1] + y0)] for p in box], dtype=np.int32)
            cv2.polylines(disp, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
            tx, ty = pts[0][0], pts[0][1]
            label = f"{text} ({conf:.2f})"
            cv2.putText(disp, label, (tx, max(ty - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            
        frame_count += 1
        if (now - t0) >= 1.0:
            disp_fps = frame_count / (now - t0)
            t0 = now
            frame_count = 0
        cv2.putText(disp, f"Display FPS: {disp_fps:.1f} | OCR every {OCR_PERIOD_S:.2f}s",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("D455 Realtime Book-Spine OCR (ROI + cached)", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()