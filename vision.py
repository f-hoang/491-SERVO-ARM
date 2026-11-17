import argparse
import json
import os

import cv2
import numpy as np

from Services.contours import classify

COLORS_FILE = "colors.json"
CALIB_FILE = "calibration.json"

# Colors for object detection using HSV
DEFAULT_COLORS = {
    "red": {"lo":[170,120,70], "hi": [10, 255, 255]},
    "blue": {"lo": [100, 120, 50], "hi": [130, 255, 255]},
    "yellow": {"lo": [20, 100, 100], "hi": [35, 255, 255]},
}

# Colors for adjusting later using BGR
DRAW_BGR = {
    "red": (0, 0, 255),
    "blue": (255, 80, 0),
    "yellow": (0, 220, 255),
}

# Sliders for adjusting the color ranges. 
TRACKBARS = ["H lo", "H hi", "S, lo", "S, hi", "V, lo", "V, hi"]


def color_mask(hsv, rng):
    """White where a pixel falls inside this color's HSV range."""
    h1, s1, v1 = rng["lo"]
    h2, s2, v2 = rng["hi"]
    
    if h1 <= h2: 
        return cv2.inRange(hsv, np.array([h1, s1, v1]), np.array([h2, s2, v2]))
    
    # Wrapping hue range 
    upper = cv2.inRange(hsv, np.array([h1, s1, v1]), np.array([179, s2, v2]))
    lower = cv2.inRange(hsv, np.array([0, s1, v1]), np.array([h2, s2, v2]))
    return cv2.bitwise_or(upper, lower)

def clean(mask):
    """Remove speckle noise and fill small holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def detect(frame, colors, min_area=800, homography=None):
    """ 

    Find colored objects in BGR frame. 
    
    Returns a list of dicts, largest first: 
    {"color": "red", "shape": "rectangle", "area": 1520,
    "px": (cx,cy), "mm": (x,y) or None, "bbox": (x,y,w,h), "contour": ...}
    
    This list is the hand-off to the arm code.
    """
    
    blurred = cv2.GaussianBlur(frame, (5,5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    found = []
    
    for name, rng in colors.items():
        mask = clean(color_mask(hsv, rng))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours: 
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx,cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            shape, _ = classify(c)
            
            found.append({
                "color" : name, 
                "shape": shape, 
                "area": int(area),
                "px": (int(cx), int(cy)),
                "mm": px_to_mm((cx,cy), homography) if homography is not None else None,
                "bbox": cv2.boundingRect(c),
                "contour": c,
            })
    found.sort(key=lambda d: d["area"], reverse = True)
    return found

def draw(frame, detections):
    for d in detections:
        bgr = DRAW_BGR.get(d["color"], (0, 255, 0))
        x,y,w,h = d["bbox"]
        cv2.drawContours(frame, [d["contour"]], -1, bgr, 2)
        cv2.circle(frame, d["px"], 4, (255, 255, 255), -1)

        label = f'{d["color"]} {d["shape"]}'
        if d["mm"] is not None: 
            label += f' ({d["mm"][0]:.0f}, {d["mm"][1]:.0f}) mm'

        cv2.putText(frame, label, (x, max(y -6, 12)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (x, max(y-6, 12)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

        
def make_homography(corners_px, width_mm, height_mm):
    """Pixel corners (TL, TR, BR, BL) -> matrix that maps to pixel on table."""
    src = np.array(corners_px, dtype=np.float32)
    dst = np.array([[0,0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)

def px_to_mm(pt, homography):
    p = np.array([[pt]], dtype=np.float32)
    x, y = cv2.perspectiveTransform(p, homography)[0][0]
    return (float(x), float(y))

""" Fucntions that create a save file for the settings. """
def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")

def open_tuner(rng):
    cv2.namedWindow("tune")
    for i, name in enumerate(TRACKBARS):
        cv2.createTrackbar(name, "tune", 0, 179 if i < 2 else 255, lambda v: None)
    set_tuner(rng)

def set_tuner(rng):
    vals = [rng["lo"][0], rng["hi"][0], rng["lo"][1], rng["hi"][1], rng["lo"][2], rng["hi"][2]]
    for name, v in zip(TRACKBARS, vals):
        cv2.setTrackbarPos(name, "tune", int(v))

def read_tuner(): 
    h1, h2, s1, s2, v1, v2 = (cv2.getTrackbarPos(n, "tune") for n in TRACKBARS)
    return {"lo": [h1, s1, v1], "hi": [h2, s2, v2]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video file path")
    ap.add_argument("--min-area", type=int, default=800, help="ignore blobs smaller than this (pixels)")
    ap.add_argument("--width", type=int, default=640, help="resize frames to this width")
    ap.add_argument("--work-w", type=float, default=279.4, help="workspace width in mm (default: letter paper size)")
    ap.add_argument("--work-h", type=float, default=215.9, help="workspace height in mm")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera/source: {args.source} " 
                         "(on a Mac, allow camera access for your terminal in System Settings)")

    colors = load_json(COLORS_FILE, DEFAULT_COLORS)
    calib = load_json(CALIB_FILE, None)
    H = None
    if calib and calib.get("frame_width") == args.width:
        H = make_homography(calib["corners_px"], calib["width_mm"], calib["height_mm"])
        print("loaded calibration form", CALIB_FILE)

    clicks = []
    calibrating = False

    def on_click(event, x, y, flags, param):
        if calibrating and event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append([x,y])

    cv2.namedWindow("vision")
    cv2.setMouseCallback("vision", on_click)

    tuning = False
    tune_names = list(colors.keys())
    tune_idx = 0
    paused = False
    frame = None
    detections = []

    while True:
        if not paused: 
            ok, frame = cap.read()
            if not ok:
                break
            scale = args.width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        if tuning:
            colors[tune_names[tune_idx]] = read_tuner()

        view = frame.copy()

        if calibrating: 
            for i, (x,y) in enumerate(clicks):
                cv2.circle(view, (x,y), 6, (0, 255, 0), -1)
                cv2.putText(view, str(i + 1), (x + 8, y - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            order = ["top-left", "top-right", "bottom-right", "bottom-left"]
            msg = f"CALIBRATE: click {order[len(clicks)]} corner" if len(clicks) < 4 else ""
            cv2.putText(view, msg, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if len(clicks) == 4:
                H = make_homography(clicks, args.work_w, args.work_h)
                save_json(CALIB_FILE, {"corners_px": clicks, "width_mm": args.work_w, "height_mm": args.work_h, "frame_width": args.width})
                calib = {"corners_px": list(clicks)}
                calibrating = False

        else: 
            detections = detect(frame, colors, args.min_area, H)
            draw(view, detections)

            if calib: 
                pts = np.array(calib["corners_px"], dtype= np.int32)
                cv2.polylines(view, [pts], True, (0, 255, 0), 1)

            status = f"objects={len(detections)} calibrated={'yes' if H is not None else 'no (press k)'}"
            if tuning:
                status += f" tuning={tune_names[tune_idx]} (n=next, s=save)"
            cv2.putText(view, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow("vision", view)
        if tuning:
            hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
            cv2.imshow("mask", clean(color_mask(hsv, colors[tune_names[tune_idx]])))

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused

        elif key == ord("p"):
            if not detections:
                print("no objects detected")
            for d in detections:
                where = f'{d["mm"][0]:.1f}, {d["mm"][1]:.1f} mm' if d["mm"] else f'px {d["px"]}'
                print(f'{d["color"]:>6} {d["shape"]:<15} at {where}')
            print("-" * 40)

        elif key == ord("t"):
            tuning = not tuning 
            if tuning:
                open_tuner(colors[tune_names[tune_idx]])
            else: 
                cv2.destroyWindow("tune")
                cv2.destroyWindow("mask")

        elif key == ord("n") and tuning:
            tune_idx = (tune_idx + 1) % len(tune_names)
            set_tuner(colors[tune_names[tune_idx]])

        elif key == ord("s"):
            save_json(COLORS_FILE, colors)

        elif key == ord("k"):
            calibrating = True
            clicks.clear()
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
