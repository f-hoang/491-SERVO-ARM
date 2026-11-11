import argparse
import cv2
import numpy as np
 
MODES = ["otsu", "adaptive", "canny"]
 
 
def preprocess(frame, mode):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
 
    if mode == "otsu":
        # INV so dark objects on a light background become white blobs
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif mode == "adaptive":
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 5)
    else:  # canny
        mask = cv2.Canny(gray, 50, 150)
        mask = cv2.dilate(mask, None, iterations=2)
 
    # Clean up noise and close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
 
 
def classify(contour):
    """Rough shape name from polygon approximation."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    n = len(approx)
    if n == 3:
        return "triangle", approx
    if n == 4:
        x, y, w, h = cv2.boundingRect(approx)
        return ("square" if 0.9 <= w / float(h) <= 1.1 else "rectangle"), approx
    if n > 6:
        return "circle/ellipse", approx
    return f"{n}-gon", approx
 
 
def draw_objects(frame, mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        count += 1
 
        shape, approx = classify(c)
        x, y, w, h = cv2.boundingRect(c)
 
        # Exact outline, simplified polygon, and bounding box
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [approx], -1, (0, 200, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 128, 0), 1)
 
        # Centroid from image moments
        m = cv2.moments(c)
        if m["m00"] > 0:
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
 
        cv2.putText(frame, f"{shape} {int(area)}", (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return count
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video file path")
    ap.add_argument("--mode", choices=MODES, default="otsu")
    ap.add_argument("--min-area", type=int, default=800, help="ignore contours smaller than this (pixels)")
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()
 
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
 
    mode = args.mode
    paused = False
    frame = None
 
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break  # end of file or camera lost
            scale = args.width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
 
        view = frame.copy()
        mask = preprocess(frame, mode)
        n = draw_objects(view, mask, args.min_area)
 
        cv2.putText(view, f"mode={mode} objects={n}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("contours", view)
        cv2.imshow("mask", mask)
 
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("m"):
            mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
        elif key == ord(" "):
            paused = not paused
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 