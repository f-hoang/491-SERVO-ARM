import argparse
import json
import os

import cv2
import numpy as np

from Services.contours import classify


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
            
            