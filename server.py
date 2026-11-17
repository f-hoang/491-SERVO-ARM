"""Run:  python server.py --source 0 --width 640   then open http://localhost:5000
Needs contours.py and arm.py in the same folder. pip install flask opencv-python numpy"""
import argparse
import json
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from arm import Arm
from contours import MODES, classify, preprocess

app = Flask(__name__, static_folder="static", static_url_path="")
arm = Arm()
S = {"mode": "otsu", "min_area": 800, "target": 0}
lock = threading.Lock()
latest = {"jpg": None, "objs": []}
LOG = []
CAL = {"pairs": [], "M": None, "err": None}  # pairs: [px, py, arm_x, arm_y]
CAL_FILE = "calibration.json"
DROP = (120.0, -120.0, 60.0)  # where picked objects get dropped, edit me
busy = False


def log(msg):
    LOG.append(f"{time.strftime('%H:%M:%S')} {msg}")
    del LOG[:-40]


def save_cal():
    json.dump(CAL, open(CAL_FILE, "w"))


if os.path.exists(CAL_FILE):
    CAL.update(json.load(open(CAL_FILE)))


def camera_loop(src, width):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log(f"could not open source {src}")
    while True:
        ok, frame = cap.read()
        if not ok:
            if isinstance(src, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video files
            time.sleep(0.2)
            continue
        s = width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=s, fy=s)
        cnts, _ = cv2.findContours(preprocess(frame, S["mode"]), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        objs = []
        for c in cnts:
            a, m = cv2.contourArea(c), cv2.moments(c)
            if a < S["min_area"] or m["m00"] == 0:
                continue
            objs.append((c, classify(c)[0], int(a), round(m["m10"] / m["m00"]), round(m["m01"] / m["m00"])))
        objs.sort(key=lambda o: o[3])  # left to right, so #ids stay stable
        for i, (c, shape, a, cx, cy) in enumerate(objs):
            col = (0, 140, 255) if i == S["target"] else (0, 255, 0)
            cv2.drawContours(frame, [c], -1, col, 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"#{i} {shape}", (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
        jpg = cv2.imencode(".jpg", frame)[1].tobytes()
        with lock:
            latest["jpg"] = jpg
            latest["objs"] = [dict(shape=sh, area=a, cx=cx, cy=cy) for _, sh, a, cx, cy in objs]
        if isinstance(src, str):
            time.sleep(0.03)


def pick_job(x, y):
    global busy
    busy, arm.stopped = True, False
    steps = [lambda: arm.move_to(x, y, 60), lambda: arm.set_gripper("open"),
             lambda: arm.move_to(x, y, 5), lambda: arm.set_gripper("closed"),
             lambda: arm.move_to(x, y, 60), lambda: arm.move_to(*DROP),
             lambda: arm.set_gripper("open")]
    try:
        for step in steps:
            if arm.stopped:
                log("pick aborted")
                return
            step()
        log("pick done")
    finally:
        busy = False


def err(msg, code=400):
    return jsonify(error=msg), code


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/video")
def video():
    def gen():
        while True:
            with lock:
                j = latest["jpg"]
            if j:
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.get("/api/state")
def state():
    with lock:
        objs = list(latest["objs"])
    return jsonify(settings=S, modes=MODES, objs=objs, arm=arm.state(), log=LOG[-12:], busy=busy,
                   cal={"pairs": CAL["pairs"], "solved": CAL["M"] is not None, "err": CAL["err"]})


@app.post("/api/settings")
def settings():
    d = request.get_json(silent=True) or {}
    if d.get("mode") in MODES:
        S["mode"] = d["mode"]
    if "min_area" in d:
        S["min_area"] = max(0, int(d["min_area"]))
    if "target" in d:
        S["target"] = int(d["target"])
    return jsonify(ok=True)


@app.post("/api/jog")
def jog():
    d = request.get_json(silent=True) or {}
    if d.get("axis") not in "xyz" or busy:
        return err("bad jog or arm busy")
    arm.jog(d["axis"], float(d.get("step", 1)))
    return jsonify(ok=True)


@app.post("/api/gripper")
def gripper():
    arm.set_gripper("closed" if arm.gripper == "open" else "open")
    return jsonify(ok=True)


@app.post("/api/home")
def home():
    log("homing arm")
    arm.home()
    return jsonify(ok=True)


@app.post("/api/stop")
def stop():
    arm.stop()
    log("STOP")
    return jsonify(ok=True)


@app.post("/api/calib/capture")
def cal_capture():
    with lock:
        objs = list(latest["objs"])
    if S["target"] >= len(objs):
        return err("select a detected object first")
    o, p = objs[S["target"]], arm.pos
    CAL["pairs"].append([o["cx"], o["cy"], p["x"], p["y"]])
    CAL["M"], CAL["err"] = None, None
    save_cal()
    log(f"captured P{len(CAL['pairs'])}: pixel ({o['cx']}, {o['cy']}) to arm ({p['x']:.1f}, {p['y']:.1f})")
    return jsonify(ok=True)


@app.post("/api/calib/solve")
def cal_solve():
    P = np.array(CAL["pairs"], dtype=float)
    if len(P) < 3:
        return err("need at least 3 points (5+ is better)")
    A = np.c_[P[:, :2], np.ones(len(P))]
    X = np.linalg.lstsq(A, P[:, 2:], rcond=None)[0]  # least squares affine fit
    CAL["M"] = X.T.tolist()
    CAL["err"] = round(float(np.mean(np.linalg.norm(A @ X - P[:, 2:], axis=1))), 2)
    save_cal()
    log(f"solved transform, reprojection error {CAL['err']} mm")
    return jsonify(ok=True)


@app.post("/api/calib/reset")
def cal_reset():
    CAL.update(pairs=[], M=None, err=None)
    save_cal()
    log("calibration reset")
    return jsonify(ok=True)


@app.post("/api/pick")
def pick():
    if busy:
        return err("arm is busy")
    if CAL["M"] is None:
        return err("calibrate first")
    with lock:
        objs = list(latest["objs"])
    if S["target"] >= len(objs):
        return err("select a detected object first")
    o = objs[S["target"]]
    x, y = (np.array(CAL["M"]) @ [o["cx"], o["cy"], 1]).tolist()
    log(f"picking #{S['target']} {o['shape']} at arm ({x:.1f}, {y:.1f})")
    threading.Thread(target=pick_job, args=(x, y), daemon=True).start()
    return jsonify(ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--width", type=int, default=640, help="keep fixed once calibrated")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from another device")
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    threading.Thread(target=camera_loop, args=(src, a.width), daemon=True).start()
    app.run(host=a.host, port=a.port, threaded=True)
