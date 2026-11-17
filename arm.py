import math
import time


class Arm:
    """Simulated arm. Replace the method bodies with calls into your real servo
    code (serial, socket, ctypes into the C++ module). Keep the method names and
    the server and UI keep working."""

    LIMITS = {"x": (50, 250), "y": (-150, 150), "z": (0, 100)}  # mm, edit to fit your arm

    def __init__(self):
        self.pos = {"x": 150.0, "y": 0.0, "z": 60.0}
        self.gripper = "open"
        self.stopped = False

    def _clamp(self, k, v):
        lo, hi = self.LIMITS[k]
        return min(hi, max(lo, v))

    def jog(self, axis, step):
        self.pos[axis] = self._clamp(axis, self.pos[axis] + step)

    def move_to(self, x, y, z):
        for k, v in zip("xyz", (x, y, z)):
            self.pos[k] = self._clamp(k, v)
        time.sleep(0.5)  # stand-in for travel time

    def set_gripper(self, state):
        self.gripper = state
        time.sleep(0.3)

    def home(self):
        self.move_to(150, 0, 60)
        self.set_gripper("open")

    def stop(self):
        self.stopped = True

    def state(self):
        x, y, z = self.pos["x"], self.pos["y"], self.pos["z"]
        # Placeholder kinematics so the UI has joint angles to show.
        return {**self.pos, "gripper": self.gripper, "sim": True,
                "j1": round(90 + math.degrees(math.atan2(y, x))),
                "j2": round(math.degrees(math.atan2(z, math.hypot(x, y))))}
