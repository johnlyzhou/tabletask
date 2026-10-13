"""Pose and button input from an iOS device running the Teledex app."""

import time

import numpy as np
from scipy.spatial.transform import Rotation


class Phone:
    """An iOS device running Teledex, reporting its pose and button states.

    Poses are ARKit camera poses converted to z-up, relative to the calibration pose;
    forward is the direction the camera faced when its AR session began.

    Args:
        scale: Metres of gripper motion per metre of phone motion.
        port: Port the phone connects to.
        show_qr: Print a QR code for the app to scan.
    """

    def __init__(self, scale: float = 2.0, port: int = 8888, show_qr: bool = True):
        try:
            import teledex
        except ImportError as error:
            raise ImportError("Phone teleoperation needs Teledex: pip install teledex") from error

        self.session = teledex.Session(port=port, show_qr=show_qr)
        self.scale = scale
        self.anchor = None
        self.origin = None
        # The app sends both buttons but Teledex keeps only A, so read B from each raw message.
        self._restart = False
        process_data = self.session._process_data

        def process_both_buttons(data):
            self._restart = bool(data.get("button_secondary", False))
            return process_data(data)

        self.session._process_data = process_both_buttons

    def start(self) -> None:
        """Serve until the phone connects and sends its first pose."""
        self.session.start()
        while self.session.get_latest_data()["position"] is None:
            time.sleep(0.01)

    def anchor_to(self, tcp_pose) -> None:
        """Map the current phone pose to the gripper pose `tcp_pose`, so motion starts from rest."""
        data = self.session.get_latest_data()
        self.anchor = np.array(data["position"]), np.array(data["rotation"]).reshape(3, 3)
        tcp_pose = np.asarray(tcp_pose)
        self.origin = tcp_pose[:3], Rotation.from_quat(tcp_pose[[4, 5, 6, 3]]).as_matrix()

    def target(self):
        """Target gripper pose (x, y, z, qw, qx, qy, qz) followed by closure (0 open, 1 closed)."""
        if self.anchor is None:
            raise RuntimeError("call anchor_to() with the gripper pose before reading a target")
        data = self.session.get_latest_data()
        position, rotation = np.array(data["position"]), np.array(data["rotation"]).reshape(3, 3)
        shift = (position - self.anchor[0]) * self.scale
        relative_rotation = rotation @ self.anchor[1].T
        quaternion = Rotation.from_matrix(relative_rotation @ self.origin[1]).as_quat()
        quaternion = quaternion[[3, 0, 1, 2]]  # scipy's xyzw to wxyz
        return np.r_[self.origin[0] + shift, quaternion, float(bool(data["toggle"]))]

    @property
    def restarting(self) -> bool:
        """Whether button B is down (drop the episode and draw the next start)."""
        return self._restart

    @property
    def clutching(self) -> bool:
        """Whether button A, the clutch, is held: the robot stays still while the phone moves."""
        return bool(self.session.get_latest_data()["button"])

    def buzz(self) -> None:
        """Vibrate the phone briefly."""
        self.session.vibrate(duration=0.05, intensity=0.6, sharpness=0.8)

    def stop(self) -> None:
        """Stop the Teledex server."""
        self.session.stop()
