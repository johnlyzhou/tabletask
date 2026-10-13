# DROID model

Adapted from EgoGym's `egogym/assets/embodiments/droid/model_pick.xml` at
commit `7203f25608987fdd2929219ff1bd1e7ece45e087`: https://github.com/omarrayyann/EgoGym.

Franka Panda and Robotiq 2F-85 geometry originate from MuJoCo Menagerie.
Their licenses are included alongside this file. EgoGym supplies the combined
assembly, home posture, and original wrist-camera calibration.

tabletask mounts the arm at table height, places `tcp` between the fingertips,
centers the wrist view on `tcp`, and adds an external camera facing the table.
Wrist actuator gains (joints 5–7) use kp=600 and kv=50 to avoid saturated
oscillation at a 2 ms timestep. Only referenced meshes are included; EgoGym is
not a runtime dependency.
