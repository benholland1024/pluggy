from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).parent.parent / "models" / "world.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# settle so the robot is resting on its wheels, not mid-spawn-drop
for _ in range(int(2.0 / model.opt.timestep)):
  mujoco.mj_step(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)

def snap(camera_name):
    renderer.update_scene(data, camera=camera_name)
    return renderer.render()  # (480, 640, 3) uint8 — an actual image

left = snap("left_eye")
right = snap("right_eye")
Image.fromarray(np.hstack([left, right])).save("stereo.png")
print("Wrote stereo.png: ", left.shape, left.dtype)