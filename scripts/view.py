import mujoco
import mujoco.viewer

# Model is immutable (.xml files)
model = mujoco.MjModel.from_xml_path("models/world.xml")
# Data is mutable state -positions, velocities, forces, etc
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)