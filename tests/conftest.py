from pathlib import Path
import mujoco
import pytest


MODEL_PATH = Path(__file__).parent.parent / "models" / "world.xml"


@pytest.fixture(scope="module")
def model():
  return mujoco.MjModel.from_xml_path(str(MODEL_PATH))

@pytest.fixture
def data(model):
  return mujoco.MjData(model)