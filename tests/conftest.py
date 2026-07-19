from pathlib import Path
import mujoco
import pytest


WORLD_MODEL_PATH = Path(__file__).parent.parent / "models" / "world.xml"
PLAYGROUND_MODEL_PATH = Path(__file__).parent.parent / "models" / "playground.xml"


@pytest.fixture(scope="module")
def world_model():
  return mujoco.MjModel.from_xml_path(str(WORLD_MODEL_PATH))

@pytest.fixture
def world_data(world_model):
  return mujoco.MjData(world_model)

@pytest.fixture(scope="module")
def playground_model():
  return mujoco.MjModel.from_xml_path(str(PLAYGROUND_MODEL_PATH))

@pytest.fixture
def playground_data(playground_model):
  return mujoco.MjData(playground_model)