"""Guards for the outlet-dataset generator script.

The script lives in scripts/ (not the package), so it is loaded by path.
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent / "scripts" / "generate_outlet_dataset.py"


def _load_generator():
  spec = importlib.util.spec_from_file_location("generate_outlet_dataset", _SCRIPT)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def test_output_dirs_are_cleaned_of_stale_files(tmp_path):
  """The train/val split is a random draw per image, so regenerating into a
  dirty directory strands old files in whichever split changed: a buggy run's
  leftovers once contaminated 195 image/label pairs of fresh training data.
  A fresh dataset must mean a fresh directory."""
  gen = _load_generator()
  stale = [
    tmp_path / "images" / "train" / "img_00001.png",
    tmp_path / "labels" / "val" / "img_00001.txt",
  ]
  for f in stale:
    f.parent.mkdir(parents=True)
    f.write_text("stale")

  gen.clean_output_dirs(tmp_path)

  for f in stale:
    assert not f.exists(), f"stale file survived: {f}"
  for split in ("train", "val"):
    assert (tmp_path / "images" / split).is_dir()
    assert (tmp_path / "labels" / split).is_dir()
