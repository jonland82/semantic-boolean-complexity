"""Run the fast exact-synthesis regression used by continuous integration."""

import importlib.util
from pathlib import Path


path = Path(__file__).resolve().parents[1] / "experiments" / "four_bit" / "scripts" / "run.py"
spec = importlib.util.spec_from_file_location("four_bit_run", path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.verify_three_bit()
print("three-input exact regression: PASS")
