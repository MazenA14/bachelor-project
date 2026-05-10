"""
Backward-compatible entry point: enriched CEEMDAN outputs are produced by the unified
pipeline (single CEEMDAN on Gold_Close + forked feature engineering).
Run `07_ceemdan_decomposition.py` — this wrapper forwards to it.
"""
import os
import runpy

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "07_ceemdan_decomposition.py")
    runpy.run_path(path, run_name="__main__")
