"""Test package configuration."""

import os
import sys

# Legacy regression tests assume CPU/x64 behavior for strict lenstronomy parity.
# Keep suite defaults CPU-stable on macOS; dedicated parity tests probe Metal.
if sys.platform == "darwin":
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
