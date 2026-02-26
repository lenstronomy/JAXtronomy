"""GPU-vs-CPU parity checks for EPL in jaxtronomy."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, Optional

import numpy as np
import pytest


def _json_from_subprocess_output(stdout: str, stderr: str) -> Dict[str, object]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    lines.extend(line.strip() for line in stderr.splitlines() if line.strip())
    for line in reversed(lines):
        try:
            return json.loads(line)
        except Exception:
            continue
    raise RuntimeError("No JSON payload received from subprocess.")


def _probe_accelerator_platform() -> Optional[str]:
    code = r"""
import json
import jax

platforms = [str(getattr(d, "platform", "")).strip().lower() for d in jax.devices()]
accel = next((p for p in platforms if p not in {"", "cpu"}), None)
print(json.dumps({"platforms": platforms, "accelerator": accel}))
"""
    env = os.environ.copy()
    env.pop("JAX_PLATFORM_NAME", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = _json_from_subprocess_output(proc.stdout, proc.stderr)
    accel = payload.get("accelerator")
    if accel is None:
        return None
    accel_s = str(accel).strip().lower()
    return accel_s or None


def _run_epl_payload(platform_env: Optional[str]) -> Dict[str, object]:
    code = r"""
import json
import numpy as np
import jax
from jaxtronomy.LensModel.Profiles.epl import EPL

rng = np.random.default_rng(20260226)
x = rng.uniform(-2.0, 2.0, size=128).astype(np.float32)
y = rng.uniform(-2.0, 2.0, size=128).astype(np.float32)

# Stay away from the singular center to keep parity metrics stable.
r = np.hypot(x, y)
mask = r < 0.15
x[mask] = x[mask] + 0.35
y[mask] = y[mask] - 0.25

theta_E = 1.15
gamma = 2.08
e1 = 0.10
e2 = -0.07

profile = EPL()

f = np.asarray(profile.function(x, y, theta_E, gamma, e1, e2), dtype=np.float64)
fx, fy = profile.derivatives(x, y, theta_E, gamma, e1, e2)
fx = np.asarray(fx, dtype=np.float64)
fy = np.asarray(fy, dtype=np.float64)
fxx, fxy, fyx, fyy = profile.hessian(x, y, theta_E, gamma, e1, e2)
fxx = np.asarray(fxx, dtype=np.float64)
fxy = np.asarray(fxy, dtype=np.float64)
fyx = np.asarray(fyx, dtype=np.float64)
fyy = np.asarray(fyy, dtype=np.float64)

payload = {
    "default_backend": str(jax.default_backend()).strip().lower(),
    "device_platform": str(getattr(jax.devices()[0], "platform", "")).strip().lower(),
    "function": f.tolist(),
    "fx": fx.tolist(),
    "fy": fy.tolist(),
    "fxx": fxx.tolist(),
    "fxy": fxy.tolist(),
    "fyx": fyx.tolist(),
    "fyy": fyy.tolist(),
}
print(json.dumps(payload))
"""
    env = os.environ.copy()
    if platform_env is None:
        env.pop("JAX_PLATFORM_NAME", None)
    else:
        env["JAX_PLATFORM_NAME"] = str(platform_env)
    env["JAX_ENABLE_X64"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return _json_from_subprocess_output(proc.stdout, proc.stderr)


def _median_rel_delta(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(a), 1e-6)
    rel = np.abs(a - b) / denom
    return float(np.median(rel))


def test_epl_gpu_cpu_parity():
    accel_platform = _probe_accelerator_platform()
    if accel_platform is None:
        pytest.skip("No JAX accelerator backend detected for GPU-vs-CPU EPL parity test.")

    cpu = _run_epl_payload("cpu")
    # Leave backend selection unset so JAX picks the platform-native accelerator
    # (e.g. METAL on Apple); forcing JAX_PLATFORM_NAME="metal" is brittle across
    # JAX versions that expose uppercase backend names.
    gpu = _run_epl_payload(None)

    assert str(cpu["device_platform"]).lower() == "cpu"
    gpu_platform = str(gpu["device_platform"]).strip().lower()
    if gpu_platform == "cpu":
        pytest.skip("Default JAX runtime resolved to CPU; no runnable GPU backend selected.")
    assert gpu_platform != "cpu"

    arrays_cpu = {k: np.asarray(cpu[k], dtype=np.float64) for k in ("function", "fx", "fy", "fxx", "fxy", "fyx", "fyy")}
    arrays_gpu = {k: np.asarray(gpu[k], dtype=np.float64) for k in ("function", "fx", "fy", "fxx", "fxy", "fyx", "fyy")}

    thresholds = {
        "function": 2.0e-3,
        "fx": 3.0e-3,
        "fy": 3.0e-3,
        "fxx": 5.0e-3,
        "fxy": 5.0e-3,
        "fyx": 5.0e-3,
        "fyy": 5.0e-3,
    }

    for key, max_med_rel in thresholds.items():
        med_rel = _median_rel_delta(arrays_cpu[key], arrays_gpu[key])
        assert med_rel <= max_med_rel, (
            f"{key} median relative delta too high: {med_rel:.6g} > {max_med_rel:.6g} "
            f"(gpu platform={gpu['device_platform']}, cpu backend={cpu['default_backend']})"
        )

    # Symmetry sanity checks on both paths.
    np.testing.assert_allclose(arrays_cpu["fxy"], arrays_cpu["fyx"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(arrays_gpu["fxy"], arrays_gpu["fyx"], atol=2e-5, rtol=2e-5)
