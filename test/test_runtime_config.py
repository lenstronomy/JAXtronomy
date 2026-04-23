import types

import pytest

from jaxtronomy import _runtime_config as runtime_config


@pytest.fixture(autouse=True)
def _reset_runtime_config():
    runtime_config._reset_runtime_config_cache_for_tests()
    yield
    runtime_config._reset_runtime_config_cache_for_tests()


def test_is_macos_metal_backend_false_off_darwin(monkeypatch):
    monkeypatch.setattr(runtime_config.sys, "platform", "linux")
    monkeypatch.setattr(runtime_config.jax, "default_backend", lambda: "metal")
    monkeypatch.setattr(
        runtime_config.jax,
        "devices",
        lambda: [types.SimpleNamespace(platform="metal")],
    )

    assert runtime_config.is_macos_metal_backend() is False


def test_is_macos_metal_backend_true_when_device_platform_reports_metal(monkeypatch):
    monkeypatch.setattr(runtime_config.sys, "platform", "darwin")
    monkeypatch.setattr(runtime_config.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        runtime_config.jax,
        "devices",
        lambda: [types.SimpleNamespace(platform="METAL")],
    )

    assert runtime_config.is_macos_metal_backend() is True


def test_configure_precision_defaults_to_false_on_macos_metal(monkeypatch):
    monkeypatch.setattr(runtime_config.sys, "platform", "darwin")
    monkeypatch.delenv("JAXTRONOMY_ENABLE_X64", raising=False)
    monkeypatch.setattr(runtime_config.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        runtime_config.jax,
        "devices",
        lambda: [types.SimpleNamespace(platform="metal")],
    )
    update_calls = []
    monkeypatch.setattr(
        runtime_config.jax.config,
        "update",
        lambda key, value: update_calls.append((key, value)),
    )

    enabled = runtime_config.configure_jax_precision_for_runtime()

    assert enabled is False
    assert update_calls == [("jax_enable_x64", False)]


def test_configure_precision_defaults_to_true_off_metal(monkeypatch):
    monkeypatch.setattr(runtime_config.sys, "platform", "linux")
    monkeypatch.delenv("JAXTRONOMY_ENABLE_X64", raising=False)
    monkeypatch.setattr(runtime_config.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        runtime_config.jax,
        "devices",
        lambda: [types.SimpleNamespace(platform="cuda")],
    )
    update_calls = []
    monkeypatch.setattr(
        runtime_config.jax.config,
        "update",
        lambda key, value: update_calls.append((key, value)),
    )

    enabled = runtime_config.configure_jax_precision_for_runtime()

    assert enabled is True
    assert update_calls == [("jax_enable_x64", True)]
