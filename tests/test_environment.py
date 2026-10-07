import sys

import pytest

from scripts import validate_environment


def set_python_version(monkeypatch, version):
    monkeypatch.setattr(sys, "version_info", (*version, "final", 0))
    monkeypatch.setattr(sys, "version", ".".join(map(str, version)))


@pytest.mark.parametrize("version", [
    (3, 12, 0), (3, 12, 1), (3, 12, 3), (3, 12, 11), (3, 12, 12),
    (3, 12, 13), (3, 12, 14), (3, 12, 99), (3, 13, 0), (3, 14, 0),
])
def test_python_only_accepts_supported_versions_without_dependencies(monkeypatch, tmp_path, version):
    monkeypatch.setattr(validate_environment, "__file__", str(tmp_path / "scripts/validate_environment.py"))
    assert not (tmp_path / ".python-version").exists()
    set_python_version(monkeypatch, version)
    monkeypatch.setattr(sys, "argv", ["validate_environment.py", "--python-only",
                                      "--requirements", "missing-requirements.txt"])

    def unexpected_dependency_check(*args):
        pytest.fail("python-only validation must not inspect or import dependencies")

    monkeypatch.setattr(validate_environment, "version", unexpected_dependency_check)
    monkeypatch.setattr(validate_environment.importlib, "import_module",
                        unexpected_dependency_check)
    validate_environment.main()


@pytest.mark.parametrize("version", [(3, 9, 20), (3, 10, 15), (3, 11, 12), (3, 11, 99)])
def test_wrong_python_version_reports_interpreter_and_recovery(monkeypatch, tmp_path, version):
    monkeypatch.setattr(validate_environment, "__file__", str(tmp_path / "scripts/validate_environment.py"))
    assert not (tmp_path / ".python-version").exists()
    set_python_version(monkeypatch, version)
    with pytest.raises(RuntimeError) as error:
        validate_environment.validate_python_version()
    message = str(error.value)
    assert f"Python >=3.12.0 is required; found {sys.version}" in message
    assert sys.executable in message
    assert "uv venv --python 3.12 --seed venv-py312" in message
    assert "export PYTHON_BIN=" in message
    assert "ENVIRONMENT.md" in message


@pytest.mark.parametrize("version", [(3, 12, 3), (3, 12, 13), (3, 13, 9)])
def test_admitted_python_still_rejects_incompatible_dependencies(monkeypatch, version):
    set_python_version(monkeypatch, version)
    monkeypatch.setattr(sys, "argv", ["validate_environment.py", "--require-cuda"])
    monkeypatch.setattr(validate_environment, "pinned_requirements",
                        lambda path: iter([("torch", "2.13.0")]))
    monkeypatch.setattr(validate_environment, "version", lambda name: "0.0.0")
    with pytest.raises(RuntimeError, match="torch: expected 2.13.0, found 0.0.0"):
        validate_environment.main()


def test_python_3123_still_requires_cuda_when_requested(monkeypatch):
    from types import SimpleNamespace

    set_python_version(monkeypatch, (3, 12, 3))
    monkeypatch.setattr(sys, "argv", ["validate_environment.py", "--require-cuda"])
    monkeypatch.setattr(validate_environment, "pinned_requirements", lambda path: iter(()))
    monkeypatch.setattr(validate_environment.importlib, "import_module", lambda name:
                        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match="CUDA is required"):
        validate_environment.main()


def test_optional_uv_preference_does_not_pin_a_patch_release():
    from pathlib import Path

    preference = Path(__file__).resolve().parents[1] / ".python-version"
    assert preference.read_text(encoding="utf-8").strip() == "3.12"


@pytest.mark.parametrize("actual", ["0.21.1", "0.21.2"])
def test_peft_supported_patch_passes_real_validator_import_gate(monkeypatch, actual):
    from types import SimpleNamespace

    set_python_version(monkeypatch, (3, 12, 3))
    monkeypatch.setattr(sys, "argv", ["validate_environment.py"])
    monkeypatch.setattr(validate_environment, "pinned_requirements",
                        lambda path: iter([("peft", "0.21.1")]))
    monkeypatch.setattr(validate_environment, "version", lambda name: actual)
    imported = []
    def import_module(name):
        imported.append(name)
        if name == "peft":
            return SimpleNamespace(**{api: object() for api in validate_environment.REQUIRED_APIS["peft"]})
        if name == "torch":
            return SimpleNamespace(__version__="2.13.0+cu130", version=SimpleNamespace(cuda="13.0"))
        return SimpleNamespace(OfflineSGLangCaptureBackend=object())
    monkeypatch.setattr(validate_environment.importlib, "import_module", import_module)
    validate_environment.main()
    assert "peft" in imported


@pytest.mark.parametrize("actual", ["0.21.0", "0.21.3", "0.22.0"])
def test_other_peft_versions_are_not_silently_accepted(actual):
    assert not validate_environment.version_matches("peft", "0.21.1", actual)


def test_patch_exception_does_not_relax_other_pins_or_custom_peft_pin():
    assert not validate_environment.version_matches("torch", "2.13.0", "2.11.0+cu130")
    assert not validate_environment.version_matches("peft", "0.22.0", "0.21.1")
    assert validate_environment.version_matches("torch", "2.13.0", "2.13.0+cu130")


def test_supported_peft_patch_with_broken_api_still_fails(monkeypatch):
    from types import SimpleNamespace

    set_python_version(monkeypatch, (3, 12, 3))
    monkeypatch.setattr(sys, "argv", ["validate_environment.py"])
    monkeypatch.setattr(validate_environment, "pinned_requirements",
                        lambda path: iter([("peft", "0.21.1")]))
    monkeypatch.setattr(validate_environment, "version", lambda name: "0.21.1")
    monkeypatch.setattr(validate_environment.importlib, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(RuntimeError, match="required APIs missing:.*get_peft_model"):
        validate_environment.main()


def test_canonical_peft_pin_is_0211():
    from pathlib import Path

    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    assert dict(validate_environment.pinned_requirements(requirements))["peft"] == "0.21.1"
