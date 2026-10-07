#!/usr/bin/env python3
"""Fail-fast validation for the pinned SpecNaacl B200 environment."""

from __future__ import annotations

import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import sys


IMPORT_NAMES = {
    "cuda-tile": "cuda.tile",
    "huggingface-hub": "huggingface_hub",
    "latex2sympy2-extended": "latex2sympy2_extended",
    "math-verify": "math_verify",
    "openai-harmony": "openai_harmony",
    "pyyaml": "yaml",
    "typing-extensions": "typing_extensions",
}
# Interpreter admission is a lower bound, not a patch-release allowlist.
# The dependency pins/imports and CUDA checks below remain separate: accepting
# a newer interpreter does not guarantee wheels/runtime support for that version.
MIN_PYTHON_VERSION = (3, 12, 0)

# Narrow patch compatibility exception, not an unbounded dependency range.
# requirements.txt still specifies exactly one reproducible installation pin.
COMPATIBLE_PATCH_VERSIONS = {"peft": frozenset({"0.21.1", "0.21.2"})}
REQUIRED_APIS = {
    "peft": (
        "get_peft_config", "get_peft_model", "LoraConfig", "TaskType", "PeftType",
        "get_peft_model_state_dict", "set_peft_model_state_dict", "PeftModel",
    ),
}


def version_matches(distribution, expected, actual):
    actual = actual.split("+", 1)[0]
    if actual == expected:
        return True
    allowed = COMPATIBLE_PATCH_VERSIONS.get(distribution.lower().replace("_", "-"), ())
    return expected in allowed and actual in allowed


def validate_python_version():
    if sys.version_info[:3] >= MIN_PYTHON_VERSION:
        return
    minimum = ".".join(map(str, MIN_PYTHON_VERSION))
    recommended = ".".join(map(str, MIN_PYTHON_VERSION[:2]))
    environment = f"venv-py{recommended.replace('.', '')}"
    raise RuntimeError(
        f"Python >={minimum} is required; found {sys.version.split()[0]}\n"
        f"Interpreter: {sys.executable}\n"
        "Create a supported environment from the project directory:\n"
        f"  uv python install {recommended}\n"
        f"  uv venv --python {recommended} --seed {environment}\n"
        f"  source {environment}/bin/activate\n"
        '  export PYTHON_BIN="$(command -v python)"\n'
        "Install the project dependencies in this environment; see ENVIRONMENT.md."
    )


def pinned_requirements(path: Path):
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match is None:
            raise RuntimeError(f"requirement is not exactly pinned: {line}")
        yield match.group(1), match.group(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", type=Path, default=Path("requirements.txt"))
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--python-only", action="store_true",
                        help="check the supported interpreter without importing dependencies")
    args = parser.parse_args()
    validate_python_version()
    if args.python_only:
        return

    failures = []
    for distribution, expected in pinned_requirements(args.requirements):
        try:
            actual = version(distribution)
        except PackageNotFoundError:
            failures.append(f"{distribution}: not installed")
            continue
        if not version_matches(distribution, expected, actual):
            failures.append(f"{distribution}: expected {expected}, found {actual}")
            continue
        module_name = IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))
        try:
            module = importlib.import_module(module_name)
            missing = [name for name in REQUIRED_APIS.get(distribution.lower(), ())
                       if not hasattr(module, name)]
            if missing:
                raise ImportError("required APIs missing: " + ", ".join(missing))
        except Exception as exc:
            failures.append(
                f"{distribution}: import {module_name} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if actual.split("+", 1)[0] != expected:
            print(f"{distribution}: accepted compatible patch {actual} (install pin {expected})")
    if failures:
        raise RuntimeError("environment validation failed:\n- " + "\n- ".join(failures))

    torch = importlib.import_module("torch")
    if args.require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) < (10, 0):
            raise RuntimeError(
                f"B200-class compute capability >=10.0 required; found {major}.{minor}"
            )

    backend = importlib.import_module("specforge.offline_capture.sglang_backend")
    if not hasattr(backend, "OfflineSGLangCaptureBackend"):
        raise RuntimeError("SpecForge SGLang offline capture API is unavailable")
    print(
        "environment validation passed: "
        f"python={sys.version.split()[0]} torch={torch.__version__} "
        f"cuda={torch.version.cuda}"
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
