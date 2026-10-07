from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def _requirement_names(path):
    names = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "http://", "https://")):
            continue
        match = re.match(r"[A-Za-z0-9_.-]+", line)
        if match:
            names.add(match.group(0).lower().replace("_", "-"))
    return names


def test_requirements_are_exact_and_contain_no_stdlib_packages():
    requirement_paths = [
        ROOT / "requirements.txt",
        ROOT / "requirements-bootstrap.txt",
        ROOT / "requirements-external.txt",
    ]
    names = set().union(*(_requirement_names(path) for path in requirement_paths))
    stdlib_names = getattr(
        sys,
        "stdlib_module_names",
        {"asyncio", "datetime", "pathlib", "statistics", "typing", "json", "os"},
    )
    stdlib = {name.lower().replace("_", "-") for name in stdlib_names}
    assert not names.intersection(stdlib)
    for requirement_path in requirement_paths:
        for raw_line in requirement_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if line and not line.startswith("-"):
                assert "==" in line, f"dependency is not exactly pinned: {line}"


def test_specforge_is_installed_without_dependency_resolution():
    environment = (ROOT / "ENVIRONMENT.md").read_text(encoding="utf-8")
    assert "pip install --no-deps -e third_party/SpecForge" in environment


def test_offline_install_never_uses_an_index_or_source_distribution():
    builder = (ROOT / "scripts/build_offline_wheelhouse.sh").read_text(encoding="utf-8")
    installer = (ROOT / "scripts/install_offline.sh").read_text(encoding="utf-8")
    assert "https://pypi.nvidia.com" in builder
    assert "cuda-tile==1.6.0rc5" in builder
    assert "--only-binary=:all:" in builder
    assert "PIP_NO_INDEX=1" in installer
    assert "PIP_CONFIG_FILE=/dev/null" in installer
    assert "--no-index" in installer
    assert "--only-binary=:all:" in installer


def test_external_bundle_contains_only_the_required_special_artifact():
    requirements = _requirement_names(ROOT / "requirements-external.txt")
    assert requirements == {"cuda-tile"}
    downloader = (ROOT / "scripts/download_external_wheels.sh").read_text(
        encoding="utf-8"
    )
    guide = (ROOT / "EXTERNAL_WHEELS_HF.md").read_text(encoding="utf-8")
    assert "--index-url https://pypi.nvidia.com" in downloader
    assert "--no-deps" in downloader
    assert "--only-binary=:all:" in downloader
    assert "hf upload" in guide
    assert "hf download" in guide
    assert "--find-links" in guide
