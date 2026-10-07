"""Import/bootstrap regressions without importing the GPU training entrypoint."""

import ast
import builtins
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from scripts import check_training_sources


ROOT = Path(__file__).resolve().parents[1]


def entrypoint_tree():
    return ast.parse((ROOT / "grpo_speculative.py").read_text(encoding="utf-8"))


@pytest.mark.parametrize("repo_already_present", [False, True])
def test_entrypoint_prefers_local_helpers_over_parent_package(tmp_path, repo_already_present):
    # Reproduce a parent helper package that lacks modeling_draft.py, exactly
    # the failure seen when the old bootstrap prepended the parent directory.
    repo = tmp_path / "SpecNaacl"
    local = repo / "helper"
    foreign = tmp_path / "helper"
    local.mkdir(parents=True)
    foreign.mkdir()
    (foreign / "__init__.py").write_text("ORIGIN = 'foreign'\n", encoding="utf-8")
    (local / "__init__.py").write_text("ORIGIN = 'local'\n", encoding="utf-8")
    (local / "modeling_draft.py").write_text("ORIGIN = 'local'\n", encoding="utf-8")
    bootstrap = []
    for node in entrypoint_tree().body:
        if isinstance(node, ast.Import) and any(alias.name == "pandas" for alias in node.names):
            break
        bootstrap.append(node)
    source = ast.unparse(ast.Module(body=bootstrap, type_ignores=[]))
    initial = [str(tmp_path)] + ([str(repo)] if repo_already_present else [])
    program = (
        f"import sys\nsys.path[:0] = {initial!r}\n"
        f"__file__ = {str(repo / 'grpo_speculative.py')!r}\n"
        f"exec({source!r})\n"
        "import helper\nfrom helper import modeling_draft\n"
        "assert helper.ORIGIN == modeling_draft.ORIGIN == 'local'\n"
        f"assert sys.path[0] == {str(repo)!r}\n"
        "print('local helper selected')\n"
    )
    result = subprocess.run([sys.executable, "-c", program], cwd=tmp_path,
                            env=dict(os.environ), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "local helper selected" in result.stdout


def draft_selection_code():
    node = next(node for node in entrypoint_tree().body
                if isinstance(node, ast.If)
                and ast.unparse(node.test) == "args.draft_backend == 'eagle3'")
    return compile(ast.Module(body=[node], type_ignores=[]), "grpo_speculative.py", "exec")


def selection_scope(backend):
    return {
        "args": SimpleNamespace(
            draft_backend=backend, eagle_feature_layers="", draft_config="config.json",
            vocab_mapping="mapping.pt", draft_initialization_mode="pretrained",
            opd_rank=8,eagle_ttt_length=7, eagle_lk_loss_type="", eagle_kl_scale=1.0, eagle_kl_decay=0.9,
        ),
        "target_model": object(), "adapter_path": "pretrained-checkpoint",
        "Eagle3FastGRPOAdapter": mock.Mock(), "config": SimpleNamespace(),
        "model_torch_dtype": "bf16",
        "method":"opd_reflex",
    }


def test_eagle_selection_does_not_import_legacy_model():
    scope = selection_scope("eagle3")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "helper.modeling_draft":
            pytest.fail("EAGLE must not depend on the unused legacy implementation")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=guarded_import):
        exec(draft_selection_code(), scope)
    factory = scope["Eagle3FastGRPOAdapter"]
    assert scope["model"] is factory.return_value
    assert factory.call_args.kwargs["draft_checkpoint"] == "pretrained-checkpoint"
    assert factory.call_args.kwargs["ttt_length"] == 7
    assert factory.call_args.kwargs["initialization_mode"] == "pretrained"


def test_legacy_selection_still_loads_the_original_model(monkeypatch):
    factory = mock.Mock()
    monkeypatch.setitem(sys.modules, "helper.modeling_draft", SimpleNamespace(Model=factory))
    scope = selection_scope("legacy")
    exec(draft_selection_code(), scope)
    factory.assert_called_once_with(scope["config"], target_model=scope["target_model"])
    factory.return_value.load_model.assert_called_once_with("pretrained-checkpoint")
    scope["Eagle3FastGRPOAdapter"].assert_not_called()
    assert scope["config"].num_hidden_layers == 1
    assert scope["config"].torch_dtype == "bf16"


def test_legacy_model_is_not_a_top_level_import():
    imports = [node.module for node in entrypoint_tree().body if isinstance(node, ast.ImportFrom)]
    assert "helper.modeling_draft" not in imports


def minimal_eagle_checkout(tmp_path):
    helper = tmp_path / "helper"
    helper.mkdir()
    for path in check_training_sources.validate_training_sources(ROOT, "eagle3"):
        (helper / path.name).write_text("", encoding="utf-8")
    return tmp_path


def test_source_check_allows_eagle_checkout_without_unused_legacy(tmp_path, monkeypatch, capsys):
    repo = minimal_eagle_checkout(tmp_path)
    assert not (repo / "helper/modeling_draft.py").exists()
    monkeypatch.setattr(check_training_sources, "REPO_ROOT", repo)
    assert check_training_sources.main(["--backend", "eagle3"]) == 0
    assert "backend=eagle3" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match="modeling_draft.py"):
        check_training_sources.validate_training_sources(repo, "legacy")


@pytest.mark.parametrize("missing", ["__init__.py", "eagle3_specforge.py", "sampling.py"])
def test_incomplete_checkout_gives_actionable_source_error(tmp_path, missing):
    repo = minimal_eagle_checkout(tmp_path)
    (repo / "helper" / missing).unlink()
    with pytest.raises(FileNotFoundError) as error:
        check_training_sources.validate_training_sources(repo, "eagle3")
    assert missing in str(error.value)
    assert "Sync the helper/ directory" in str(error.value)
    assert "not a pip package" in str(error.value)


def test_source_check_cli_exits_cleanly_for_incomplete_checkout(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(check_training_sources, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit) as error:
        check_training_sources.main(["--backend", "eagle3"])
    assert error.value.code == 2
    assert "Incomplete SpecNaacl checkout" in capsys.readouterr().err


def test_launcher_checks_local_sources_before_full_runtime_validation():
    source = (ROOT / "scripts/launch/train_model.sh").read_text(encoding="utf-8")
    source_check = source.index('"$PROJECT_DIR/scripts/check_training_sources.py" --backend eagle3')
    assert source.index('"$PROJECT_DIR/scripts/validate_environment.py" --python-only') < source_check
    assert source_check < source.index('--requirements "$PROJECT_DIR/requirements.txt" --require-cuda')
