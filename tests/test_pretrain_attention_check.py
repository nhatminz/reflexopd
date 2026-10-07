import importlib.util
from pathlib import Path
from unittest import mock

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_pretrain_attention.py"
spec = importlib.util.spec_from_file_location("check_pretrain_attention", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_checker_reports_explicit_sdpa_without_claiming_a_cuda_probe(capsys):
    assert check.main(["--backend", "sdpa", "--probe"]) == 0
    output = capsys.readouterr()
    assert "explicit selection" in output.out
    assert "probe passed" not in output.out
    assert not output.err


def test_checker_fails_fa_with_an_actionable_command_without_fallback(capsys):
    with mock.patch("specforge.training.pretrain_attention.validate_attention_backend",
                    side_effect=RuntimeError("missing flash_attn_varlen_func")) as validate:
        assert check.main(["--backend", "fa", "--probe"]) == 2
    validate.assert_called_once_with("fa", probe=True)
    output = capsys.readouterr()
    assert "missing flash_attn_varlen_func" in output.err
    assert "PRETRAIN_ATTENTION_BACKEND=sdpa" in output.err
    assert "No backend or training setting was changed automatically" in output.err
    assert "Traceback" not in output.err
    assert not output.out


def test_checker_probes_fa_when_explicitly_requested(capsys):
    with mock.patch("specforge.training.pretrain_attention.validate_attention_backend",
                    return_value="fa") as validate:
        assert check.main(["--backend", "fa", "--probe"]) == 0
    validate.assert_called_once_with("fa", probe=True)
    assert "CUDA forward/backward probe passed" in capsys.readouterr().out


def test_checker_rejects_unknown_backend():
    with pytest.raises(SystemExit) as error:
        check.main(["--backend", "typo"])
    assert error.value.code == 2
