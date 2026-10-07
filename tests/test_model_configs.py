import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_generate_eagle3_config_for_supported_families(tmp_path, family):
    model = tmp_path / family
    model.mkdir()
    config = {
        "model_type": family,
        "vocab_size": 128,
        "hidden_size": 32,
        "num_hidden_layers": 12,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
    }
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / f"{family}.json"
    script = Path(__file__).parents[1] / "scripts" / "generate_eagle3_config.py"
    subprocess.run([
        sys.executable, str(script), "--target-model-path", str(model),
        "--output", str(output), "--draft-vocab-size", "64",
    ], check=True, capture_output=True, text=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["architectures"] == ["LlamaForCausalLMEagle3"]
    assert result["draft_vocab_size"] == 64
    assert result["eagle_config"]["eagle_aux_hidden_state_layer_ids"] == [1, 5, 8]


def test_generate_eagle3_config_rejects_unknown_family(tmp_path):
    model = tmp_path / "unknown"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "mamba"}), encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "generate_eagle3_config.py"
    result = subprocess.run([
        sys.executable, str(script), "--target-model-path", str(model),
        "--output", str(tmp_path / "out.json"),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "validated only" in result.stderr
