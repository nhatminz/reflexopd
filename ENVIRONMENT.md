# Reproducible B200 environment

The project's interpreter gate is **Python >=3.12.0**, not an exact patch
allowlist. Python 3.12.3, 3.12.12, 3.12.13 and later releases pass this gate.
Use a maintained 3.12 patch release when creating a new production environment.
The optional `.python-version` file contains `3.12` as a uv preference, not a
fixed patch version. Runtime validation does not read it, so it can be omitted
on the server. Newer minor versions also pass the interpreter gate, but are not
automatically certified for the pinned CUDA/dependency stack: compatible wheels,
dependency versions, imports and GPU APIs must still be checked independently.
The library stack is PyTorch 2.13.0 with the CUDA 13.0
wheel, Transformers 5.12.1, and SGLang 0.5.18. CUDA 13.x requires an NVIDIA
driver from the R580 branch or newer. The PyTorch wheel carries its CUDA runtime
libraries; a system CUDA toolkit is not required unless building an optional
CUDA extension.

## Online installation

### Existing PEFT 0.21.1 environment

Keep the installed PEFT 0.21.1; no Internet download or upgrade is required for
this fix. The canonical installation pin is `peft==0.21.1`. Runtime validation
also admits the explicit 0.21.2 patch, checks PEFT imports and the LoRA/checkpoint
APIs used by training/benchmarking, and keeps all other version pins strict.
This is not certification of every PEFT execution path; validate the actual
server environment and run a short training smoke before a long job.

```bash
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" scripts/validate_environment.py --require-cuda
CUDA_VISIBLE_DEVICES=0 bash train_qwen3_1p7b.sh
```

Do not replace the venv or reinstall the full requirements set to resolve the
old `peft: expected 0.21.2, found 0.21.1` gate. Transfer the updated
`requirements.txt` and `scripts/validate_environment.py` to the server first.

For a new environment, install Python 3.12 with `uv`, then create the
environment from that managed interpreter. An existing Python >=3.12.0
environment can be kept; use the validation commands below:

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
nvidia-smi
uv python install 3.12
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
python scripts/validate_environment.py --python-only
python -m pip install --upgrade pip setuptools wheel

python -m pip install torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install --no-deps -e third_party/SpecForge --no-build-isolation

python scripts/validate_environment.py --require-cuda
python -m pip check
python -m compileall -q .
pytest -q
```

Confirm that `torch.__version__` is `2.13.0+cu130` and `torch.version.cuda` is
`13.0`:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
```

## Preparing an offline wheelhouse

Run this on an Internet-connected Linux x86-64 machine with Python 3.12, then
copy both the project and `wheelhouse/` to the B200 machine. `pip wheel` is used
instead of `pip download` so source distributions are built before transfer.

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
uv python install 3.12
uv venv --python 3.12 --seed .wheel-builder
source .wheel-builder/bin/activate
python -m pip install --upgrade pip wheel setuptools
mkdir -p wheelhouse

python -m pip wheel --wheel-dir wheelhouse torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip wheel --wheel-dir wheelhouse --find-links wheelhouse \
  -r requirements.txt
```

On the offline B200 machine:

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
python3.12 scripts/validate_environment.py --python-only
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
python -m pip install --no-index --find-links wheelhouse torch==2.13.0
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
python -m pip install --no-deps -e third_party/SpecForge --no-build-isolation

python scripts/validate_environment.py --require-cuda
python -m pip check
python -m compileall -q .
pytest -q
```

No `requirements-optional.txt` is needed for this pipeline. New EAGLE pretraining
runs request the optional standard `flash_attn` v2 backend and explicitly fail
if its CUDA forward/backward interface is unavailable; there is no silent
fallback. Install a compatible build or set `PRETRAIN_ATTENTION_BACKEND=sdpa`
(or `flex_attention`) explicitly. Resumed runs retain their saved backend.
SGLang 0.5.18 has its own mandatory `flash-attn-4`
dependency; it remains governed by SGLang's package metadata and must be present
for `pip check` to pass.

### Recovery: `cannot import name 'flash_attn_varlen_func'`

This means the selected EAGLE `fa` backend cannot import the standard varlen
interface it uses. An importable `flash_attn` namespace or a successful SGLang
dependency check is not proof that this interface is available. The launcher
stops before feature capture/training; this particular error is not an OOM.
Do not remove SGLang dependencies or change the pinned Torch stack to bypass it.

Choose the existing backend that does not need the external FA interface:

```bash
export PRETRAIN_ATTENTION_BACKEND=sdpa
# Rerun the original pretrain command with its model/dataset/batch settings.
bash pretrain_qwen25_3b.sh  # include your original environment assignments
```

To check the choice independently of dataset preparation:

```bash
python scripts/check_pretrain_attention.py --backend sdpa
# If you want to use standard FA, validate its real CUDA forward/backward:
CUDA_VISIBLE_DEVICES=0 python scripts/check_pretrain_attention.py --backend fa --probe
```

SDPA is an explicit choice, not a silent fallback, and its existing EAGLE cached
TTT implementation is unchanged. Checking `sdpa` only validates the selection;
it is not a CUDA execution/throughput benchmark. On backend-check failure the
launcher now prints the SDPA override and the prepared run path. A model wrapper
can reuse that directory with `RESUME=/absolute/run/path`. If a real checkpoint
exists, normal resume validation still applies; never change its saved backend
implicitly. Batch size 64 in the reported command was not changed by this fix;
whether it fits at length 2048/TTT 7 must be established on the GPU separately.

## Using an existing Python 3.12.3 (or newer) environment

Both launchers accept Python >=3.12.0 without a `.python-version` file, including
3.12.3. The optional file requests the 3.12 series for new uv environments; it
does not change the interpreter in an existing venv. The explicit `--python 3.12` option
in the installation commands also works without this file (see
[uv Python version files](https://docs.astral.sh/uv/concepts/python-versions/#python-version-files)).
CPython documents ABI compatibility across patch releases within the same minor
release when builds match (see [C API stability](https://docs.python.org/3/c-api/stable.html)).
This supports relaxing the patch allowlist; it does not by itself establish
compatibility of private extension APIs or different Python minor versions.
The project still checks dependency versions, imports, and required runtime APIs.

If an older checkout rejects 3.12.3 because it only accepts 3.12.12/3.12.13, update
the project code and keep the existing environment:

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" --version
"$PYTHON_BIN" scripts/validate_environment.py --python-only
"$PYTHON_BIN" scripts/validate_environment.py --require-cuda
"$PYTHON_BIN" -m pip check
```

Then rerun the original pretrain command in the same shell. Export `PYTHON_BIN`
explicitly because an older value may still point to a different environment.
If dependency validation reports missing or incompatible packages, install the
pinned dependencies using the online or offline commands above.

## Recovery: `No module named 'helper.modeling_draft'` during EAGLE training

`helper/` is project source, not a pip dependency. The reported server passed
Python 3.12.3 / Torch 2.13.0+cu130 validation, then failed on an unconditional
legacy draft import despite selecting `--draft_backend eagle3`. The old
entrypoint also prepended the parent directory ahead of the project itself,
allowing an unrelated parent `helper` package to shadow the local package.
The traceback alone does not establish whether the remote checkout was
incomplete or a foreign helper package was selected.

Update the project code and keep the existing environment/checkpoints:

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" scripts/check_training_sources.py --backend eagle3
"$PYTHON_BIN" -c 'import helper; print(helper.__file__)'
```

The helper path should point inside this project's `helper/` directory. Sync
the complete project revision, including `helper/__init__.py`, if source files
are missing; do not install an unrelated `helper` package to work around it.
Then rerun the original training command. The entrypoint now puts its own
checkout first and imports `helper.modeling_draft` only for the legacy backend.
The launcher checks required local sources before importing the full CUDA
runtime or starting torchrun. EAGLE does not require the unused legacy module.
The checker checks files only; it is not an end-to-end GPU/dependency benchmark.
