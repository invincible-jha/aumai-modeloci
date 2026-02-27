# Getting Started with aumai-modeloci

This guide walks you from a fresh Python environment to a packed, inspected, and unpacked OCI
model archive in about five minutes.

---

## Prerequisites

- Python 3.11 or later
- `pip` (or `uv`, `poetry`, `pipenv` — any installer that handles `pyproject.toml`)
- A directory containing your model files (weights, config, tokenizer, etc.)

---

## Installation

### From PyPI

```bash
pip install aumai-modeloci
```

### From source

```bash
git clone https://github.com/aumai/aumai-modeloci.git
cd aumai-modeloci
pip install -e ".[dev]"
```

### Verify the installation

```bash
aumai-modeloci --version
python -c "import aumai_modeloci; print(aumai_modeloci.__version__)"
```

---

## Step-by-Step Tutorial

### Step 1 — Prepare a model directory

For this tutorial we will create a toy model directory. In production, point `--model-dir` at
your real weights, config, and tokenizer files.

```python
# create_toy_model.py
import os, pathlib

model_dir = pathlib.Path("./tutorial-model")
model_dir.mkdir(exist_ok=True)

(model_dir / "model.bin").write_bytes(b"\x00" * 1024)  # fake weights
(model_dir / "config.json").write_text('{"hidden_size": 768}')
(model_dir / "tokenizer.json").write_text('{"vocab_size": 30522}')
print(f"Created {model_dir} with {len(list(model_dir.iterdir()))} files")
```

```bash
python create_toy_model.py
```

### Step 2 — Pack the model

```bash
aumai-modeloci pack \
  --model-dir ./tutorial-model \
  --name tutorial-bert \
  --version 0.1.0 \
  --framework pytorch \
  --architecture transformer \
  --metadata '{"task":"classification","accuracy":0.94}'
```

Expected output:

```
Packaged model: ./tutorial-bert-0.1.0.tar
  Name        : tutorial-bert
  Version     : 0.1.0
  Framework   : pytorch
  Archive     : ./tutorial-bert-0.1.0.tar
```

The archive `tutorial-bert-0.1.0.tar` is written to the parent directory of `--model-dir`
(the current directory in this case).

### Step 3 — Inspect the archive

```bash
aumai-modeloci inspect --archive ./tutorial-bert-0.1.0.tar
```

```
Model    : tutorial-bert
Version  : 0.1.0
Framework: pytorch
Arch     : transformer
Metadata : {"task": "classification", "accuracy": 0.94}

Layers (3):
  model.bin                                   1.0 KB  sha256:4f53cda18c2b...
  config.json                                 0.0 KB  sha256:9a19c1bde96a...
  tokenizer.json                              0.0 KB  sha256:3b4c5d6e7f8a...

Layer verification (3 layers):
  OK  sha256:4f53cda18c2b...
  OK  sha256:9a19c1bde96a...
  OK  sha256:3b4c5d6e7f8a...
All layers verified.
```

The inspect command verifies every layer digest without writing anything to disk. If it exits
cleanly the archive is intact.

### Step 4 — Unpack the archive

```bash
aumai-modeloci unpack \
  --archive ./tutorial-bert-0.1.0.tar \
  --output ./unpacked-model
```

```
Unpacked to: ./unpacked-model
  Model   : tutorial-bert v0.1.0
  Framework: pytorch
```

The output directory will contain the original model files plus `manifest.json`,
`config.json`, and the `blobs/sha256/` directory with the raw layer blobs.

### Step 5 — Use the Python API

```python
from aumai_modeloci.core import ModelPackager, ModelUnpackager
from aumai_modeloci.models import OCIConfig

# Pack
config = OCIConfig(
    model_name="tutorial-bert",
    version="0.1.0",
    framework="pytorch",
    architecture="transformer",
)
packager = ModelPackager()
archive = packager.package("./tutorial-model", config)
print(f"Packed: {archive}")

# Verify
unpacker = ModelUnpackager()
results = unpacker.verify_layers(archive)
assert all(valid for _, valid in results), "Integrity check failed!"
print("All layers valid.")

# Unpack
config_out = unpacker.unpack(archive, "./unpacked-model")
print(f"Unpacked: {config_out.model_name} {config_out.version}")
```

---

## Common Patterns

### Pattern 1 — CI/CD integrity gate

Add a step to your deployment pipeline that verifies the archive before pushing it to a
registry or deploying it to production:

```python
from aumai_modeloci.core import ModelUnpackager

def integrity_gate(archive_path: str) -> None:
    unpacker = ModelUnpackager()
    results = unpacker.verify_layers(archive_path)
    failed = [(d, v) for d, v in results if not v]
    if failed:
        raise RuntimeError(
            f"Archive integrity check failed for {len(failed)} layer(s): "
            + ", ".join(d for d, _ in failed)
        )
    print(f"Integrity gate passed: {len(results)} layer(s) verified.")

integrity_gate("my-model-1.0.0.tar")
```

### Pattern 2 — Metadata-rich packing for MLOps

Embed experiment tracking metadata, evaluation metrics, and dataset lineage directly in the
archive config so downstream consumers have full provenance:

```python
from aumai_modeloci.core import ModelPackager
from aumai_modeloci.models import OCIConfig

config = OCIConfig(
    model_name="llm-customer-support",
    version="2.3.1",
    framework="pytorch",
    architecture="transformer",
    metadata={
        "base_model": "llama-3-8b",
        "fine_tune_dataset": "cs-support-v4",
        "training_run": "mlflow://experiments/42/runs/abc123",
        "eval_accuracy": 0.913,
        "eval_f1": 0.887,
        "training_date": "2025-11-15",
        "trainer": "jane.smith@example.com",
    },
)

packager = ModelPackager()
archive = packager.package("./llm-finetuned", config)
print(f"Archive with full provenance: {archive}")
```

### Pattern 3 — Reading config without unpacking

Use the `inspect` CLI or read `config.json` directly from the tar in Python when you only
need the metadata:

```python
import json, tarfile
from aumai_modeloci.models import OCIConfig

with tarfile.open("llm-customer-support-2.3.1.tar", "r") as tar:
    member = tar.getmember("config.json")
    fh = tar.extractfile(member)
    config = OCIConfig.model_validate(json.loads(fh.read()))

print(config.model_name, config.version)
print(config.metadata.get("eval_accuracy"))
```

### Pattern 4 — Programmatic layer manifest

When integrating with an OCI registry push (via `oras push` or similar), generate the
manifest programmatically and use it to construct the registry API payload:

```python
from aumai_modeloci.core import ModelPackager
from aumai_modeloci.models import ModelLayer, OCIConfig

config = OCIConfig(
    model_name="onnx-resnet",
    version="1.0.0",
    framework="onnx",
    architecture="cnn",
)
# Suppose you have already computed layer digests via some other pipeline
layers = [
    ModelLayer(
        digest="sha256:aabbcc...",
        size=52_428_800,
        annotations={"org.opencontainers.image.title": "resnet50.onnx"},
    ),
]
packager = ModelPackager()
manifest = packager.create_manifest(config, layers)
# manifest.model_dump_json() is ready to POST to /v2/<name>/manifests/<tag>
print(manifest.model_dump_json(indent=2))
```

### Pattern 5 — Adding a layer after the fact

When you need to append an extra file (vocabulary, lookup table, external config) to an
already-packed archive:

```python
from aumai_modeloci.core import ModelPackager

packager = ModelPackager()
layer = packager.add_layer(
    archive_path="my-model-1.0.0.tar",
    file_path="./extra-vocabulary.json",
)
print(f"Added: {layer.annotations.get('org.opencontainers.image.title')}")
print(f"Digest: {layer.digest}")
```

---

## Troubleshooting FAQ

**Q: `aumai-modeloci pack` exits with `Error: './my-model' is not a directory.`**

The path passed to `--model-dir` must be an existing directory, not a file or a zip archive.
Verify with `ls ./my-model` before running the command.

---

**Q: `inspect` reports `FAIL` for a layer — what does that mean?**

The stored blob filename (which is its SHA-256) does not match the SHA-256 of its contents.
The archive was corrupted or modified after it was packed. Re-pack from the source directory.

---

**Q: I get `FileNotFoundError: config.json not found in archive` during `unpack`.**

The archive was not created by `aumai-modeloci pack` — it does not follow the expected layout.
Run `inspect` on it to see what is actually inside.

---

**Q: My model directory has subdirectories. Are they preserved?**

Yes. `ModelPackager` uses `model_path.rglob("*")` to collect all files recursively and stores
the relative path in the `org.opencontainers.image.title` annotation. Unpacking restores the
full relative path.

---

**Q: Can I push the archive to Docker Hub or AWS ECR?**

Not directly with the `.tar` output — that is an OCI layout archive, not an OCI image index.
Use `oras push` or `crane push` to push the unpacked OCI layout directory to a registry that
supports OCI artifacts (ECR, GHCR, Artifact Registry).

---

**Q: The Python API raises `TypeError: extractall() got an unexpected keyword argument 'filter'` on Python 3.11.**

This is handled automatically. On Python < 3.12 the code falls back to manual path-traversal
validation and calls `extractall` without the `filter` argument. No action needed.

---

**Q: Can I use a custom media type for my layers?**

Yes. Construct `ModelLayer` objects manually and pass them to `create_manifest`. The
`media_type` field defaults to `application/vnd.oci.image.layer.v1.tar+gzip` but can be any
string.
