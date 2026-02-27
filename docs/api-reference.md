# API Reference — aumai-modeloci

Complete reference for all public classes, methods, and Pydantic models exported by
`aumai-modeloci`.

---

## Module: `aumai_modeloci.core`

Public exports: `ModelPackager`, `ModelUnpackager`

---

### class `ModelPackager`

Packages an ML model directory into an OCI-compliant tar archive.

The archive layout is:

```
blobs/sha256/<hex>    # one gzip-compressed tar blob per source file
config.json           # human-readable OCIConfig at archive root
manifest.json         # OCIManifest
```

Every blob is named by the SHA-256 of its compressed bytes, providing content-addressable
storage consistent with the OCI Image Specification.

---

#### `ModelPackager.package`

```python
def package(self, model_dir: str, config: OCIConfig) -> str
```

Create an OCI-compliant tar archive from `model_dir`.

Recursively walks `model_dir` with `Path.rglob("*")`, creates one gzip-compressed tar layer
per file, writes all blobs to a temporary directory under `blobs/sha256/`, serializes the
`OCIConfig` and `OCIManifest`, then bundles everything into a single output tar.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model_dir` | `str` | Path to an existing directory containing model files |
| `config` | `OCIConfig` | Metadata config to embed in the archive |

**Returns**

`str` — Absolute path to the created tar archive. The filename is
`<config.model_name>-<config.version>.tar`, written to the parent directory of `model_dir`.

**Raises**

| Exception | Condition |
|-----------|-----------|
| `NotADirectoryError` | `model_dir` does not exist or is not a directory |

**Example**

```python
from aumai_modeloci.core import ModelPackager
from aumai_modeloci.models import OCIConfig

config = OCIConfig(
    model_name="bert-base",
    version="1.0.0",
    framework="pytorch",
    architecture="transformer",
)
packager = ModelPackager()
path = packager.package("/models/bert-base", config)
print(path)  # /models/bert-base-1.0.0.tar
```

---

#### `ModelPackager.create_manifest`

```python
def create_manifest(
    self, config: OCIConfig, layers: list[ModelLayer]
) -> OCIManifest
```

Build an `OCIManifest` from a config and a list of layer descriptors.

Serializes `config` to JSON, computes its SHA-256 digest, constructs the OCI config
descriptor, and assembles the manifest with the provided `layers`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `config` | `OCIConfig` | Model configuration to use as the manifest config descriptor |
| `layers` | `list[ModelLayer]` | Ordered list of layer descriptors |

**Returns**

`OCIManifest` — A manifest with `schema_version=2`, config descriptor, and layer list.

**Example**

```python
from aumai_modeloci.core import ModelPackager
from aumai_modeloci.models import ModelLayer, OCIConfig

config = OCIConfig(model_name="resnet", version="1.0.0", framework="onnx", architecture="cnn")
layers = [
    ModelLayer(digest="sha256:abc123", size=52_428_800,
               annotations={"org.opencontainers.image.title": "resnet50.onnx"})
]
packager = ModelPackager()
manifest = packager.create_manifest(config, layers)
print(manifest.schema_version)   # 2
print(len(manifest.layers))      # 1
```

---

#### `ModelPackager.add_layer`

```python
def add_layer(self, archive_path: str, file_path: str) -> ModelLayer
```

Add a single file as a new layer to an existing OCI tar archive.

Creates a gzip-compressed tar blob for `file_path`, appends it to the archive under
`blobs/sha256/<hex>`, and returns the `ModelLayer` descriptor. Note: this does not update
`manifest.json` inside the archive — use `create_manifest` to regenerate the manifest if
needed.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `archive_path` | `str` | Path to an existing OCI tar archive |
| `file_path` | `str` | Path to the file to add as a new layer |

**Returns**

`ModelLayer` — Descriptor for the newly added layer.

**Raises**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | `file_path` does not exist or is not a file |

**Example**

```python
from aumai_modeloci.core import ModelPackager

packager = ModelPackager()
layer = packager.add_layer("my-model-1.0.0.tar", "./extra-config.json")
print(layer.digest)   # sha256:...
print(layer.size)     # integer byte count
```

---

### class `ModelUnpackager`

Unpacks OCI-compliant model archives and verifies layer integrity.

---

#### `ModelUnpackager.unpack`

```python
def unpack(self, archive_path: str, output_dir: str) -> OCIConfig
```

Extract `archive_path` into `output_dir` and return the parsed `OCIConfig`.

On Python 3.12+ uses `tarfile.extractall(filter="data")` which rejects absolute paths and
`..` components. On older Python, manually validates every member path against the resolved
output directory before extraction.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `archive_path` | `str` | Path to an OCI tar archive |
| `output_dir` | `str` | Destination directory (created with `mkdir -p` if absent) |

**Returns**

`OCIConfig` — Parsed config from `config.json` at the archive root.

**Raises**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | `config.json` is not present in the archive |
| `ValueError` | A tar member attempted path traversal (Python < 3.12 path) |

**Example**

```python
from aumai_modeloci.core import ModelUnpackager

unpacker = ModelUnpackager()
config = unpacker.unpack("bert-base-1.0.0.tar", "/tmp/bert-out")
print(config.model_name)   # bert-base
print(config.version)      # 1.0.0
```

---

#### `ModelUnpackager.verify_layers`

```python
def verify_layers(self, archive_path: str) -> list[tuple[str, bool]]
```

Verify the SHA-256 digest of every layer blob in the archive.

Reads `manifest.json` to discover all expected layer digests, then for each layer:
locates the blob file at `blobs/sha256/<hex>`, reads its bytes in full, recomputes SHA-256,
and compares with the stored digest.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `archive_path` | `str` | Path to an OCI tar archive |

**Returns**

`list[tuple[str, bool]]` — One entry per layer: `(digest_string, is_valid)`. A layer is
valid when its stored blob content hashes to the same digest as its blob filename.

**Raises**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | `manifest.json` is not present in the archive |
| `RuntimeError` | `manifest.json` cannot be read from the archive |

**Example**

```python
from aumai_modeloci.core import ModelUnpackager

unpacker = ModelUnpackager()
results = unpacker.verify_layers("bert-base-1.0.0.tar")
for digest, is_valid in results:
    print("OK" if is_valid else "FAIL", digest)
all_ok = all(v for _, v in results)
```

---

## Module: `aumai_modeloci.models`

Public exports: `ModelLayer`, `OCIConfig`, `OCIManifest`

All models use Pydantic v2. They support `model_dump_json()`, `model_validate()`,
`model_copy()`, and the full Pydantic v2 API.

---

### class `ModelLayer`

A single layer in an OCI image, representing a compressed blob for one file.

```python
class ModelLayer(BaseModel):
    digest: str
    size: int
    media_type: str
    annotations: dict[str, str]
```

**Fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `digest` | `str` | required | Content digest in `sha256:<hex>` format |
| `size` | `int` | required | Uncompressed blob size in bytes |
| `media_type` | `str` | `"application/vnd.oci.image.layer.v1.tar+gzip"` | OCI media type for this layer |
| `annotations` | `dict[str, str]` | `{}` | OCI annotations; `org.opencontainers.image.title` holds the relative file path |

**Example**

```python
from aumai_modeloci.models import ModelLayer

layer = ModelLayer(
    digest="sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    size=1_048_576,
    annotations={"org.opencontainers.image.title": "pytorch_model.bin"},
)
print(layer.media_type)
# application/vnd.oci.image.layer.v1.tar+gzip
```

---

### class `OCIConfig`

Configuration metadata for a packaged ML model. Serialized as `config.json` inside every
archive. Follows the spirit of the OCI Image Configuration specification adapted for ML
artifacts.

```python
class OCIConfig(BaseModel):
    model_name: str
    version: str
    framework: str
    architecture: str
    metadata: dict[str, Any]
```

**Fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | required | Human-readable model name |
| `version` | `str` | required | Version string; semver recommended (`1.2.3`) |
| `framework` | `str` | required | ML framework: `pytorch`, `tensorflow`, `onnx`, `safetensors`, or any string |
| `architecture` | `str` | required | Architecture family: `transformer`, `cnn`, `rnn`, `custom`, etc. |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary additional metadata (base model, dataset, eval metrics, etc.) |

**Example**

```python
from aumai_modeloci.models import OCIConfig

config = OCIConfig(
    model_name="llm-support-bot",
    version="3.1.0",
    framework="pytorch",
    architecture="transformer",
    metadata={
        "base_model": "llama-3-8b",
        "fine_tune_dataset": "support-v4",
        "eval_accuracy": 0.913,
    },
)
print(config.model_dump_json(indent=2))
```

---

### class `OCIManifest`

OCI Image Manifest schema version 2. Follows the
[OCI Image Manifest Specification](https://github.com/opencontainers/image-spec/blob/main/manifest.md).

```python
class OCIManifest(BaseModel):
    schema_version: int
    media_type: str
    config: dict[str, Any]
    layers: list[dict[str, Any]]
```

**Fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `schema_version` | `int` | `2` | OCI manifest schema version |
| `media_type` | `str` | `"application/vnd.oci.image.manifest.v1+json"` | OCI manifest media type |
| `config` | `dict[str, Any]` | required | Config descriptor: `{"mediaType":..., "digest":..., "size":...}` |
| `layers` | `list[dict[str, Any]]` | `[]` | Layer descriptors, each with `mediaType`, `digest`, `size`, `annotations` |

**Manifest JSON example**

```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "config": {
    "mediaType": "application/vnd.oci.image.config.v1+json",
    "digest": "sha256:9e8f7a6b...",
    "size": 312
  },
  "layers": [
    {
      "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
      "digest": "sha256:a3b9c1d2...",
      "size": 4456789,
      "annotations": {
        "org.opencontainers.image.title": "pytorch_model.bin"
      }
    }
  ]
}
```

---

## Module: `aumai_modeloci`

```python
__version__: str  # e.g. "0.1.0"
```

---

## CLI Reference

The CLI is implemented in `aumai_modeloci.cli` using Click. The entry point is `aumai-modeloci`.

### `aumai-modeloci pack`

```
Usage: aumai-modeloci pack [OPTIONS]

  Package a model directory into an OCI-compliant tar archive.

Options:
  --model-dir PATH        Directory containing model files.  [required]
  --name TEXT             Model name.  [required]
  --version TEXT          Model version.  [required]
  --framework TEXT        ML framework.  [default: pytorch]
  --architecture TEXT     Model architecture.  [default: transformer]
  --metadata TEXT         Extra metadata as JSON string.  [default: {}]
  --version               Show version and exit.
  --help                  Show this message and exit.
```

### `aumai-modeloci unpack`

```
Usage: aumai-modeloci unpack [OPTIONS]

  Unpack an OCI model archive.

Options:
  --archive PATH     Path to the OCI tar archive.  [required]
  --output PATH      Directory to unpack into.  [required]
  --help             Show this message and exit.
```

### `aumai-modeloci inspect`

```
Usage: aumai-modeloci inspect [OPTIONS]

  Inspect an OCI model archive without extracting it.

Options:
  --archive PATH     Path to the OCI tar archive.  [required]
  --help             Show this message and exit.
```

Exit code 1 if any layer fails digest verification.

---

## Internal Helpers (not public API)

These are documented for reference but are not part of the stable public API:

| Symbol | Description |
|--------|-------------|
| `_sha256_file(path)` | Return `sha256:<hex>` for a file on disk (streamed in 64 KB chunks) |
| `_sha256_bytes(data)` | Return `sha256:<hex>` for an in-memory byte string |
| `_MANIFEST_FILENAME` | `"manifest.json"` |
| `_CONFIG_FILENAME` | `"config.json"` |
| `_LAYERS_DIR` | `"blobs/sha256"` |
