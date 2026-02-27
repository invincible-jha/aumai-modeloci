"""Shared test fixtures for aumai-modeloci."""

from __future__ import annotations

from pathlib import Path

import pytest

from aumai_modeloci.core import ModelPackager, ModelUnpackager
from aumai_modeloci.models import OCIConfig


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_config() -> OCIConfig:
    return OCIConfig(
        model_name="test-model",
        version="1.0.0",
        framework="pytorch",
        architecture="transformer",
        metadata={"author": "test"},
    )


@pytest.fixture()
def minimal_config() -> OCIConfig:
    return OCIConfig(
        model_name="minimal",
        version="0.1",
        framework="onnx",
        architecture="cnn",
    )


# ---------------------------------------------------------------------------
# Model directory with sample files
# ---------------------------------------------------------------------------


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    """A temp directory with a few fake model files."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "weights.bin").write_bytes(b"\x00\x01\x02\x03" * 256)
    (d / "config.json").write_text('{"hidden_size": 768}', encoding="utf-8")
    sub = d / "tokenizer"
    sub.mkdir()
    (sub / "vocab.txt").write_text("hello\nworld\n", encoding="utf-8")
    return d


@pytest.fixture()
def single_file(tmp_path: Path) -> Path:
    """A single fake model file."""
    f = tmp_path / "model.onnx"
    f.write_bytes(b"onnx_model_data" * 100)
    return f


# ---------------------------------------------------------------------------
# Core objects
# ---------------------------------------------------------------------------


@pytest.fixture()
def packager() -> ModelPackager:
    return ModelPackager()


@pytest.fixture()
def unpacker() -> ModelUnpackager:
    return ModelUnpackager()


# ---------------------------------------------------------------------------
# Pre-built archive fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def packed_archive(
    packager: ModelPackager,
    model_dir: Path,
    sample_config: OCIConfig,
) -> str:
    """Return the path to a packed OCI archive."""
    return packager.package(str(model_dir), sample_config)
