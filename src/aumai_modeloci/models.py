"""Pydantic models for aumai-modeloci."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "ModelLayer",
    "OCIConfig",
    "OCIManifest",
]


class ModelLayer(BaseModel):
    """A single layer in an OCI image, representing a file blob."""

    digest: str            # sha256:<hex>
    size: int              # bytes
    media_type: str = "application/vnd.oci.image.layer.v1.tar+gzip"
    annotations: dict[str, str] = Field(default_factory=dict)


class OCIConfig(BaseModel):
    """Configuration metadata for a packaged ML model."""

    model_name: str
    version: str
    framework: str         # e.g. "pytorch", "tensorflow", "onnx", "safetensors"
    architecture: str      # e.g. "transformer", "cnn", "custom"
    metadata: dict[str, Any] = Field(default_factory=dict)


class OCIManifest(BaseModel):
    """
    OCI Image Manifest (schema version 2).

    Follows the OCI Image Manifest Specification
    https://github.com/opencontainers/image-spec/blob/main/manifest.md
    """

    schema_version: int = 2
    media_type: str = "application/vnd.oci.image.manifest.v1+json"
    config: dict[str, Any]
    layers: list[dict[str, Any]] = Field(default_factory=list)
