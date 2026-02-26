"""Core logic for aumai-modeloci."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .models import ModelLayer, OCIConfig, OCIManifest

__all__ = [
    "ModelPackager",
    "ModelUnpackager",
]

_MANIFEST_FILENAME = "manifest.json"
_CONFIG_FILENAME = "config.json"
_LAYERS_DIR = "blobs/sha256"


def _sha256_file(path: str) -> str:
    """Return 'sha256:<hex>' digest for the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _sha256_bytes(data: bytes) -> str:
    """Return 'sha256:<hex>' digest for *data*."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


class ModelPackager:
    """
    Packages an ML model directory into an OCI-compliant tar archive.

    Archive layout::

        blobs/sha256/<hex>      # layer tarballs
        config.json             # OCIConfig (JSON)
        manifest.json           # OCIManifest (JSON)
    """

    def package(self, model_dir: str, config: OCIConfig) -> str:
        """
        Create an OCI-compliant tar archive from *model_dir*.

        Returns the path to the created archive.
        """
        model_path = Path(model_dir)
        if not model_path.is_dir():
            raise NotADirectoryError(f"{model_dir!r} is not a directory.")

        output_archive = str(
            model_path.parent / f"{config.model_name}-{config.version}.tar"
        )

        layers: list[ModelLayer] = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            blobs_dir = tmp / _LAYERS_DIR
            blobs_dir.mkdir(parents=True)

            # Walk model_dir and create one layer per file
            for file_path in sorted(model_path.rglob("*")):
                if file_path.is_file():
                    layer = self._create_layer_blob(
                        str(file_path), str(blobs_dir), str(model_path)
                    )
                    layers.append(layer)

            # Write config
            config_bytes = config.model_dump_json(indent=2).encode("utf-8")
            config_digest = _sha256_bytes(config_bytes)
            config_blob_path = blobs_dir / config_digest.split(":")[1]
            config_blob_path.write_bytes(config_bytes)

            # Build manifest
            manifest = self.create_manifest(config, layers)
            manifest_json = manifest.model_dump_json(indent=2)
            (tmp / _MANIFEST_FILENAME).write_text(manifest_json, encoding="utf-8")

            # Write human-readable config.json at root for convenience
            (tmp / _CONFIG_FILENAME).write_bytes(config_bytes)

            # Bundle everything into the output tar
            with tarfile.open(output_archive, "w") as tar:
                tar.add(str(tmp), arcname="")

        return output_archive

    def create_manifest(
        self, config: OCIConfig, layers: list[ModelLayer]
    ) -> OCIManifest:
        """Build an OCIManifest from config and layer list."""
        config_bytes = config.model_dump_json().encode("utf-8")
        config_digest = _sha256_bytes(config_bytes)

        config_descriptor: dict[str, Any] = {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "digest": config_digest,
            "size": len(config_bytes),
        }
        layer_descriptors: list[dict[str, Any]] = [
            {
                "mediaType": layer.media_type,
                "digest": layer.digest,
                "size": layer.size,
                "annotations": layer.annotations,
            }
            for layer in layers
        ]
        return OCIManifest(
            config=config_descriptor,
            layers=layer_descriptors,
        )

    def add_layer(
        self, archive_path: str, file_path: str
    ) -> ModelLayer:
        """
        Add a file as a new layer to an existing archive.

        Returns a ``ModelLayer`` descriptor for the added file.
        """
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"File not found: {file_path!r}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            blobs_dir = tmp / _LAYERS_DIR
            blobs_dir.mkdir(parents=True)

            # Create a layer blob for the new file
            layer = self._create_layer_blob(file_path, str(blobs_dir), "")

            # Append blob to the archive
            with tarfile.open(archive_path, "a") as tar:
                blob_name = layer.digest.split(":")[1]
                tar.add(
                    str(blobs_dir / blob_name),
                    arcname=f"{_LAYERS_DIR}/{blob_name}",
                )

        return layer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_layer_blob(
        self, file_path: str, blobs_dir: str, base_dir: str
    ) -> ModelLayer:
        """
        Create a gzip-compressed tar layer from a single file.

        The layer blob is written to *blobs_dir* with its SHA-256 as filename.
        """
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as layer_tar:
            arcname = (
                os.path.relpath(file_path, base_dir) if base_dir else Path(file_path).name
            )
            layer_tar.add(file_path, arcname=arcname)
        blob_bytes = buf.getvalue()
        digest = _sha256_bytes(blob_bytes)
        hex_digest = digest.split(":")[1]
        blob_path = Path(blobs_dir) / hex_digest
        blob_path.write_bytes(blob_bytes)
        rel_path = os.path.relpath(file_path, base_dir) if base_dir else Path(file_path).name
        return ModelLayer(
            digest=digest,
            size=len(blob_bytes),
            annotations={"org.opencontainers.image.title": str(rel_path)},
        )


class ModelUnpackager:
    """
    Unpacks OCI-compliant model archives and verifies layer integrity.
    """

    def unpack(self, archive_path: str, output_dir: str) -> OCIConfig:
        """
        Extract *archive_path* into *output_dir*.

        Reads ``config.json`` from the archive root and returns the
        parsed ``OCIConfig``.
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(path=str(output))

        config_file = output / _CONFIG_FILENAME
        if not config_file.exists():
            raise FileNotFoundError(
                f"config.json not found in archive {archive_path!r}."
            )
        return OCIConfig.model_validate(
            json.loads(config_file.read_text())
        )

    def verify_layers(
        self, archive_path: str
    ) -> list[tuple[str, bool]]:
        """
        Verify the SHA-256 digest of every layer blob in the archive.

        Returns a list of (digest, is_valid) tuples.  A layer is valid
        when its stored filename matches the SHA-256 of its content.
        """
        results: list[tuple[str, bool]] = []

        with tarfile.open(archive_path, "r") as tar:
            members = {m.name: m for m in tar.getmembers()}

            # Read manifest to know which digests to expect
            manifest_member = members.get(_MANIFEST_FILENAME)
            if manifest_member is None:
                raise FileNotFoundError(
                    f"manifest.json not found in {archive_path!r}."
                )
            manifest_fh = tar.extractfile(manifest_member)
            if manifest_fh is None:
                raise RuntimeError("Could not read manifest.json from archive.")
            manifest = OCIManifest.model_validate(
                json.loads(manifest_fh.read().decode("utf-8"))
            )

            for layer_desc in manifest.layers:
                digest: str = layer_desc["digest"]
                hex_digest = digest.split(":")[1]
                blob_name = f"{_LAYERS_DIR}/{hex_digest}"
                blob_member = members.get(blob_name)
                if blob_member is None:
                    results.append((digest, False))
                    continue
                blob_fh = tar.extractfile(blob_member)
                if blob_fh is None:
                    results.append((digest, False))
                    continue
                actual_digest = _sha256_bytes(blob_fh.read())
                results.append((digest, actual_digest == digest))

        return results
