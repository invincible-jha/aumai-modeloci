"""Tests for aumai_modeloci.core."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from aumai_modeloci.core import ModelPackager, ModelUnpackager, _sha256_bytes, _sha256_file
from aumai_modeloci.models import ModelLayer, OCIConfig, OCIManifest


# ---------------------------------------------------------------------------
# _sha256_bytes / _sha256_file
# ---------------------------------------------------------------------------


class TestSha256Helpers:
    def test_sha256_bytes_format(self) -> None:
        digest = _sha256_bytes(b"hello")
        assert digest.startswith("sha256:")
        assert len(digest) == len("sha256:") + 64

    def test_sha256_bytes_known_value(self) -> None:
        expected = "sha256:" + hashlib.sha256(b"hello").hexdigest()
        assert _sha256_bytes(b"hello") == expected

    def test_sha256_file_matches_bytes(self, tmp_path: Path) -> None:
        data = b"test file content"
        f = tmp_path / "test.bin"
        f.write_bytes(data)
        assert _sha256_file(str(f)) == _sha256_bytes(data)

    def test_sha256_file_format(self, tmp_path: Path) -> None:
        f = tmp_path / "f.bin"
        f.write_bytes(b"data")
        digest = _sha256_file(str(f))
        assert digest.startswith("sha256:")
        assert len(digest) == len("sha256:") + 64


# ---------------------------------------------------------------------------
# ModelLayer / OCIConfig / OCIManifest models
# ---------------------------------------------------------------------------


class TestModels:
    def test_model_layer_defaults(self) -> None:
        layer = ModelLayer(digest="sha256:" + "a" * 64, size=100)
        assert layer.media_type == "application/vnd.oci.image.layer.v1.tar+gzip"
        assert layer.annotations == {}

    def test_oci_config_fields(self, sample_config: OCIConfig) -> None:
        assert sample_config.model_name == "test-model"
        assert sample_config.version == "1.0.0"
        assert sample_config.framework == "pytorch"
        assert sample_config.architecture == "transformer"
        assert sample_config.metadata == {"author": "test"}

    def test_oci_manifest_defaults(self) -> None:
        manifest = OCIManifest(config={"digest": "sha256:abc", "size": 10})
        assert manifest.schema_version == 2
        assert manifest.layers == []

    def test_oci_config_serializes_to_json(self, sample_config: OCIConfig) -> None:
        data = json.loads(sample_config.model_dump_json())
        assert data["model_name"] == "test-model"
        assert data["framework"] == "pytorch"


# ---------------------------------------------------------------------------
# ModelPackager
# ---------------------------------------------------------------------------


class TestModelPackager:
    def test_package_returns_tar_path(
        self,
        packager: ModelPackager,
        model_dir: Path,
        sample_config: OCIConfig,
    ) -> None:
        archive = packager.package(str(model_dir), sample_config)
        assert archive.endswith(".tar")
        assert Path(archive).exists()

    def test_archive_name_contains_model_and_version(
        self,
        packager: ModelPackager,
        model_dir: Path,
        sample_config: OCIConfig,
    ) -> None:
        archive = packager.package(str(model_dir), sample_config)
        name = Path(archive).name
        assert sample_config.model_name in name
        assert sample_config.version in name

    def test_archive_contains_manifest_json(
        self,
        packed_archive: str,
    ) -> None:
        with tarfile.open(packed_archive, "r") as tar:
            names = [m.name for m in tar.getmembers()]
        assert "manifest.json" in names

    def test_archive_contains_config_json(
        self,
        packed_archive: str,
    ) -> None:
        with tarfile.open(packed_archive, "r") as tar:
            names = [m.name for m in tar.getmembers()]
        assert "config.json" in names

    def test_archive_contains_blobs_directory(
        self,
        packed_archive: str,
    ) -> None:
        with tarfile.open(packed_archive, "r") as tar:
            names = [m.name for m in tar.getmembers()]
        blob_entries = [n for n in names if "blobs/sha256" in n]
        assert len(blob_entries) > 0

    def test_config_json_in_archive_matches_input(
        self,
        packed_archive: str,
        sample_config: OCIConfig,
    ) -> None:
        with tarfile.open(packed_archive, "r") as tar:
            members = {m.name: m for m in tar.getmembers()}
            fh = tar.extractfile(members["config.json"])
            assert fh is not None
            cfg = OCIConfig.model_validate(json.loads(fh.read()))
        assert cfg.model_name == sample_config.model_name
        assert cfg.version == sample_config.version
        assert cfg.framework == sample_config.framework

    def test_manifest_has_layers_for_each_file(
        self,
        packed_archive: str,
        model_dir: Path,
    ) -> None:
        file_count = sum(1 for f in model_dir.rglob("*") if f.is_file())
        with tarfile.open(packed_archive, "r") as tar:
            members = {m.name: m for m in tar.getmembers()}
            fh = tar.extractfile(members["manifest.json"])
            assert fh is not None
            manifest = OCIManifest.model_validate(json.loads(fh.read()))
        assert len(manifest.layers) == file_count

    def test_package_raises_for_non_directory(
        self, packager: ModelPackager, tmp_path: Path, sample_config: OCIConfig
    ) -> None:
        not_a_dir = tmp_path / "notadir.bin"
        not_a_dir.write_bytes(b"x")
        with pytest.raises(NotADirectoryError):
            packager.package(str(not_a_dir), sample_config)

    def test_package_empty_directory(
        self, packager: ModelPackager, tmp_path: Path, sample_config: OCIConfig
    ) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        archive = packager.package(str(empty_dir), sample_config)
        assert Path(archive).exists()

    def test_create_manifest_has_correct_media_type(
        self,
        packager: ModelPackager,
        sample_config: OCIConfig,
    ) -> None:
        layer = ModelLayer(
            digest="sha256:" + "a" * 64,
            size=100,
            annotations={"org.opencontainers.image.title": "weights.bin"},
        )
        manifest = packager.create_manifest(sample_config, [layer])
        assert manifest.schema_version == 2
        assert "application/vnd.oci.image.config" in manifest.config["mediaType"]
        assert len(manifest.layers) == 1

    def test_create_manifest_layer_descriptor_fields(
        self,
        packager: ModelPackager,
        sample_config: OCIConfig,
    ) -> None:
        layer = ModelLayer(
            digest="sha256:" + "b" * 64,
            size=512,
            annotations={"org.opencontainers.image.title": "model.bin"},
        )
        manifest = packager.create_manifest(sample_config, [layer])
        desc = manifest.layers[0]
        assert desc["digest"] == layer.digest
        assert desc["size"] == 512
        assert "annotations" in desc

    def test_add_layer_returns_model_layer(
        self,
        packager: ModelPackager,
        packed_archive: str,
        tmp_path: Path,
    ) -> None:
        extra_file = tmp_path / "extra.bin"
        extra_file.write_bytes(b"extra content" * 10)
        layer = packager.add_layer(packed_archive, str(extra_file))
        assert isinstance(layer, ModelLayer)
        assert layer.digest.startswith("sha256:")
        assert layer.size > 0

    def test_add_layer_missing_file_raises(
        self,
        packager: ModelPackager,
        packed_archive: str,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            packager.add_layer(packed_archive, "/nonexistent/file.bin")

    def test_layer_blob_digest_matches_content(
        self,
        packager: ModelPackager,
        packed_archive: str,
    ) -> None:
        """Every blob in the archive must have a name matching its sha256."""
        with tarfile.open(packed_archive, "r") as tar:
            for member in tar.getmembers():
                if "blobs/sha256/" in member.name:
                    hex_digest = member.name.split("/")[-1]
                    fh = tar.extractfile(member)
                    if fh:
                        actual = hashlib.sha256(fh.read()).hexdigest()
                        assert actual == hex_digest, (
                            f"Blob {hex_digest!r} digest mismatch"
                        )


# ---------------------------------------------------------------------------
# ModelUnpackager
# ---------------------------------------------------------------------------


class TestModelUnpackager:
    def test_unpack_returns_oci_config(
        self,
        unpacker: ModelUnpackager,
        packed_archive: str,
        tmp_path: Path,
        sample_config: OCIConfig,
    ) -> None:
        out_dir = str(tmp_path / "unpacked")
        config = unpacker.unpack(packed_archive, out_dir)
        assert isinstance(config, OCIConfig)
        assert config.model_name == sample_config.model_name

    def test_unpack_creates_output_directory(
        self,
        unpacker: ModelUnpackager,
        packed_archive: str,
        tmp_path: Path,
    ) -> None:
        out_dir = tmp_path / "new" / "nested" / "dir"
        unpacker.unpack(packed_archive, str(out_dir))
        assert out_dir.exists()

    def test_unpack_writes_config_json(
        self,
        unpacker: ModelUnpackager,
        packed_archive: str,
        tmp_path: Path,
    ) -> None:
        out_dir = str(tmp_path / "unpacked")
        unpacker.unpack(packed_archive, out_dir)
        assert (Path(out_dir) / "config.json").exists()

    def test_unpack_missing_config_raises(
        self,
        unpacker: ModelUnpackager,
        tmp_path: Path,
    ) -> None:
        # Create an archive with no config.json
        bad_archive = str(tmp_path / "bad.tar")
        with tarfile.open(bad_archive, "w") as tar:
            dummy = tmp_path / "dummy.txt"
            dummy.write_text("x")
            tar.add(str(dummy), arcname="dummy.txt")
        with pytest.raises(FileNotFoundError, match="config.json"):
            unpacker.unpack(bad_archive, str(tmp_path / "out"))

    def test_verify_layers_returns_list(
        self,
        unpacker: ModelUnpackager,
        packed_archive: str,
    ) -> None:
        results = unpacker.verify_layers(packed_archive)
        assert isinstance(results, list)

    def test_verify_layers_all_valid(
        self,
        unpacker: ModelUnpackager,
        packed_archive: str,
    ) -> None:
        results = unpacker.verify_layers(packed_archive)
        assert len(results) > 0
        for digest, valid in results:
            assert valid, f"Layer {digest} failed verification"

    def test_verify_layers_missing_manifest_raises(
        self,
        unpacker: ModelUnpackager,
        tmp_path: Path,
    ) -> None:
        bad_archive = str(tmp_path / "no_manifest.tar")
        with tarfile.open(bad_archive, "w") as tar:
            dummy = tmp_path / "dummy.txt"
            dummy.write_text("x")
            tar.add(str(dummy), arcname="dummy.txt")
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            unpacker.verify_layers(bad_archive)

    def test_unpack_then_repack_produces_valid_archive(
        self,
        packager: ModelPackager,
        unpacker: ModelUnpackager,
        packed_archive: str,
        tmp_path: Path,
        sample_config: OCIConfig,
    ) -> None:
        out_dir = tmp_path / "unpacked"
        config = unpacker.unpack(packed_archive, str(out_dir))
        assert config.model_name == sample_config.model_name
        # Verify the unpacked archive was also valid
        results = unpacker.verify_layers(packed_archive)
        for _, valid in results:
            assert valid
