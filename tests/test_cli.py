"""Tests for aumai_modeloci CLI."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from aumai_modeloci.cli import main
from aumai_modeloci.models import OCIConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    (d / "weights.bin").write_bytes(b"\xDE\xAD\xBE\xEF" * 64)
    (d / "config.json").write_text('{"layers": 12}', encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


class TestVersionFlag:
    def test_version_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


# ---------------------------------------------------------------------------
# pack command
# ---------------------------------------------------------------------------


class TestPackCommand:
    def test_pack_creates_archive(self, tmp_path: Path) -> None:
        model_d = _make_model_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "pack",
                "--model-dir", str(model_d),
                "--name", "my-model",
                "--version", "1.0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Packaged model" in result.output

    def test_pack_output_mentions_name_and_version(
        self, tmp_path: Path
    ) -> None:
        model_d = _make_model_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "pack",
                "--model-dir", str(model_d),
                "--name", "bert-base",
                "--version", "2.0",
            ],
        )
        assert result.exit_code == 0
        assert "bert-base" in result.output
        assert "2.0" in result.output

    def test_pack_custom_framework_and_arch(self, tmp_path: Path) -> None:
        model_d = _make_model_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "pack",
                "--model-dir", str(model_d),
                "--name", "resnet",
                "--version", "0.5",
                "--framework", "tensorflow",
                "--architecture", "cnn",
            ],
        )
        assert result.exit_code == 0
        assert "tensorflow" in result.output

    def test_pack_with_metadata(self, tmp_path: Path) -> None:
        model_d = _make_model_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "pack",
                "--model-dir", str(model_d),
                "--name", "mymodel",
                "--version", "1.0",
                "--metadata", '{"author": "alice"}',
            ],
        )
        assert result.exit_code == 0

    def test_pack_invalid_metadata_json_fails(self, tmp_path: Path) -> None:
        model_d = _make_model_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "pack",
                "--model-dir", str(model_d),
                "--name", "m",
                "--version", "1",
                "--metadata", "not-json",
            ],
        )
        assert result.exit_code != 0
        assert "invalid JSON" in result.output

    def test_pack_missing_name_fails(self, tmp_path: Path) -> None:
        model_d = _make_model_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["pack", "--model-dir", str(model_d), "--version", "1.0"],
        )
        assert result.exit_code != 0

    def test_pack_nonexistent_dir_fails(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "pack",
                "--model-dir", str(tmp_path / "no_such_dir"),
                "--name", "m",
                "--version", "1",
            ],
        )
        assert result.exit_code != 0

    def test_pack_produces_valid_tar(self, tmp_path: Path) -> None:
        model_d = _make_model_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "pack",
                "--model-dir", str(model_d),
                "--name", "mymodel",
                "--version", "1.0",
            ],
        )
        assert result.exit_code == 0
        # Extract archive path from output — line looks like "  Archive     : /some/path.tar"
        # Split on ":" but rejoin all parts after the first colon (handles Windows drive letters)
        archive_line = [ln for ln in result.output.splitlines() if "Archive" in ln]
        assert archive_line
        after_colon = archive_line[0].split(":", 1)[1].strip()
        assert tarfile.is_tarfile(after_colon)


# ---------------------------------------------------------------------------
# unpack command
# ---------------------------------------------------------------------------


class TestUnpackCommand:
    def _build_archive(self, tmp_path: Path) -> Path:
        from aumai_modeloci.core import ModelPackager
        from aumai_modeloci.models import OCIConfig

        model_d = _make_model_dir(tmp_path)
        config = OCIConfig(
            model_name="test",
            version="1.0",
            framework="pytorch",
            architecture="transformer",
        )
        packager = ModelPackager()
        archive = packager.package(str(model_d), config)
        return Path(archive)

    def test_unpack_extracts_to_output_dir(self, tmp_path: Path) -> None:
        archive = self._build_archive(tmp_path)
        out_dir = str(tmp_path / "out")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["unpack", "--archive", str(archive), "--output", out_dir],
        )
        assert result.exit_code == 0, result.output
        assert "Unpacked to" in result.output
        assert (Path(out_dir) / "config.json").exists()

    def test_unpack_shows_model_info(self, tmp_path: Path) -> None:
        archive = self._build_archive(tmp_path)
        out_dir = str(tmp_path / "out")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["unpack", "--archive", str(archive), "--output", out_dir],
        )
        assert "test" in result.output
        assert "pytorch" in result.output

    def test_unpack_missing_archive_fails(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "unpack",
                "--archive", str(tmp_path / "nope.tar"),
                "--output", str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# inspect command
# ---------------------------------------------------------------------------


class TestInspectCommand:
    def _build_archive(self, tmp_path: Path) -> Path:
        from aumai_modeloci.core import ModelPackager
        from aumai_modeloci.models import OCIConfig

        model_d = _make_model_dir(tmp_path)
        config = OCIConfig(
            model_name="inspect-model",
            version="3.0",
            framework="onnx",
            architecture="cnn",
            metadata={"info": "test"},
        )
        packager = ModelPackager()
        archive = packager.package(str(model_d), config)
        return Path(archive)

    def test_inspect_shows_model_info(self, tmp_path: Path) -> None:
        archive = self._build_archive(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main, ["inspect", "--archive", str(archive)]
        )
        assert result.exit_code == 0, result.output
        assert "inspect-model" in result.output
        assert "3.0" in result.output
        assert "onnx" in result.output

    def test_inspect_shows_layers(self, tmp_path: Path) -> None:
        archive = self._build_archive(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main, ["inspect", "--archive", str(archive)]
        )
        assert "Layers" in result.output

    def test_inspect_shows_verification(self, tmp_path: Path) -> None:
        archive = self._build_archive(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main, ["inspect", "--archive", str(archive)]
        )
        assert "verified" in result.output.lower() or "OK" in result.output

    def test_inspect_nonexistent_archive_fails(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["inspect", "--archive", str(tmp_path / "nope.tar")],
        )
        assert result.exit_code != 0
