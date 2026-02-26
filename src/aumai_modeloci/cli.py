"""CLI entry point for aumai-modeloci."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .core import ModelPackager, ModelUnpackager
from .models import OCIConfig


@click.group()
@click.version_option()
def main() -> None:
    """AumAI ModelOCI — OCI-compliant packaging for ML models."""


@main.command("pack")
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing the model files.",
)
@click.option("--name", required=True, help="Model name.")
@click.option("--version", "model_version", required=True, help="Model version.")
@click.option(
    "--framework",
    default="pytorch",
    show_default=True,
    help="ML framework (e.g. pytorch, tensorflow, onnx).",
)
@click.option(
    "--architecture",
    default="transformer",
    show_default=True,
    help="Model architecture description.",
)
@click.option(
    "--metadata",
    "metadata_json",
    default="{}",
    help="Extra metadata as JSON string.",
)
def pack_command(
    model_dir: str,
    name: str,
    model_version: str,
    framework: str,
    architecture: str,
    metadata_json: str,
) -> None:
    """Package a model directory into an OCI-compliant tar archive."""
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        click.echo(f"Error: invalid JSON for --metadata: {exc}", err=True)
        sys.exit(1)

    config = OCIConfig(
        model_name=name,
        version=model_version,
        framework=framework,
        architecture=architecture,
        metadata=metadata,
    )
    packager = ModelPackager()
    try:
        archive_path = packager.package(model_dir, config)
    except (NotADirectoryError, FileNotFoundError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Packaged model: {archive_path}")
    click.echo(f"  Name        : {name}")
    click.echo(f"  Version     : {model_version}")
    click.echo(f"  Framework   : {framework}")
    click.echo(f"  Archive     : {archive_path}")


@main.command("unpack")
@click.option(
    "--archive",
    "archive_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the OCI tar archive.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to unpack into.",
)
def unpack_command(archive_path: str, output_dir: str) -> None:
    """Unpack an OCI model archive."""
    unpacker = ModelUnpackager()
    try:
        config = unpacker.unpack(archive_path, output_dir)
    except (FileNotFoundError, Exception) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Unpacked to: {output_dir}")
    click.echo(f"  Model   : {config.model_name} v{config.version}")
    click.echo(f"  Framework: {config.framework}")


@main.command("inspect")
@click.option(
    "--archive",
    "archive_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the OCI tar archive.",
)
def inspect_command(archive_path: str) -> None:
    """Inspect an OCI model archive without extracting it."""
    import tarfile

    try:
        with tarfile.open(archive_path, "r") as tar:
            members = {m.name: m for m in tar.getmembers()}

            # Read config
            cfg_member = members.get("config.json")
            if cfg_member:
                fh = tar.extractfile(cfg_member)
                if fh:
                    config = OCIConfig.model_validate(json.loads(fh.read()))
                    click.echo(f"Model    : {config.model_name}")
                    click.echo(f"Version  : {config.version}")
                    click.echo(f"Framework: {config.framework}")
                    click.echo(f"Arch     : {config.architecture}")
                    if config.metadata:
                        click.echo(f"Metadata : {json.dumps(config.metadata)}")

            # Read manifest
            man_member = members.get("manifest.json")
            if man_member:
                fh = tar.extractfile(man_member)
                if fh:
                    from .models import OCIManifest
                    manifest = OCIManifest.model_validate(json.loads(fh.read()))
                    click.echo(f"\nLayers ({len(manifest.layers)}):")
                    for layer in manifest.layers:
                        title = layer.get("annotations", {}).get(
                            "org.opencontainers.image.title", "(unknown)"
                        )
                        size_kb = layer.get("size", 0) / 1024
                        click.echo(
                            f"  {title:<40}  {size_kb:6.1f} KB  "
                            f"{layer.get('digest', '')[:23]}..."
                        )

            # Verify layers
            unpacker = ModelUnpackager()
            verification = unpacker.verify_layers(archive_path)
            click.echo(f"\nLayer verification ({len(verification)} layers):")
            all_valid = True
            for digest, valid in verification:
                status = "OK" if valid else "FAIL"
                if not valid:
                    all_valid = False
                click.echo(f"  {status}  {digest[:30]}...")
            if all_valid:
                click.echo("All layers verified.")
            else:
                click.echo("WARNING: some layers failed verification!", err=True)
                sys.exit(1)

    except Exception as exc:
        click.echo(f"Error inspecting archive: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
