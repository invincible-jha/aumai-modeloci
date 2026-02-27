"""
aumai-modeloci quickstart — working demo of pack, inspect, unpack, and verify.

Run directly:

    python examples/quickstart.py

All demos use a temporary directory and clean up after themselves.
"""

from __future__ import annotations

import json
import pathlib
import tempfile


# ---------------------------------------------------------------------------
# Demo 1: Pack a model directory into an OCI archive
# ---------------------------------------------------------------------------

def demo_pack_model() -> str:
    """Create a toy model directory and pack it into an OCI tar archive."""
    print("\n=== Demo 1: Pack a model directory ===")

    from aumai_modeloci.core import ModelPackager
    from aumai_modeloci.models import OCIConfig

    # Create a temporary model directory with fake model files
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = pathlib.Path(tmp) / "my-classifier"
        model_dir.mkdir()

        # Write fake model files (stand-ins for real weights / tokenizer / config)
        (model_dir / "pytorch_model.bin").write_bytes(b"\x00" * 2048)
        (model_dir / "config.json").write_text(
            json.dumps({"hidden_size": 768, "num_labels": 3}), encoding="utf-8"
        )
        (model_dir / "tokenizer.json").write_text(
            json.dumps({"vocab_size": 30522}), encoding="utf-8"
        )
        print(f"  Created model directory: {model_dir}")
        print(f"  Files: {[f.name for f in model_dir.iterdir()]}")

        # Build the OCIConfig
        config = OCIConfig(
            model_name="my-classifier",
            version="1.0.0",
            framework="pytorch",
            architecture="transformer",
            metadata={
                "task": "text-classification",
                "language": "en",
                "accuracy": 0.936,
            },
        )
        print(f"\n  Config model_name : {config.model_name}")
        print(f"  Config version    : {config.version}")
        print(f"  Config framework  : {config.framework}")
        print(f"  Config metadata   : {config.metadata}")

        # Pack
        packager = ModelPackager()
        archive_path = packager.package(str(model_dir), config)
        archive = pathlib.Path(archive_path)
        print(f"\n  Archive created   : {archive.name}")
        print(f"  Archive size      : {archive.stat().st_size:,} bytes")

        # Move archive outside the tmp dir before it is deleted
        import shutil
        final_path = pathlib.Path(tempfile.gettempdir()) / archive.name
        shutil.move(str(archive), str(final_path))

    print(f"\n  Final archive     : {final_path}")
    return str(final_path)


# ---------------------------------------------------------------------------
# Demo 2: Inspect an archive without extracting
# ---------------------------------------------------------------------------

def demo_inspect_archive(archive_path: str) -> None:
    """Inspect the manifest and config of an OCI archive without extracting."""
    print("\n=== Demo 2: Inspect archive ===")

    import tarfile
    from aumai_modeloci.models import OCIConfig, OCIManifest

    with tarfile.open(archive_path, "r") as tar:
        members = {m.name: m for m in tar.getmembers()}

        # Read config
        cfg_member = members.get("config.json")
        if cfg_member:
            fh = tar.extractfile(cfg_member)
            if fh:
                config = OCIConfig.model_validate(json.loads(fh.read()))
                print(f"  Model     : {config.model_name}")
                print(f"  Version   : {config.version}")
                print(f"  Framework : {config.framework}")
                print(f"  Arch      : {config.architecture}")
                print(f"  Metadata  : {config.metadata}")

        # Read manifest
        man_member = members.get("manifest.json")
        if man_member:
            fh = tar.extractfile(man_member)
            if fh:
                manifest = OCIManifest.model_validate(json.loads(fh.read()))
                print(f"\n  Schema version : {manifest.schema_version}")
                print(f"  Layers ({len(manifest.layers)}):")
                for layer in manifest.layers:
                    title = layer.get("annotations", {}).get(
                        "org.opencontainers.image.title", "(unknown)"
                    )
                    size_kb = layer.get("size", 0) / 1024
                    digest_short = layer.get("digest", "")[:30]
                    print(f"    {title:<30}  {size_kb:6.2f} KB  {digest_short}...")


# ---------------------------------------------------------------------------
# Demo 3: Verify layer integrity
# ---------------------------------------------------------------------------

def demo_verify_layers(archive_path: str) -> None:
    """Verify the SHA-256 digest of every layer blob in the archive."""
    print("\n=== Demo 3: Verify layer integrity ===")

    from aumai_modeloci.core import ModelUnpackager

    unpacker = ModelUnpackager()
    results = unpacker.verify_layers(archive_path)

    all_valid = True
    for digest, is_valid in results:
        status = "OK  " if is_valid else "FAIL"
        if not is_valid:
            all_valid = False
        print(f"  {status}  {digest[:40]}...")

    if all_valid:
        print(f"\n  All {len(results)} layer(s) verified successfully.")
    else:
        print("\n  WARNING: one or more layers failed verification!")


# ---------------------------------------------------------------------------
# Demo 4: Unpack an archive
# ---------------------------------------------------------------------------

def demo_unpack_archive(archive_path: str) -> None:
    """Extract an OCI archive and read the returned OCIConfig."""
    print("\n=== Demo 4: Unpack archive ===")

    from aumai_modeloci.core import ModelUnpackager

    with tempfile.TemporaryDirectory() as output_dir:
        unpacker = ModelUnpackager()
        config = unpacker.unpack(archive_path, output_dir)

        print(f"  Unpacked to  : {output_dir}")
        print(f"  Model name   : {config.model_name}")
        print(f"  Version      : {config.version}")
        print(f"  Framework    : {config.framework}")

        extracted_files = list(pathlib.Path(output_dir).rglob("*"))
        model_files = [f for f in extracted_files if f.is_file() and "blobs" not in str(f)]
        print(f"  Model files  : {[f.name for f in model_files]}")


# ---------------------------------------------------------------------------
# Demo 5: Build a manifest programmatically
# ---------------------------------------------------------------------------

def demo_build_manifest() -> None:
    """Construct an OCIManifest from manually created ModelLayer objects."""
    print("\n=== Demo 5: Build manifest programmatically ===")

    from aumai_modeloci.core import ModelPackager
    from aumai_modeloci.models import ModelLayer, OCIConfig

    config = OCIConfig(
        model_name="onnx-resnet",
        version="2.0.0",
        framework="onnx",
        architecture="cnn",
        metadata={"input_shape": [1, 3, 224, 224]},
    )

    # Simulate externally computed layer descriptors
    layers = [
        ModelLayer(
            digest="sha256:" + "a" * 64,
            size=52_428_800,
            annotations={"org.opencontainers.image.title": "resnet50.onnx"},
        ),
        ModelLayer(
            digest="sha256:" + "b" * 64,
            size=4_096,
            annotations={"org.opencontainers.image.title": "labels.txt"},
        ),
    ]

    packager = ModelPackager()
    manifest = packager.create_manifest(config, layers)

    print(f"  Schema version : {manifest.schema_version}")
    print(f"  Media type     : {manifest.media_type}")
    print(f"  Layer count    : {len(manifest.layers)}")
    print("\n  Manifest JSON (truncated):")
    manifest_json = manifest.model_dump_json(indent=2)
    for line in manifest_json.splitlines()[:20]:
        print(f"    {line}")
    print("    ...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("aumai-modeloci quickstart demo")
    print("=" * 40)

    # Demo 1: Pack — produces an archive file
    archive_path = demo_pack_model()

    # Demos 2-4: Operate on the archive produced in Demo 1
    demo_inspect_archive(archive_path)
    demo_verify_layers(archive_path)
    demo_unpack_archive(archive_path)

    # Demo 5: No file I/O needed
    demo_build_manifest()

    # Clean up the archive
    import os
    os.unlink(archive_path)

    print("\n" + "=" * 40)
    print("All demos completed successfully.")


if __name__ == "__main__":
    main()
