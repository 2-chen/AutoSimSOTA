"""Pinned official texture retrieval into the new benchmark worktree only."""
import argparse
import shutil
import zipfile
from pathlib import Path

from autosim.research.common import atomic_json, digest, immutable_json, now


def main():
    from huggingface_hub import hf_hub_download
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    destination = output / "native/RoboTwin/assets"
    request = {"repo_id": "TianxingChen/RoboTwin2.0", "repo_type": "dataset",
               "revision": "785feb15aa4a4f532395ad2b1d2be5f28cb561ad", "filename": "background_texture.zip",
               "expected_bytes": 10970687027}
    immutable_json(output / "texture_request.json", request)
    if shutil.disk_usage(output).free < 40 * 1024**3:
        raise RuntimeError("40 GiB free required for download/extraction safety reserve")
    atomic_json(output / "texture_state.json", {"status": "downloading", "started_at": now()})
    archive = Path(hf_hub_download(**{k: v for k, v in request.items() if k != "expected_bytes"},
                                  local_dir=output / "downloads"))
    if archive.stat().st_size != request["expected_bytes"]:
        raise ValueError("official texture archive size mismatch")
    with zipfile.ZipFile(archive) as stream:
        members = [m for m in stream.infolist() if m.filename.startswith("background_texture/")]
        if not members or sum(m.file_size for m in members) > 30 * 1024**3:
            raise ValueError("unexpected archive contents/size")
        for member in members:
            path = destination / member.filename
            if not path.resolve().is_relative_to(destination.resolve()) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("unsafe archive path/symlink")
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            if path.exists():
                if path.stat().st_size != member.file_size:
                    raise ValueError(f"existing texture mismatch: {path}")
                # A partial extraction is not trusted based on size alone.
                import zlib
                crc = 0
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024**2), b""):
                        crc = zlib.crc32(chunk, crc)
                if crc != member.CRC:
                    raise ValueError(f"existing texture CRC mismatch: {path}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".extracting")
            if temporary.exists():
                raise ValueError(f"partial extraction needs audit: {temporary}")
            with stream.open(member) as source, temporary.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024**2)
            temporary.rename(path)
    atomic_json(output / "texture_state.json", {"status": "completed", "finished_at": now(),
                "request": request, "archive_sha256": digest(archive),
                "seen_files": len(list((destination / "background_texture/seen").glob("*"))),
                "unseen_files": len(list((destination / "background_texture/unseen").glob("*")))})


if __name__ == "__main__":
    main()
