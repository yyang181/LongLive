#!/usr/bin/env python3
"""Extract one representative raw video for every P2P-full game.

The metadata parquet is used to choose the longest recording for each game.
Archives are only streamed: only matching ``video.mp4`` members are written,
and neither annotations nor the auxiliary 192x192 videos are extracted.

For Roblox, ``env_sub_type`` identifies the game.  Known spelling variants
are normalized with the same helper used by the existing P2P visualizer.
All other games are grouped by their stripped ``env_name`` value.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def is_roblox_environment(name: str) -> bool:
    return "".join(str(name).strip().lower().split()) in {"roblox", "roblos", "rpblox"}


def normalize_game_name(name: str) -> str:
    """Normalize the known noisy Roblox subtype labels in the P2P metadata."""
    value = "-".join(str(name).strip().lower().replace("_", "-").split())
    value = value.replace("/", "-").replace("=", "-").replace(".", "")
    aliases = {
        "a-adusty-trip": "a-dusty-trip", "a-dirty-trip": "a-dusty-trip",
        "a-dusry-trip": "a-dusty-trip", "a-dust-trip": "a-dusty-trip",
        "a-dusty-drive": "a-dusty-trip", "a-duty-trip": "a-dusty-trip", "dusty-trip": "a-dusty-trip",
        "balde-ball": "blade-ball", "blade-bal": "blade-ball", "blade-ball-elemental": "blade-ball", "blade-ball-training": "blade-ball",
        "be-a-torando": "be-a-tornado", "be-a-trex": "be-a-t-rex", "be-a-skane": "be-a-snake", "be-snake": "be-a-snake", "shark": "be-a-shark", "be-tornado": "be-a-tornado",
        "hypershoot": "hypershot", "hipershot": "hypershot",
        "natural-disaster": "natural-disaster-survival", "atural-disaster-survival": "natural-disaster-survival",
        "natural-disaster-survivalnatural-disaster-survival": "natural-disaster-survival", "natural-disaster-sruvival": "natural-disaster-survival",
        "natural-disaster-surival": "natural-disaster-survival", "natural-disaster-surviva": "natural-disaster-survival", "natural-disaster-survivala": "natural-disaster-survival",
        "natural-disaster-survivor": "natural-disaster-survival", "natural-disaster-suvival": "natural-disaster-survival", "natural-disater-survival": "natural-disaster-survival",
        "natural-survival-disaster": "natural-disaster-survival", "nature-disaster-survival": "natural-disaster-survival", "nataural-disaster-survival": "natural-disaster-survival", "ntural-survival-disaster": "natural-disaster-survival",
        "murder-vs-xerif": "murderers-vs-sheriffs", "murderers-sheriffs": "murderers-vs-sheriffs", "murderers-vs-sherrifs": "murderers-vs-sheriffs", "murders-vs-sheriffs": "murderers-vs-sheriffs", "be-a-snakemurderers-vs-sheriffs": "murderers-vs-sheriffs",
        "rival-res-1080p": "rivals", "rivals-res-1080p": "rivals", "roblox-rivals-res-1080p": "rivals", "rivais": "rivals",
        "eat-the-world": "eat-a-world", "slap-batte": "slap-battle", "slap-battles": "slap-battle", "slapr-battle": "slap-battle",
    }
    return aliases.get(value, value or "unknown")


def output_group(env_name: str, env_sub_type: str) -> str:
    """Return a stable, readable game directory relative to the output root."""
    env_name = str(env_name).strip()
    env_sub_type = str(env_sub_type).strip()
    if is_roblox_environment(env_name):
        return f"roblox/{normalize_game_name(env_sub_type)}"
    # Keep the source label's identity while making it safe as a directory name.
    return re.sub(r"[^A-Za-z0-9._-]+", "-", env_name.lower()).strip(".-") or "unknown"


def select_samples(metadata_path: Path) -> tuple[dict[str, dict], list[dict]]:
    """Select the longest metadata row per game and return a JSON manifest."""
    rows = pq.read_table(metadata_path).to_pylist()
    selected: dict[str, dict] = {}
    for row in rows:
        group = output_group(row["env_name"], row["env_sub_type"])
        candidate = {
            "id": str(row["id"]),
            "filepath": str(row["filepath"]),
            "game": group,
            "env_name": row["env_name"],
            "env_sub_type": row["env_sub_type"],
            "num_frames": row["num_frames"],
        }
        current = selected.get(group)
        if current is None or (candidate["num_frames"], candidate["id"]) > (
            current["num_frames"], current["id"],
        ):
            selected[group] = candidate

    # UUID is the tar directory name.  It is unique, but make that invariant
    # explicit before beginning a multi-hour archive scan.
    by_id = {item["filepath"]: item for item in selected.values()}
    if len(by_id) != len(selected):
        raise RuntimeError("The selected games do not have unique recording UUIDs")
    return by_id, [selected[key] for key in sorted(selected)]


def extract_archive(archive: Path, chosen: dict[str, dict], output_dir: Path) -> Counter:
    """Stream one archive and write only selected original video members."""
    completed: Counter = Counter()
    with tarfile.open(archive, "r|gz") as handle:
        for member in handle:
            if not member.isfile() or not member.name.endswith("/video.mp4"):
                continue
            sample_id = Path(member.name).parent.name
            choice = chosen.get(sample_id)
            if choice is None:
                continue
            destination = output_dir / choice["game"] / f"{sample_id}.mp4"
            if destination.is_file() and destination.stat().st_size > 0:
                completed[choice["game"]] += 1
                continue
            source = handle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read {archive}:{member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".mp4.part")
            with temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            temporary.replace(destination)
            completed[choice["game"]] += 1
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2p_path", type=Path, default=Path("/nfs/hongfenglai/p2p-full-data"))
    parser.add_argument("--output_dir", type=Path, default=Path("/nfs/yixinyang/code/LongLive/data/p2pfull_sample"))
    parser.add_argument("--max_archives", type=int, default=0, help="For a smoke test; 0 scans all archives.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.p2p_path / "dataset"
    archives = sorted(dataset_dir.glob("batch_*.tar.gz"))
    if args.max_archives:
        archives = archives[: args.max_archives]
    if not archives:
        raise RuntimeError(f"No batch archives under {dataset_dir}")

    chosen, manifest = select_samples(args.p2p_path / "data_metadata.parquet")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Selected {len(chosen)} games; streaming {len(archives)} archives.", flush=True)
    found: Counter = Counter()
    for index, archive in enumerate(archives, 1):
        print(f"[{index}/{len(archives)}] {archive.name}", flush=True)
        found.update(extract_archive(archive, chosen, args.output_dir))

    missing = sorted(item["game"] for item in manifest if found[item["game"]] == 0)
    print(f"Complete: {len(chosen) - len(missing)}/{len(chosen)} games.", flush=True)
    if missing:
        print("Missing: " + ", ".join(missing), flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
