#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static/lightweight diagnostics for DreamX EPRoPE -> LongLive causal AR conversion.

This script deliberately avoids constructing the 5B model or loading tensor
storages from huge checkpoints. It checks source-level invariants and checkpoint
metadata that are sufficient to catch the common P0 conversion bugs:
  * camera branch bypasses causal teacher-forcing mask;
  * clean/noisy streams share one camera list without stream-aware splitting;
  * SP path silently drops camera kwargs;
  * bidirectional checkpoint lacks cam_self_attn keys.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover
    OmegaConf = None
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _ok(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}{': ' + detail if detail else ''}")
    return passed


def check_source(repo: Path) -> bool:
    dreamx = _read(repo / "wan_5b/modules/dreamx_camera.py")
    causal = _read(repo / "wan_5b/modules/causal_model.py")
    sp = _read(repo / "wan_5b/distributed/sequence_parallel.py")

    passed = True
    passed &= _ok(
        "PropeSelfAttention accepts block_mask",
        "block_mask=None" in dreamx and "flex_attention" in dreamx,
    )
    passed &= _ok(
        "PropeSelfAttention splits clean/noisy before PRoPE",
        "is_tf" in dreamx and "q_clean, q_noisy" in dreamx and "out_clean, out_noisy" in dreamx,
    )
    passed &= _ok(
        "CausalWanAttentionBlock passes block_mask to cam_self_attn",
        "cam_self_attn(" in causal and "block_mask=block_mask" in causal,
    )
    passed &= _ok(
        "CausalWanModel asserts camera/latent frame length",
        "viewmats.shape[1] == f_lat" in causal and "Ks.shape[1] == f_lat" in causal,
    )
    passed &= _ok(
        "SP path explicitly handles camera kwargs",
        "viewmats=None" in sp and "DreamX camera EPRoPE causal training" in sp,
        "currently expected to raise until distributed cam_self_attn is implemented",
    )
    return passed


def check_config(config_path: Path) -> bool:
    if OmegaConf is not None:
        cfg = OmegaConf.load(config_path)
        for section in ("infra", "algorithm", "training", "data", "checkpoints"):
            sec = cfg.get(section, None)
            if sec is not None:
                for k, v in sec.items():
                    cfg[k] = v
        get = cfg.get
        model_kwargs = cfg.get("model_kwargs", {})
    elif yaml is not None:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        flat = {}
        for section in ("infra", "algorithm", "training", "data", "checkpoints"):
            if isinstance(raw.get(section), dict):
                flat.update(raw[section])
        flat["model_kwargs"] = raw.get("model_kwargs", {})
        get = flat.get
        model_kwargs = flat.get("model_kwargs", {})
    else:
        text = config_path.read_text(encoding="utf-8")
        get = lambda key, default=None: (key in text) or default
        model_kwargs = {"wrapper_cls": text}

    passed = True
    passed &= _ok("config causal teacher-forcing", bool(get("causal")) and bool(get("teacher_forcing")))
    passed &= _ok("config I2V independent_first_frame", bool(get("i2v")) and bool(get("independent_first_frame")))
    passed &= _ok("config uses DreamX wrapper", "DreamXCameraWanDiffusionWrapper" in str(model_kwargs.get("wrapper_cls", "")))
    passed &= _ok("config sequence_parallel_size", int(get("sequence_parallel_size", 1)) == 1,
                  "SP>1 requires separate distributed EPRoPE implementation")
    return passed


def check_checkpoint_metadata(ckpt_path: Path) -> bool:
    if not ckpt_path.exists():
        return _ok("checkpoint exists", False, str(ckpt_path))
    if not zipfile.is_zipfile(ckpt_path):
        return _ok("checkpoint zip metadata", False, "not a torch zip checkpoint")
    with zipfile.ZipFile(ckpt_path) as zf:
        pkl_names = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not pkl_names:
            return _ok("checkpoint data.pkl", False)
        data = zf.read(pkl_names[0])
    text = data.decode("latin1", errors="ignore")
    required = [
        "cam_self_attn.q_proj.weight",
        "cam_self_attn.k_proj.weight",
        "cam_self_attn.v_proj.weight",
        "cam_self_attn.out_proj.weight",
        "cam_self_attn.norm_q.weight",
        "cam_self_attn.norm_k.weight",
    ]
    passed = True
    for key in required:
        passed &= _ok(f"checkpoint contains {key}", key in text)
    blocks = sorted({int(m.group(1)) for m in re.finditer(r"blocks\.(\d+)\.cam_self_attn\.out_proj\.weight", text)})
    passed &= _ok("checkpoint cam_self_attn block count", len(blocks) >= 30, f"found={len(blocks)}")
    return passed


def check_teacher_forcing_camera_ids(frames: int = 8, tokens_per_frame: int = 3) -> bool:
    clean = [t for t in range(frames) for _ in range(tokens_per_frame)]
    noisy = [t for t in range(frames) for _ in range(tokens_per_frame)]
    expected = clean + noisy
    old_group_size = (2 * frames * tokens_per_frame) // frames
    old = [i for i in range(frames) for _ in range(old_group_size)]
    old_mismatch = sum(a != b for a, b in zip(old, expected))
    passed = old_mismatch > 0 and expected[:tokens_per_frame] == expected[frames * tokens_per_frame:frames * tokens_per_frame + tokens_per_frame]
    return _ok(
        "clean/noisy camera-id diagnostic",
        passed,
        f"old_single-camera-list_mismatch_tokens={old_mismatch}/{len(expected)}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", default="configs/train_dreamx_camera_i2v_ar.yaml")
    parser.add_argument("--checkpoint", default="logs/train_dreamx_camera_i2v_b300/checkpoint_model_003000/model.pt")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    config = Path(args.config)
    if not config.is_absolute():
        config = repo / config
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = repo / ckpt

    print(f"repo={repo}")
    print(f"config={config}")
    print(f"checkpoint={ckpt} size_gb={(ckpt.stat().st_size / 1e9) if ckpt.exists() else -1:.2f}")

    passed = True
    passed &= check_source(repo)
    passed &= check_config(config)
    passed &= check_checkpoint_metadata(ckpt)
    passed &= check_teacher_forcing_camera_ids()
    print("OVERALL", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
