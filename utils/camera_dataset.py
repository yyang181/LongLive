# SPDX-License-Identifier: Apache-2.0
"""LMDB dataset providing (clean_latent, viewmats, Ks, prompts) for
Camera (PRoPE) Bidirectional SFT on Wan2.2-TI2V-5B.

Schema (per record key suffix `_<idx>_data`):
  latents     : float16  (F, C, H, W)        — Wan2.2 VAE latents
  prompts     : utf-8 str                    — caption
  intrinsics  : float32  (4,)                — [fx_norm, fy_norm, cx_norm, cy_norm]
  poses       : float32  (F, 7)              — [tx,ty,tz, qx,qy,qz,qw] w2c (OpenCV)

Top-level keys store global shapes:
  latents_shape, prompts_shape, intrinsics_shape, poses_shape
"""

import os
from typing import Tuple

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# LMDB helpers (kept self-contained so this dataset has no extra deps)
# ---------------------------------------------------------------------------
def _get_array_shape(env, array_name):
    with env.begin() as txn:
        raw = txn.get(f"{array_name}_shape".encode())
    if raw is None:
        return None
    return tuple(map(int, raw.decode().split()))


def _retrieve_row(env, array_name, dtype, idx, shape=None):
    with env.begin() as txn:
        raw = txn.get(f"{array_name}_{idx}_data".encode())
    if dtype == str:
        return raw.decode()
    arr = np.frombuffer(raw, dtype=dtype)
    if shape is not None and len(shape) > 0:
        arr = arr.reshape(shape)
    return arr


def cycle(dl):
    while True:
        for data in dl:
            yield data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CameraLatentLMDBDataset(Dataset):
    """LMDB dataset yielding `clean_latent`, `viewmats`, `Ks`, `prompts`.

    `data_path` may either be:
      * a single LMDB directory (contains ``data.mdb``); or
      * a parent directory of multiple LMDB shards (sub-directories each
        containing a ``data.mdb``).
    """

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.max_pair = max_pair

        if os.path.isfile(os.path.join(data_path, "data.mdb")):
            self._sharded = False
            self.env = lmdb.open(data_path, readonly=True,
                                 lock=False, readahead=False, meminit=False)
            self.latents_shape = _get_array_shape(self.env, "latents")
            self.intrinsics_shape = _get_array_shape(self.env, "intrinsics")
            self.poses_shape = _get_array_shape(self.env, "poses")
            return

        # ---- Sharded path ----
        self._sharded = True
        self.envs = []
        self.index = []  # list of (shard_id, local_idx)
        self._latents_shapes = []
        self._intrinsics_shapes = []
        self._poses_shapes = []
        for fname in sorted(os.listdir(data_path)):
            sub = os.path.join(data_path, fname)
            if not os.path.isdir(sub):
                continue
            if not os.path.isfile(os.path.join(sub, "data.mdb")):
                continue
            env = lmdb.open(sub, readonly=True, lock=False,
                            readahead=False, meminit=False)
            sid = len(self.envs)
            self.envs.append(env)
            ls = _get_array_shape(env, "latents")
            self._latents_shapes.append(ls)
            self._intrinsics_shapes.append(_get_array_shape(env, "intrinsics"))
            self._poses_shapes.append(_get_array_shape(env, "poses"))
            for j in range(ls[0]):
                self.index.append((sid, j))

    def __len__(self):
        if self._sharded:
            return min(len(self.index), self.max_pair)
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        if self._sharded:
            sid, local_idx = self.index[idx]
            env = self.envs[sid]
            ls = self._latents_shapes[sid]
            ints = self._intrinsics_shapes[sid]
            ps = self._poses_shapes[sid]
            latents = _retrieve_row(env, "latents", np.float16, local_idx, ls[1:])
            prompts = _retrieve_row(env, "prompts", str, local_idx)
            intrinsics = _retrieve_row(env, "intrinsics", np.float32, local_idx, ints[1:])
            poses = _retrieve_row(env, "poses", np.float32, local_idx, ps[1:])
        else:
            latents = _retrieve_row(self.env, "latents", np.float16, idx, self.latents_shape[1:])
            prompts = _retrieve_row(self.env, "prompts", str, idx)
            intrinsics = _retrieve_row(self.env, "intrinsics", np.float32, idx, self.intrinsics_shape[1:])
            poses = _retrieve_row(self.env, "poses", np.float32, idx, self.poses_shape[1:])

        # latents may be (F, C, H, W) -> add T (denoising-step) axis if absent.
        if latents.ndim == 4:
            latents_t = torch.tensor(latents, dtype=torch.float32)
        else:
            # (T, F, C, H, W) -> take last (clean) step
            latents_t = torch.tensor(latents, dtype=torch.float32)[-1]

        viewmats, Ks = build_viewmats_and_Ks(intrinsics, poses)
        return {
            "prompts": prompts,
            "clean_latent": latents_t,
            "viewmats": torch.tensor(viewmats, dtype=torch.float32),
            "Ks": torch.tensor(Ks, dtype=torch.float32),
        }


def build_viewmats_and_Ks(
    intrinsics: np.ndarray, poses: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw `(intrinsics, poses)` storage to (viewmats, Ks).

    Args:
        intrinsics: (4,) float32 [fx, fy, cx, cy] (normalized to image size)
        poses:      (T, 7) float32 [tx,ty,tz, qx,qy,qz,qw] w2c (OpenCV)

    Returns:
        viewmats: (T, 4, 4) float32 — w2c, normalized so frame 0 is identity
        Ks:       (T, 3, 3) float32
    """
    T = len(poses)
    fx, fy, cx, cy = intrinsics

    viewmats = np.zeros((T, 4, 4), dtype=np.float32)
    for i in range(T):
        tx, ty, tz, qx, qy, qz, qw = poses[i]
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        viewmats[i, :3, :3] = R
        viewmats[i, :3, 3] = [tx, ty, tz]
        viewmats[i, 3, 3] = 1.0

    # Normalize: align everything to the first frame's coordinate system.
    c2w = np.linalg.inv(viewmats)
    C0_inv = np.linalg.inv(c2w[0])
    c2w_aligned = np.array([C0_inv @ C for C in c2w])
    viewmats = np.linalg.inv(c2w_aligned).astype(np.float32)

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    Ks = np.tile(K, (T, 1, 1))
    return viewmats, Ks
