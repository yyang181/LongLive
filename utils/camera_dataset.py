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
from pathlib import Path
from typing import Optional, Tuple

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


def _retrieve_optional_text(env, array_name, idx):
    """Read an optional UTF-8 metadata field from an LMDB record."""
    with env.begin() as txn:
        raw = txn.get(f"{array_name}_{idx}_data".encode())
    return None if raw is None else raw.decode("utf-8")


def _reference_name_from_source_path(source_path: str) -> str:
    """Derive a collision-resistant output name from a stored source path.

    MIND stores every source clip as ``.../<sample>/video.mp4``.  In that
    case the video stem alone is not useful, so retain its sample directory
    and (when present) its first-/third-person dataset component.
    """
    path = Path(source_path)
    if path.stem != "video":
        return path.stem
    perspective = next(
        (parent.name for parent in path.parents
         if parent.name in {"1st_data", "3rd_data"}),
        None,
    )
    return f"{perspective}_{path.parent.name}" if perspective else path.parent.name


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

    def __init__(
        self,
        data_path: str,
        max_pair: int = int(1e8),
        target_num_frames: Optional[int] = None,
        expected_latent_shape: Optional[Tuple[int, int, int]] = None,
        skip_short_lmdb: bool = False,
    ):
        """Create a camera-LMDB dataset with optional prefix time cropping.

        Temporal cropping is applied identically to VAE latents and camera
        poses. Spatial/channel changes are rejected: they require VAE
        re-encoding rather than resizing stored latents.  When
        ``skip_short_lmdb`` is set, an LMDB (or a shard in a shard directory)
        with fewer than ``target_num_frames`` is excluded instead of aborting
        dataset construction.
        """
        self.max_pair = max_pair
        self.skip_short_lmdb = bool(skip_short_lmdb)
        self.skipped_short_lmdbs = []  # (path, available latent frames)
        self.target_num_frames = (
            None if target_num_frames is None else int(target_num_frames)
        )
        if self.target_num_frames is not None and self.target_num_frames <= 0:
            raise ValueError(
                f"target_num_frames must be positive, got {self.target_num_frames}."
            )
        self.expected_latent_shape = (
            None
            if expected_latent_shape is None
            else tuple(int(v) for v in expected_latent_shape)
        )
        if self.expected_latent_shape is not None and len(self.expected_latent_shape) != 3:
            raise ValueError(
                "expected_latent_shape must be (C, H, W), got "
                f"{self.expected_latent_shape}."
            )

        if os.path.isfile(os.path.join(data_path, "data.mdb")):
            self._sharded = False
            self.env = lmdb.open(data_path, readonly=True,
                                 lock=False, readahead=False, meminit=False)
            self.latents_shape = _get_array_shape(self.env, "latents")
            self.intrinsics_shape = _get_array_shape(self.env, "intrinsics")
            self.poses_shape = _get_array_shape(self.env, "poses")
            latent_frames = self._validate_storage_shapes(
                self.latents_shape, self.poses_shape, data_path
            )
            if self._exclude_short_lmdb(data_path, latent_frames):
                # Keep this object valid but empty so a mixed list of source
                # datasets can simply concatenate it with the valid sources.
                self.env.close()
                self.env = None
                self._empty = True
            else:
                self._empty = False
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
            ls = _get_array_shape(env, "latents")
            ps = _get_array_shape(env, "poses")
            latent_frames = self._validate_storage_shapes(ls, ps, sub)
            if self._exclude_short_lmdb(sub, latent_frames):
                env.close()
                continue
            sid = len(self.envs)
            self.envs.append(env)
            self._latents_shapes.append(ls)
            self._intrinsics_shapes.append(_get_array_shape(env, "intrinsics"))
            self._poses_shapes.append(ps)
            for j in range(ls[0]):
                self.index.append((sid, j))

    def _exclude_short_lmdb(self, source: str, latent_frames: int) -> bool:
        """Return whether a valid-but-short LMDB should be excluded."""
        if (
            self.target_num_frames is None
            or latent_frames >= self.target_num_frames
        ):
            return False
        if not self.skip_short_lmdb:
            raise ValueError(
                f"Camera LMDB at {source} has {latent_frames} frames, but config "
                f"requests {self.target_num_frames}. Padding is not supported. "
                "Set skip_short_lmdb=True to exclude this LMDB."
            )
        self.skipped_short_lmdbs.append((source, latent_frames))
        return True

    def _validate_storage_shapes(self, latents_shape, poses_shape, source: str) -> int:
        """Validate temporal cropping without hiding latent/pose mismatches."""
        if latents_shape is None or poses_shape is None:
            raise ValueError(f"Camera LMDB at {source} is missing latent or pose shape metadata.")

        # Top-level latent shape is (N,F,C,H,W), or legacy (N,T,F,C,H,W).
        row_shape = tuple(latents_shape[1:])
        if len(row_shape) == 4:
            latent_shape = row_shape
        elif len(row_shape) == 5:
            latent_shape = row_shape[1:]
        else:
            raise ValueError(
                f"Unsupported latent row shape {row_shape} in {source}; "
                "expected (F,C,H,W) or (T,F,C,H,W)."
            )
        latent_frames, *latent_spatial = latent_shape

        pose_row_shape = tuple(poses_shape[1:])
        if len(pose_row_shape) != 2 or pose_row_shape[1] != 7:
            raise ValueError(
                f"Unsupported pose row shape {pose_row_shape} in {source}; "
                "expected (F,7)."
            )
        pose_frames = pose_row_shape[0]
        if pose_frames != latent_frames:
            raise ValueError(
                f"Camera LMDB at {source} has {latent_frames} latent frames but "
                f"{pose_frames} camera poses; they must match."
            )
        if (
            self.expected_latent_shape is not None
            and tuple(latent_spatial) != self.expected_latent_shape
        ):
            raise ValueError(
                f"Camera LMDB at {source} stores latent C/H/W={tuple(latent_spatial)}, "
                f"but config requests {self.expected_latent_shape}. Only temporal "
                "cropping is supported; re-encode the videos for a new resolution."
            )
        return latent_frames

    def __len__(self):
        if self._sharded:
            return min(len(self.index), self.max_pair)
        if self._empty:
            return 0
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
            source_path = _retrieve_optional_text(env, "paths", local_idx)
        else:
            latents = _retrieve_row(self.env, "latents", np.float16, idx, self.latents_shape[1:])
            prompts = _retrieve_row(self.env, "prompts", str, idx)
            intrinsics = _retrieve_row(self.env, "intrinsics", np.float32, idx, self.intrinsics_shape[1:])
            poses = _retrieve_row(self.env, "poses", np.float32, idx, self.poses_shape[1:])
            source_path = _retrieve_optional_text(self.env, "paths", idx)

        if self.target_num_frames is not None:
            # Crop the same latent-frame prefix for image and camera streams.
            if latents.ndim == 4:
                latents = latents[:self.target_num_frames]
            else:
                latents = latents[:, :self.target_num_frames]
            poses = poses[:self.target_num_frames]

        # latents may be (F, C, H, W) -> add T (denoising-step) axis if absent.
        # Keep storage dtype (fp16) here: the trainer immediately re-casts to
        # ``self.dtype`` (bf16) on the GPU, so an intermediate fp32 promotion
        # in the worker would only double the worker→pin-memory→GPU bytes for
        # no benefit. ``_retrieve_row`` returns a read-only view over the
        # LMDB mmap (``np.frombuffer``); ``torch.from_numpy`` requires a
        # writable buffer, so we ``.copy()`` once into a writable np array
        # (this is the only required allocation in this hot path).
        if latents.ndim == 4:
            latents_t = torch.from_numpy(latents.copy())
        else:
            # (T, F, C, H, W) -> take last (clean) step
            latents_t = torch.from_numpy(latents[-1].copy())

        viewmats, Ks = build_viewmats_and_Ks(intrinsics, poses)
        item = {
            "prompts": prompts,
            "clean_latent": latents_t,
            "viewmats": torch.tensor(viewmats, dtype=torch.float32),
            "Ks": torch.tensor(Ks, dtype=torch.float32),
        }
        if source_path:
            item["reference_name"] = _reference_name_from_source_path(source_path)
        return item


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
