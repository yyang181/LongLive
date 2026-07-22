# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import os
import re
import sys

# torchrun no longer consistently prepends the script directory to sys.path,
# which breaks absolute project imports when launched from another cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# torchvision 0.27+ removed write_video/read_video. Several modules import the
# symbols at module import time, so patch them before importing project code.
import torchvision.io as _tv_io
if not hasattr(_tv_io, "write_video"):
    import imageio.v2 as _imageio_v2

    def _shim_write_video(filename, video_array, fps, **_unused):
        if hasattr(video_array, "detach"):
            video_array = video_array.detach().cpu().numpy()
        _imageio_v2.mimwrite(filename, video_array, fps=fps, codec="libx264", quality=8)

    _tv_io.write_video = _shim_write_video
if not hasattr(_tv_io, "read_video"):
    import imageio.v3 as _imageio_v3
    import torch as _torch_for_shim

    def _shim_read_video(filename, pts_unit="sec", output_format="THWC", **_unused):
        frames = _imageio_v3.imread(filename, plugin="pyav")
        tensor = _torch_for_shim.from_numpy(frames)
        if output_format == "TCHW":
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor, None, {}

    _tv_io.read_video = _shim_read_video

import argparse
import torch
from omegaconf import OmegaConf
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import CausalDiffusionInferencePipeline
from utils.dataset import MultiTextConcatDataset, MultiVideoConcatDataset, eval_collate_fn, multi_video_collate_fn
from utils.misc import set_seed
from utils.config import normalize_config, section_get, wan_default_config
from utils.nvfp4_checkpoint import (
    clean_fsdp_state_dict_keys,
    drop_fouroversix_master_weights,
    is_nvfp4_state_dict,
    is_te_nvfp4_checkpoint,
    quantize_model_for_fouroversix_nvfp4,
    unwrap_generator_state_dict,
)

from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller
from utils.camera_dataset import CameraLatentLMDBDataset

try:
    from PIL import Image as _PIL_Image
except ImportError:  # pragma: no cover
    _PIL_Image = None


class ThreeFileI2VDataset(torch.utils.data.Dataset):
    """AR i2v inference dataset that mirrors the bidirectional
    ``scripts/inference/inference_bidir_camera.py`` three-file input contract:

        inference.image_path      – one source-image path per line
        inference.prompt_path     – one prompt per line
        inference.trajectory_path – one camera-trajectory string per line
                                    (aligned 1-to-1 with the two files above)

    Produces items in the same shape/keys expected by ``inference.py``'s i2v
    branch: ``batch["image"]`` is a ``[C, H, W]`` float tensor in ``[-1, 1]``
    (``multi_video_collate_fn``-style layout) and ``batch["prompts"]`` is a
    single-string ``List[str]`` so the downstream ``batch['prompts'][0]``
    picks up the text prompt for that clip.  Camera trajectories are still
    read directly from ``inference.trajectory_path`` by the existing
    ``use_camera`` block in the inference loop, so nothing extra needs to be
    threaded through the dataset here.
    """

    def __init__(self, image_path_file: str, prompt_path_file: str,
                 video_size, num_output_frames: int):
        if _PIL_Image is None:
            raise RuntimeError(
                "PIL is required for ThreeFileI2VDataset; please `pip install pillow`."
            )
        with open(image_path_file) as f:
            self.image_paths = [ln.strip() for ln in f if ln.strip()]
        with open(prompt_path_file) as f:
            self.prompts = [ln.strip() for ln in f if ln.strip()]
        assert len(self.image_paths) == len(self.prompts), (
            f"image_paths ({len(self.image_paths)}) and prompts "
            f"({len(self.prompts)}) must align (one per line)"
        )
        self.video_size = tuple(video_size)  # (H, W)
        self.num_output_frames = int(num_output_frames)
        self._mode = "ThreeFileI2V"

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path: str) -> torch.Tensor:
        img = _PIL_Image.open(path).convert("RGB")
        H, W = self.video_size
        img = img.resize((W, H), _PIL_Image.BICUBIC)
        arr = torch.from_numpy(np.array(img, dtype=np.float32))  # (H, W, 3)
        arr = (arr / 127.5) - 1.0
        return arr.permute(2, 0, 1).contiguous()  # (C, H, W) in [-1, 1]

    def __getitem__(self, idx):
        image = self._load_image(self.image_paths[idx])
        prompt = self.prompts[idx]
        return {
            "idx": idx,
            "image": image,                     # (C, H, W) float32 in [-1, 1]
            "prompts": [prompt],                # List[str]; length aligns with
                                                # ``block_prompts`` downstream
            "image_path": self.image_paths[idx],
        }


def three_file_i2v_collate_fn(batch):
    """Collate function for :class:`ThreeFileI2VDataset` (batch_size=1).

    Matches the layout emitted by ``multi_video_collate_fn`` for the fields
    consumed downstream: stacks ``image`` -> ``[B, C, H, W]`` and keeps
    ``prompts`` as a length-B list of ``List[str]``.
    """
    assert len(batch) == 1, "ThreeFileI2VDataset only supports batch_size=1"
    b = batch[0]
    return {
        "idx": torch.tensor(b["idx"], dtype=torch.long),
        "image": b["image"].unsqueeze(0),      # (1, C, H, W)
        "prompts": [b["prompts"]],             # [ ["…"] ]
        "image_path": [b["image_path"]],
    }


class IndexedCameraLMDBInferenceDataset(torch.utils.data.Dataset):
    """Add stable sample indices to camera-LMDB records for inference output."""

    _mode = "CameraLatentLMDBInference"

    def __init__(self, lmdb_path, *, max_samples, target_num_frames,
                 expected_latent_shape):
        self.dataset = CameraLatentLMDBDataset(
            lmdb_path,
            max_pair=max_samples,
            target_num_frames=target_num_frames,
            expected_latent_shape=expected_latent_shape,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        item["idx"] = int(idx)
        return item


def camera_lmdb_i2v_collate_fn(batch):
    """Collate one camera-LMDB record into inference.py's batch contract."""
    if len(batch) != 1:
        raise ValueError("Camera LMDB inference currently requires batch_size=1.")
    item = batch[0]
    return {
        "idx": torch.tensor(item["idx"], dtype=torch.long),
        # Keep this nested shape: downstream expects prompts[0] to be the
        # per-causal-block prompt list.
        "prompts": [[str(item["prompts"])]],
        "clean_latent": item["clean_latent"].unsqueeze(0),
        "viewmats": item["viewmats"].unsqueeze(0),
        "Ks": item["Ks"].unsqueeze(0),
    }


def save_prompts_to_txt(prompts_for_sample, prompt_txt_path: str, is_main_process: bool):
    """Save per-block prompts alongside the video.

    Consecutive identical prompts are merged, e.g.:
        [0] a, [1] a, [2] b  =>  [0,1] a\\n[2] b\\n
    """
    try:
        with open(prompt_txt_path, "w", encoding="utf-8") as f:
            if len(prompts_for_sample) == 0:
                return
            current_prompt = prompts_for_sample[0]
            current_indices = [0]
            for seg_idx in range(1, len(prompts_for_sample)):
                p = prompts_for_sample[seg_idx]
                if p == current_prompt:
                    current_indices.append(seg_idx)
                else:
                    indices_str = ",".join(str(i) for i in current_indices)
                    f.write(f"[{indices_str}] {current_prompt}\n")
                    current_prompt = p
                    current_indices = [seg_idx]
            indices_str = ",".join(str(i) for i in current_indices)
            f.write(f"[{indices_str}] {current_prompt}\n")
    except Exception as e:
        if is_main_process:
            print(f"Warning: failed to save prompts to {prompt_txt_path}: {e}")

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument(
    "--lmdb_path",
    type=str,
    default=None,
    help=(
        "Optional camera LMDB test set. Each record supplies its prompt, "
        "I2V first latent, viewmats and Ks; configured image/prompt/trajectory "
        "inputs are ignored."
    ),
)
parser.add_argument(
    "--max_clips",
    type=int,
    default=-1,
    help="Maximum LMDB records to render; <= 0 renders the complete test set.",
)
parser.add_argument(
    "--max_lmdb_frames",
    type=int,
    default=None,
    help=(
        "Optional decoded-frame cap for --lmdb_path. Must be 4*k+1 so it "
        "aligns with Wan's temporal VAE stride (for example 77 or 149)."
    ),
)
te_quant_group = parser.add_mutually_exclusive_group()
te_quant_group.add_argument(
    "--use_te_quant",
    dest="use_te_quant",
    action="store_true",
    help="Override config and enable TransformerEngine quantization",
)
te_quant_group.add_argument(
    "--no_use_te_quant",
    dest="use_te_quant",
    action="store_false",
    help="Override config and disable TransformerEngine quantization",
)
parser.set_defaults(use_te_quant=None)
args, unknown = parser.parse_known_args()

config = OmegaConf.load(args.config_path)
if unknown:
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(unknown))
config = normalize_config(config)
if args.use_te_quant is not None:
    config.model_quant_use_transformer_engine = args.use_te_quant

if not hasattr(config, "sampling_steps") or config.sampling_steps is None:
    raise ValueError("sampling_steps must be defined in the inference config")

if not hasattr(config, "guidance_scale") or config.guidance_scale is None:
    config.guidance_scale = 1.0

config.use_ema = section_get(config, "inference", "use_ema", getattr(config, "use_ema", False))
config.output_folder = section_get(config, "inference", "output_folder", getattr(config, "output_folder", "videos/longlive2"))

# If the ckpt path encodes a training step (e.g. ``checkpoint_model_000200``),
# append that suffix to the output_folder so different checkpoints don't
# clobber each other (mirrors scripts/inference/inference_bidir_camera.py).
# For parameter-efficient runs the LoRA checkpoint is the trained artifact and
# generator_ckpt is only the immutable base, so name outputs after lora_ckpt.
_ckpt_for_suffix = (
    getattr(config, "lora_ckpt", None)
    or getattr(config, "generator_ckpt", None)
    or ""
)
if _ckpt_for_suffix:
    _m = re.search(r"checkpoint_model_(\d+)", _ckpt_for_suffix)
    if _m:
        config.output_folder = os.path.join(config.output_folder, _m.group(1))

config.num_samples = section_get(config, "inference", "num_samples", getattr(config, "num_samples", 1))
config.num_output_frames = getattr(config, "num_output_frames", config.image_or_video_shape[1])
config.save_with_index = getattr(config, "save_with_index", False)
config.inference_iter = getattr(config, "inference_iter", -1)
lmdb_path = args.lmdb_path or section_get(
    config, "inference", "lmdb_path", getattr(config, "lmdb_path", None)
)
lmdb_max_samples = int(args.max_clips)
if lmdb_path:
    if not getattr(config, "i2v", False):
        raise ValueError("--lmdb_path requires an I2V inference config.")
    if lmdb_max_samples == 0:
        raise ValueError("--max_clips must be positive or negative (for all samples).")
    if not config.save_with_index:
        config.save_with_index = True
        print("[data] LMDB mode forces save_with_index=true to avoid prompt-name collisions.")
    if args.max_lmdb_frames is not None:
        if args.max_lmdb_frames < 1 or (args.max_lmdb_frames - 1) % 4 != 0:
            raise ValueError(
                "--max_lmdb_frames must be a positive 4*k+1 value "
                f"(for example 77 or 149), got {args.max_lmdb_frames}."
            )
        lmdb_latent_frames = (args.max_lmdb_frames - 1) // 4 + 1
        config.num_output_frames = lmdb_latent_frames
        config.image_or_video_shape[1] = lmdb_latent_frames
        print(
            f"[data] LMDB inference limited to {lmdb_latent_frames} latent "
            f"frames ({args.max_lmdb_frames} decoded frames)."
        )

if bool(getattr(config, "fp8_quant", False)) and bool(
    getattr(config, "model_quant", False)
):
    raise ValueError("fp8_quant and model_quant (NVFP4) are mutually exclusive.")


def _maybe_to_dict(value):
    if value is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return dict(value)


def _config_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _expected_inference_samples(config):
    inference_iter = int(getattr(config, "inference_iter", -1))
    if inference_iter >= 0:
        return inference_iter + 1
    return None


def _resolve_torch_compile(config):
    setting = getattr(config, "torch_compile", False)
    if isinstance(setting, str) and setting.strip().lower() == "auto":
        if not (
            bool(getattr(config, "model_quant", False))
            or bool(getattr(config, "fp8_quant", False))
        ):
            return False, "auto disabled because quantization is false"
        min_samples = int(getattr(config, "torch_compile_min_samples", 2))
        expected_samples = _expected_inference_samples(config)
        if expected_samples is not None and expected_samples < min_samples:
            return (
                False,
                "auto disabled because expected samples "
                f"({expected_samples}) < torch_compile_min_samples ({min_samples})",
            )
        return True, "auto enabled for repeated quantized inference"
    return _config_bool(setting, default=False), "explicit setting"


def quantize_generator_model(model, config, keep_master_weights):
    from utils.quant import (
        ModelQuantizationConfig,
        _materialize_mixed_quantized_weights_for_inference,
        _materialize_quantized_weights_for_inference,
        _materialize_transformer_engine_weights_for_inference,
        quantize_model_with_filter,
    )

    use_transformer_engine = bool(getattr(config, "model_quant_use_transformer_engine", False))
    te_inference_only = bool(getattr(config, "model_quant_te_inference_only", use_transformer_engine))
    te_low_precision_weights = bool(getattr(config, "model_quant_te_low_precision_weights", te_inference_only))
    te_fallback_to_fouroversix = bool(getattr(config, "model_quant_te_fallback_to_fouroversix", False))

    quant_cfg = ModelQuantizationConfig(
        scale_rule=getattr(config, "model_quant_scale_rule", "static_6"),
        quantize_backend=getattr(config, "model_quant_backend", None),
        activation_scale_rule=getattr(
            config,
            "model_quant_activation_scale_rule",
            getattr(config, "model_quant_scale_rule", "static_6"),
        ),
        weight_scale_rule=getattr(config, "model_quant_weight_scale_rule", None),
        gradient_scale_rule=getattr(config, "model_quant_gradient_scale_rule", None),
    )
    quant_cfg.keep_master_weights = keep_master_weights
    model, matched_modules = quantize_model_with_filter(
        model,
        quant_config=quant_cfg,
        filtered_modules=getattr(config, "model_quant_filtered_modules", None),
        use_default_filtered_modules=getattr(config, "model_quant_use_default_filtered_modules", True),
        cast_model_to_bf16=True,
        materialize_for_inference=False,
        use_transformer_engine=use_transformer_engine,
        te_inference_only=te_inference_only,
        te_low_precision_weights=te_low_precision_weights,
        te_recipe_kwargs=_maybe_to_dict(getattr(config, "model_quant_te_recipe_kwargs", None)),
        te_module_kwargs=_maybe_to_dict(getattr(config, "model_quant_te_module_kwargs", None)),
        te_fallback_to_fouroversix=te_fallback_to_fouroversix,
        verbose=True,
    )
    materialize_fn = _materialize_quantized_weights_for_inference
    if use_transformer_engine and te_fallback_to_fouroversix:
        materialize_fn = _materialize_mixed_quantized_weights_for_inference
    elif use_transformer_engine:
        materialize_fn = _materialize_transformer_engine_weights_for_inference
    if local_rank == 0:
        print(f"[NVFP4] Generator quantized; {len(matched_modules)} modules excluded")
    return model, materialize_fn


def materialize_quantized_generator(model, device, materialize_fn, stage_desc):
    mat_modules, master_bytes, quantized_bytes = materialize_fn(
        model,
        target_device=device,
    )
    if local_rank == 0:
        print(
            f"[NVFP4] Materialized quantized generator weights {stage_desc}: "
            f"{len(mat_modules)} modules, "
            f"master_weight={master_bytes / (1024 ** 3):.3f} GiB, "
            f"quantized_weight={quantized_bytes / (1024 ** 3):.3f} GiB"
        )


def configure_generator_torch_compile(pipeline, config):
    compile_enabled, reason = _resolve_torch_compile(config)
    if not compile_enabled:
        if local_rank == 0 and str(getattr(config, "torch_compile", "false")).lower() == "auto":
            print(f"[torch.compile] skipped: {reason}")
        return
    target = str(getattr(config, "torch_compile_target", "generator_model")).lower()
    if target not in {"generator_model", "model"}:
        if local_rank == 0:
            print(f"[torch.compile][warn] Unsupported target={target}; expected generator_model")
        return
    if not hasattr(pipeline.generator, "configure_torch_compile"):
        if local_rank == 0:
            print("[torch.compile][warn] Current generator does not expose configure_torch_compile; skipping")
        return
    compiled = pipeline.generator.configure_torch_compile(
        backend=str(getattr(config, "torch_compile_backend", "inductor")),
        mode=getattr(config, "torch_compile_mode", "max-autotune-no-cudagraphs"),
        fullgraph=_config_bool(getattr(config, "torch_compile_fullgraph", False)),
        dynamic=_config_bool(getattr(config, "torch_compile_dynamic", False)),
        options=_maybe_to_dict(getattr(config, "torch_compile_options", None)),
        suppress_errors=_config_bool(getattr(config, "torch_compile_suppress_errors", True), default=True),
    )
    if local_rank == 0:
        status = "enabled" if compiled else "not enabled"
        print(f"[torch.compile] {status}: target={target}")

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config.seed + local_rank)
    config.distributed = True  # Mark as distributed for pipeline
else:
    local_rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False  # Mark as non-distributed

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

torch.set_grad_enabled(False)


# Initialize pipeline
pipeline = CausalDiffusionInferencePipeline(config, device=device)

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

merge_lora = bool(getattr(config, "merge_lora", False))
has_lora_adapter = bool(getattr(config, "adapter", None) and configure_lora_for_model is not None)
if has_lora_adapter and (
    bool(getattr(config, "model_quant", False))
    or bool(getattr(config, "fp8_quant", False))
) and not merge_lora:
    if local_rank == 0:
        print(
            "[quant][LoRA] merge_lora=false is unsupported with quantization; "
            "forcing merge_lora=true so the LoRA is folded into the BF16 base before quantization."
        )
    merge_lora = True
    config.merge_lora = True
materialize_quantized_weights_for_inference = None
generator_checkpoint = None
generator_lora_state = None
lora_checkpoint = None
generator_ckpt_path = getattr(config, "generator_ckpt", None)
loaded_prequantized_generator = False
prequantized_generator_backend = None
if generator_ckpt_path:
    generator_checkpoint = torch.load(generator_ckpt_path, map_location="cpu")
    is_lora_only_checkpoint = (
        isinstance(generator_checkpoint, dict)
        and "generator_lora" in generator_checkpoint
        and not any(key in generator_checkpoint for key in ("generator", "generator_ema", "model"))
    )
    if is_lora_only_checkpoint:
        generator_lora_state = generator_checkpoint["generator_lora"]
        if local_rank == 0:
            print(f"Found LoRA generator weights in {generator_ckpt_path}")
    else:
        raw_gen_state_dict = unwrap_generator_state_dict(generator_checkpoint, use_ema=config.use_ema)
        if config.use_ema:
            raw_gen_state_dict = clean_fsdp_state_dict_keys(raw_gen_state_dict)
        if is_te_nvfp4_checkpoint(generator_checkpoint):
            raise ValueError(
                "This checkpoint was saved as a TransformerEngine module state_dict, "
                "which is not packed NVFP4 and is no longer a supported export format. "
                "Regenerate with `--backend transformer_engine` to save merged bf16 weights "
                "for TE runtime quantization, or use `--backend fouroversix` for a compact "
                "materialized NVFP4 checkpoint."
            )
        elif is_nvfp4_state_dict(raw_gen_state_dict):
            if not getattr(config, "model_quant", False):
                raise ValueError(
                    "generator_ckpt is a materialized NVFP4 checkpoint, but model_quant is false. "
                    "Set model_quant: true in the inference yaml."
                )
            if getattr(config, "model_quant_use_transformer_engine", False):
                raise ValueError(
                    "Materialized NVFP4 generator checkpoints use FourOverSix modules. "
                    "Set model_quant_use_transformer_engine: false when loading this checkpoint."
                )
            if local_rank == 0:
                print(f"[NVFP4] Loading materialized generator checkpoint from {generator_ckpt_path}")
            pipeline.generator.model, matched_modules = quantize_model_for_fouroversix_nvfp4(
                pipeline.generator.model,
                config=config,
                keep_master_weights=False,
                verbose=(local_rank == 0),
            )
            dropped_modules = drop_fouroversix_master_weights(pipeline.generator.model)
            pipeline.generator.load_state_dict(raw_gen_state_dict, strict=True)
            loaded_prequantized_generator = True
            prequantized_generator_backend = "fouroversix"
            if local_rank == 0:
                print(
                    "[NVFP4] Prepared quantized generator architecture: "
                    f"{len(dropped_modules)} materialized modules, "
                    f"{len(matched_modules)} modules excluded"
                )
        elif config.use_ema:
            missing, unexpected = pipeline.generator.load_state_dict(raw_gen_state_dict, strict=False)
            if local_rank == 0:
                if len(missing) > 0:
                    print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
                if len(unexpected) > 0:
                    print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")
        else:
            print(f"Loading generator from {generator_ckpt_path}")
            pipeline.generator.load_state_dict(raw_gen_state_dict, strict=True)

pipeline.is_lora_enabled = False
pipeline.is_lora_merged = False

if loaded_prequantized_generator:
    if has_lora_adapter or merge_lora or getattr(config, "lora_ckpt", None):
        if local_rank == 0:
            print("[NVFP4] Pre-quantized generator checkpoint is already saved with merged weights; skipping LoRA setup")
    has_lora_adapter = False
    merge_lora = False
    config.merge_lora = False

if getattr(config, "model_quant", False) and not merge_lora and not loaded_prequantized_generator:
    pipeline.generator.model, materialize_quantized_weights_for_inference = quantize_generator_model(
        pipeline.generator.model,
        config=config,
        keep_master_weights=has_lora_adapter,
    )

if has_lora_adapter:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
        if merge_lora:
            print("LoRA weights will be merged into the base model before inference")
    # Apply LoRA to the generator transformer after loading base weights.
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # Load LoRA weights from lora_ckpt. If omitted, fall back to generator_ckpt
    # when it directly contains generator_lora.
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA weights from lora_ckpt: {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    elif generator_lora_state is not None:
        if local_rank == 0:
            print(f"Loading LoRA weights from generator_ckpt: {generator_ckpt_path}")
        peft.set_peft_model_state_dict(pipeline.generator.model, generator_lora_state)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint configured; using initialized LoRA adapters")

    if merge_lora:
        if local_rank == 0:
            print("Merging LoRA weights into generator before quantization / inference...")
        pipeline.generator.model = pipeline.generator.model.merge_and_unload(safe_merge=True)
        pipeline.is_lora_merged = True
    else:
        pipeline.is_lora_enabled = True
elif merge_lora and local_rank == 0:
    print("merge_lora=True requested but no adapter config was found; continuing without LoRA merge")

# Load query_memory_encoder before deleting checkpoints (InfMem combined path).
# LoRA InfMem training stores the adapter and QueryMemoryEncoder together in
# lora_ckpt while generator_ckpt remains the immutable full-model base.  Prefer
# that training checkpoint for memory state, falling back to generator_ckpt for
# full-model InfMem checkpoints.
_infmem_enc_loaded = False
infmem_checkpoint = generator_checkpoint
if isinstance(lora_checkpoint, dict) and any(
    key in lora_checkpoint
    for key in ("query_memory_encoder", "query_memory_encoder_ema")
):
    infmem_checkpoint = lora_checkpoint

if isinstance(infmem_checkpoint, dict):
    from utils.infinity_memory_hooks import (
        get_infmem_encoder,
        select_infmem_checkpoint_state,
    )

    enc_state, enc_state_source = select_infmem_checkpoint_state(
        infmem_checkpoint,
        use_ema=config.use_ema,
    )
    _enc = get_infmem_encoder(pipeline.generator)
    if enc_state is not None and _enc is not None:
        # Validate the saved runtime architecture before load_state_dict. This
        # turns opaque query_init size mismatches into an actionable config
        # error (for example M_tokens_per_frame=32 vs 880).
        enc_meta = infmem_checkpoint.get("query_memory_encoder_meta")
        if isinstance(enc_meta, dict):
            inner = getattr(pipeline.generator, "model", None)
            expected_meta = {
                "num_params": sum(p.numel() for p in _enc.parameters()),
                "n_encoder_layers": len(getattr(_enc, "layers", [])),
                "local_attn_size": getattr(inner, "local_attn_size", None),
                "sink_size": getattr(inner, "sink_size", None),
                "relative_rope_pmax": getattr(inner, "relative_rope_pmax", None),
                "q_frames": getattr(_enc, "Q_frames", None),
                "memory_tokens_per_frame": getattr(_enc, "M_tokens_per_frame", None),
            }
            mismatches = []
            for key, expected in expected_meta.items():
                saved = enc_meta.get(key)
                if saved is not None and expected is not None and saved != expected:
                    mismatches.append(f"{key}: checkpoint={saved}, config={expected}")
            if mismatches:
                raise RuntimeError(
                    "QueryMemoryEncoder checkpoint/config mismatch: "
                    + "; ".join(mismatches)
                    + ". Make inference model_kwargs.memory_kwargs and streaming "
                    "attention settings match the training run."
                )
        _enc.load_state_dict(enc_state, strict=True)
        # Move to the inference device; the post-pipeline check casts bf16.
        _enc.to(device=device)
        _enc.eval()
        _enc.requires_grad_(False)
        # Print loading summary.
        n_loaded = len(enc_state)
        n_params = sum(p.numel() for p in _enc.parameters())
        enc_dtype = next(_enc.parameters()).dtype
        enc_device = next(_enc.parameters()).device
        # Checksum for verification (saved).
        saved_stats = {
            "n_tensors": n_loaded,
            "total_sum": sum(v.float().sum().item() for v in enc_state.values()),
        }
        if local_rank == 0:
            print(
                f"[InfMem] Loaded query_memory_encoder for inference: "
                f"source={enc_state_source}, use_ema={config.use_ema}, "
                f"tensors={n_loaded}, params={n_params:,}, "
                f"dtype={enc_dtype}, device={enc_device}, "
                f"saved_checksum={saved_stats['total_sum']:.4f}",
                flush=True,
            )
        _infmem_enc_loaded = True
    elif enc_state is not None:
        # checkpoint has encoder but the selected wrapper has no encoder
        # attached — this is a configuration mismatch. Default fail-fast
        # unless explicitly allowed.
        if not bool(getattr(config, "allow_drop_infmem_checkpoint", False)):
            raise RuntimeError(
                "Checkpoint contains QueryMemoryEncoder weights, but the "
                "selected wrapper has no encoder attached. Set "
                "allow_drop_infmem_checkpoint=true to explicitly discard "
                "the encoder weights."
            )
        if local_rank == 0:
            print("[InfMem][WARN] query_memory_encoder in checkpoint dropped "
                  "(allow_drop_infmem_checkpoint=true).")
    elif _enc is not None:
        # Check if the wrapper has an encoder but the checkpoint doesn't.
        raise RuntimeError(
            "Pipeline has a QueryMemoryEncoder but the checkpoint does not "
            "contain compatible QueryMemoryEncoder weights. Refusing to use a "
            "randomly-initialized encoder for inference."
        )

del infmem_checkpoint
del lora_checkpoint
del generator_checkpoint


# Move pipeline to appropriate dtype and device
if loaded_prequantized_generator:
    pipeline.text_encoder.to(dtype=torch.bfloat16)
    pipeline.vae.to(dtype=torch.bfloat16)
else:
    pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)

if getattr(config, "model_quant", False) and not loaded_prequantized_generator:
    if merge_lora:
        pipeline.generator.model, materialize_quantized_weights_for_inference = quantize_generator_model(
            pipeline.generator.model,
            config=config,
            keep_master_weights=False,
        )
        stage_desc = "after LoRA merge" if pipeline.is_lora_merged else "for inference"
    else:
        stage_desc = "after LoRA wrapping" if pipeline.is_lora_enabled else "for inference"
    materialize_quantized_generator(
        pipeline.generator.model,
        device=device,
        materialize_fn=materialize_quantized_weights_for_inference,
        stage_desc=stage_desc,
    )
elif loaded_prequantized_generator and local_rank == 0:
    print(f"[NVFP4] Using pre-saved {prequantized_generator_backend} generator weights from checkpoint")

pipeline.generator.model.eval().requires_grad_(False)
if bool(getattr(config, "fp8_quant", False)):
    from utils.fp8 import quantize_model_fp8

    quantize_model_fp8(pipeline.generator.model, verbose=(local_rank == 0))
configure_generator_torch_compile(pipeline, config)

# ---- InfMem encoder post-pipeline verification ----
# After all pipeline ``.to()`` calls, re-verify the encoder is on the right
# device/dtype. Echo-Infinity runs the external encoder in the runtime dtype.
if _infmem_enc_loaded:
    from utils.infinity_memory_hooks import (
        get_infmem_encoder, move_infmem_encoder, state_dict_stats,
    )
    enc = get_infmem_encoder(pipeline.generator)
    if enc is not None:
        _target_dtype = torch.bfloat16
        _dev_ok = all(p.device == device for p in enc.parameters())
        _dtype_ok = all(p.dtype == _target_dtype for p in enc.parameters())
        if not (_dev_ok and _dtype_ok):
            if local_rank == 0:
                print(
                    f"[InfMem] encoder post-pipeline check: device_ok={_dev_ok}, "
                    f"dtype_ok={_dtype_ok} — forcing {_target_dtype} + correct device."
                )
            move_infmem_encoder(
                pipeline.generator,
                device=device,
                dtype=_target_dtype,
                force_cast=True,
            )
        # Re-enforce eval + no-grad after the forced cast.
        enc.eval()
        enc.requires_grad_(False)
        # Checksum comparison (saved vs loaded).
        loaded_stats = state_dict_stats(enc.state_dict())
        if local_rank == 0:
            print(
                f"[InfMem] encoder verify: device={next(enc.parameters()).device}, "
                f"dtype={next(enc.parameters()).dtype}, "
                f"loaded_checksum={loaded_stats['total_sum']:.4f}, "
                f"saved_checksum={saved_stats['total_sum']:.4f}"
            )
        if abs(loaded_stats["total_sum"] - saved_stats["total_sum"]) > 1e-3:
            if local_rank == 0:
                print(
                    f"[InfMem][WARN] encoder checksum mismatch after pipeline.to(): "
                    f"saved={saved_stats['total_sum']:.4f}, "
                    f"loaded={loaded_stats['total_sum']:.4f}"
                )

vae_device_str = getattr(config, "vae_device", None)
use_dedicated_vae_device = bool(getattr(config, "streaming_vae", False)) and bool(vae_device_str)
if use_dedicated_vae_device:
    vae_device = torch.device(vae_device_str)
    pipeline.vae.to(device="cpu")
    pipeline.vae.to(device=vae_device)
    if hasattr(pipeline.vae, "mean"):
        pipeline.vae.mean = pipeline.vae.mean.to(device=vae_device)
        pipeline.vae.std = pipeline.vae.std.to(device=vae_device)
    if local_rank == 0:
        print(f"[inference] VAE on {vae_device}, diffusion on {device}")
else:
    pipeline.vae.to(device=device)
    if vae_device_str and local_rank == 0:
        print(f"[inference] Ignoring vae_device={vae_device_str} because streaming_vae is false")

# Create dataset
nfpb = getattr(config, 'num_frame_per_block', 8)
data_path = getattr(config, "data_path", None)
chunks_per_shot = getattr(config, 'chunks_per_shot', 0)
scene_cut_prefix = getattr(config, 'scene_cut_prefix', "The scene transitions. ")
if lmdb_path:
    shape = list(config.image_or_video_shape)
    lmdb_limit = int(1e8) if lmdb_max_samples < 0 else lmdb_max_samples
    dataset = IndexedCameraLMDBInferenceDataset(
        lmdb_path,
        max_samples=lmdb_limit,
        target_num_frames=config.num_output_frames,
        expected_latent_shape=tuple(int(value) for value in shape[2:]),
    )
    if len(dataset) == 0:
        raise ValueError(f"No samples found in camera LMDB: {lmdb_path}")
    collate_fn = camera_lmdb_i2v_collate_fn
    num_blocks = config.num_output_frames // nfpb
    if local_rank == 0:
        print(
            f"[data] camera LMDB mode: path={lmdb_path}, samples={len(dataset)}, "
            f"latent_frames={config.num_output_frames}."
        )
elif getattr(config, "i2v", False):
    model_name = config.model_kwargs.model_name
    frame_raw_height = list(config.image_or_video_shape)[3] * wan_default_config[model_name]["spatial_compression_ratio"]
    frame_raw_width = list(config.image_or_video_shape)[4] * wan_default_config[model_name]["spatial_compression_ratio"]
    temporal_compression_ratio = wan_default_config[model_name]["temporal_compression_ratio"]
    total_frames = (config.num_output_frames - 1) * temporal_compression_ratio + 1

    # ---------- Three-file i2v mode --------------------------------------
    # Mirror ``scripts/inference/inference_bidir_camera.py``: when the
    # config exposes ``inference.image_path`` + ``inference.prompt_path``
    # we bypass ``MultiVideoConcatDataset`` (which needs a full video/
    # caption directory tree) and instead read three line-aligned txt
    # files (image_path / prompt_path / trajectory_path). This is the
    # natural mode for AR + camera i2v inference on hand-authored prompts,
    # exactly like the bidirectional camera config does.
    inf_cfg_for_data = section_get(config, "inference", None, None) or {}
    image_path_file = getattr(config, "image_path", inf_cfg_for_data.get("image_path", None))
    prompt_path_file = getattr(config, "prompt_path", inf_cfg_for_data.get("prompt_path", None))
    if image_path_file and prompt_path_file:
        if local_rank == 0:
            print(f"[data] three-file i2v mode: image_path={image_path_file}, "
                  f"prompt_path={prompt_path_file}")
        dataset = ThreeFileI2VDataset(
            image_path_file=image_path_file,
            prompt_path_file=prompt_path_file,
            video_size=(frame_raw_height, frame_raw_width),
            num_output_frames=config.num_output_frames,
        )
        collate_fn = three_file_i2v_collate_fn
    else:
        dataset = MultiVideoConcatDataset(
            data_dir=data_path,
            video_size=(frame_raw_height, frame_raw_width),
            total_frames=total_frames,
            deterministic=True,
            num_frame_per_block=nfpb,
            temporal_compression_ratio=temporal_compression_ratio,
            target_fps=24 if "5B" in model_name else 16,
            allow_padding=getattr(config, "allow_padding", False),
            min_latent_frames=getattr(config, "min_latent_frames", 0),
            single_video_only=getattr(config, "uniform_prompt", False),
            independent_first_frame=getattr(config, "independent_first_frame", False),
            return_image=True,
            max_chunks_per_shot=getattr(config, "max_chunks_per_shot", 0),
            scene_cut_prefix=scene_cut_prefix,
        )
        collate_fn = multi_video_collate_fn
    num_blocks = config.num_output_frames // nfpb
else:
    num_blocks = config.num_output_frames // nfpb
    dataset = MultiTextConcatDataset(
        data_path=data_path,
        num_blocks=num_blocks,
        chunks_per_shot=chunks_per_shot,
        scene_cut_prefix=scene_cut_prefix,
        deterministic=True,
    )
    collate_fn = eval_collate_fn
if local_rank == 0:
    print(f"[data] data_path={data_path}, mode={getattr(dataset, '_mode', dataset.__class__.__name__)}, num_blocks={num_blocks}")
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0,
                        drop_last=False, collate_fn=collate_fn)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


# ---------- action overlay (optional) ----------------------------------
# Mirror scripts/inference/inference_bidir_camera.py: render a Genie-3-style
# WASD-cluster + rotation joystick on top of every output frame, driven by
# the per-frame relative pose extracted from the *raw* (pre-VAE-stride) c2w
# trajectory. The overlay key/joystick mapping follows the sana-WM convention
# (W/S = forward/back, A/D = strafe, joystick = yaw/pitch), regardless of
# which DSL produced the trajectory.
_inf_cfg_overlay = section_get(config, "inference", None, None) or {}
action_overlay = bool(_inf_cfg_overlay.get("action_overlay", False))
overlay_corner = str(_inf_cfg_overlay.get("overlay_corner", "bottom-left")).lower()
if action_overlay and overlay_corner not in {
    "bottom-left", "bottom-right", "top-left", "top-right",
}:
    raise ValueError(
        f"inference.overlay_corner must be one of "
        f"bottom-left/bottom-right/top-left/top-right, got {overlay_corner!r}"
    )
if action_overlay and local_rank == 0:
    print(f"[inference] action_overlay = True (corner={overlay_corner})")


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    # Three-file I2V inputs carry the original reference-frame path through
    # the dataset/collate function.  Prefer its basename for generated clips
    # so outputs can be matched to their conditioning frames without relying
    # on the line index in ``images.txt``.
    reference_image_stem = None
    reference_image_paths = batch.get("image_path") if isinstance(batch, dict) else None
    if reference_image_paths:
        if isinstance(reference_image_paths, (list, tuple)):
            reference_image_path = reference_image_paths[0]
        else:
            reference_image_path = reference_image_paths
        if isinstance(reference_image_path, (str, bytes, os.PathLike)):
            reference_image_stem = os.path.splitext(
                os.path.basename(os.fspath(reference_image_path))
            )[0]

    # Skip already-completed videos (mirrors inference_bidir_camera.py):
    # check whether the expected output file exists and is non-empty, so
    # re-running a partially-finished inference job only processes the
    # remaining clips instead of redoing everything.
    if idx < num_prompts:
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            _skip_model_type = "lora"
        elif getattr(config, 'use_ema', False):
            _skip_model_type = "ema"
        else:
            _skip_model_type = "regular"
        if dist.is_initialized():
            _skip_rank = dist.get_rank()
        else:
            _skip_rank = 0
        if reference_image_stem:
            _skip_base = (
                reference_image_stem if config.num_samples == 1
                else f'{reference_image_stem}-{0}'
            )
        elif config.save_with_index:
            _skip_base = f'rank{_skip_rank}-{idx}-0_{_skip_model_type}'
        else:
            # prompt is not known yet here, so we can only skip in index mode
            _skip_base = None
        if _skip_base is not None:
            _skip_ext = '.pt' if section_get(
                config, "inference", "save_latents_only",
                getattr(config, "save_latents_only", getattr(config, "save_latent_only", False)),
                aliases=("save_latent_only", "return_latents"),
            ) else '.mp4'
            _skip_path = os.path.join(config.output_folder, f'{_skip_base}{_skip_ext}')
            if os.path.exists(_skip_path) and os.path.getsize(_skip_path) > 0:
                print(f"[skip] rank{_skip_rank}-{idx} already exists -> {_skip_path}")
                continue

    all_video = []

    # MultiTextConcatDataset + eval_collate_fn: prompts[0] is List[str].
    block_prompts = list(batch['prompts'][0])
    # The AR pipeline expects one prompt per chunk in ``conditional_dict_list``
    # (built from ``text_prompts[0]``), where the number of chunks is
    # ``num_blocks`` (+1 when independent_first_frame is on and there is no
    # ``initial_latent`` — not our i2v case). Single-prompt inputs (e.g. the
    # three-file i2v mode) therefore need to be repeat-padded to ``num_blocks``
    # so ``conditional_dict_list[chunk_index]`` never IndexErrors. Mirror the
    # exact policy used by ``inference_sp.py``.
    if len(block_prompts) < num_blocks:
        block_prompts += [block_prompts[-1]] * (num_blocks - len(block_prompts))
    elif len(block_prompts) > num_blocks:
        block_prompts = block_prompts[:num_blocks]
    prompt = block_prompts[0]  # for filename
    prompts = [block_prompts] * config.num_samples

    shape = config.image_or_video_shape
    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, shape[2], shape[3], shape[4]], device=device, dtype=torch.bfloat16
    )
    initial_latent = None
    if getattr(config, "i2v", False):
        if lmdb_path:
            # The camera LMDB stores the VAE latent used during AR training.
            # Only frame zero is I2V conditioning; the remaining GT latents
            # are deliberately not passed to inference.
            initial_latent = batch["clean_latent"][:, :1].to(
                device=device, dtype=torch.bfloat16, non_blocking=True
            )
        else:
            image = batch["image"].to(device=device, dtype=torch.bfloat16)
            if image.ndim == 4:
                image = image.unsqueeze(2)
            elif image.ndim != 5:
                raise ValueError(f"Expected i2v image with shape [B,C,H,W] or [B,C,T,H,W], got {tuple(image.shape)}")
            initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        if initial_latent.shape[0] != config.num_samples:
            initial_latent = initial_latent.repeat(config.num_samples, 1, 1, 1, 1)
        if config.num_output_frames <= initial_latent.shape[1]:
            raise ValueError(
                f"num_output_frames must exceed the i2v conditioning frames; "
                f"got {config.num_output_frames} and {initial_latent.shape[1]}"
            )
    print("sampled_noise.device", sampled_noise.device)
    print("prompts", prompts)
    print('sampled_noise.shape', sampled_noise.shape, 'prompts', prompts)
    save_latents_only = section_get(
        config,
        "inference",
        "save_latents_only",
        getattr(config, "save_latents_only", getattr(config, "save_latent_only", False)),
        aliases=("save_latent_only", "return_latents"),
    )
    inference_kwargs = dict(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=save_latents_only,
    )
    if initial_latent is not None:
        inference_kwargs["initial_latent"] = initial_latent

    # Camera AR inference: build viewmats/Ks from trajectory if configured.
    raw_c2w_per_sample = None
    cam_algo = getattr(config, "algorithm", {}) or {}
    use_camera = bool(cam_algo.get("use_camera", False)) or bool(getattr(config, "use_camera", False))
    if use_camera and not lmdb_path:
        inf_cfg = section_get(config, "inference", None, None) or {}
        traj_path = getattr(config, "trajectory_path", inf_cfg.get("trajectory_path", None))
        traj_fmt = str(getattr(config, "trajectory_format", inf_cfg.get("trajectory_format", "worldplaygen"))).lower()
        H_lat = list(config.image_or_video_shape)[3]
        W_lat = list(config.image_or_video_shape)[4]
        F_lat = config.num_output_frames
        target_h = H_lat * 16
        target_w = W_lat * 16

        # Read one trajectory line per sample.
        if traj_path and os.path.isfile(traj_path):
            with open(traj_path) as f:
                traj_lines = [ln.strip() for ln in f if ln.strip()]
        else:
            traj_lines = []

        viewmats_list = []
        Ks_list = []
        raw_c2w_list = []
        for si in range(config.num_samples):
            traj = traj_lines[si % len(traj_lines)] if traj_lines else "w-10"
            if traj_fmt == "dreamx_action_dsl":
                from utils.dreamx_trajectory import action_to_viewmats_Ks, parse_trajectory_string
                action_seq, speeds = parse_trajectory_string(traj)
                duration = int(inf_cfg.get("duration_per_segment", 33))
                target_length = min(1 + 4 * (F_lat - 1), 1 + len(action_seq) * duration)
                vm, ks = action_to_viewmats_Ks(action_seq, speeds, duration=duration,
                                                target_length=target_length, h=target_h, w=target_w,
                                                dtype=torch.float32, device="cpu")
                if vm.shape[0] < F_lat:
                    pad = F_lat - vm.shape[0]
                    vm = torch.cat([vm, vm[-1:].expand(pad, -1, -1).clone()], dim=0)
                    ks = torch.cat([ks, ks[-1:].expand(pad, -1, -1).clone()], dim=0)
                elif vm.shape[0] > F_lat:
                    vm, ks = vm[:F_lat], ks[:F_lat]
                if action_overlay:
                    from utils.dreamx_trajectory import action_to_raw_c2w
                    raw_c2w_full = action_to_raw_c2w(
                        action_seq, speeds,
                        duration=duration, target_length=target_length,
                    )
            else:
                # sana_dsl: reuse the exact same helpers as the bidirectional
                # camera inference script (scripts/inference/inference_bidir_camera.py)
                # so the resulting (viewmats, Ks) tensors are bit-for-bit
                # identical for the same trajectory + F_lat.
                #   * poses_from_action_string rolls out the DSL to
                #     ``(F_lat, 7)`` w2c pose7 (t + quat), sub-sampling the
                #     per-raw-frame c2w rollout on the Wan2.2 latent stride.
                #   * build_viewmats_and_Ks turns those + normalized intrinsics
                #     into ``(F_lat, 4, 4)`` viewmats and ``(F_lat, 3, 3)`` Ks.
                # Sana DSL doesn't carry intrinsics, so we fall back to the
                # WorldPlayGen defaults (as bidir does via a dummy "w-1" pose
                # string) via ``poses_from_pose_str``.
                from utils.sana_camera_control import poses_from_action_string
                from utils.camera_dataset import build_viewmats_and_Ks
                from scripts.data_preprocessing.build_camera_lmdb_5b import poses_from_pose_str
                poses = poses_from_action_string(traj, F_lat)
                intrinsics_norm, _ = poses_from_pose_str("w-1", F_lat, target_h, target_w)
                vm_np, ks_np = build_viewmats_and_Ks(intrinsics_norm, poses)
                vm = torch.tensor(vm_np, dtype=torch.float32)
                ks = torch.tensor(ks_np, dtype=torch.float32)
                if action_overlay:
                    if traj_fmt == "worldplaygen":
                        from scripts.data_preprocessing.build_camera_lmdb_5b import (
                            _generate_camera_trajectory_local,
                            _parse_pose_string,
                        )
                        _raw_c2w_list = _generate_camera_trajectory_local(
                            _parse_pose_string(traj))
                        raw_c2w_full = np.stack(_raw_c2w_list, axis=0).astype(np.float32)
                    else:  # sana_dsl
                        from utils.sana_camera_control import action_string_to_c2w
                        raw_c2w_full = action_string_to_c2w(traj).astype(np.float32)
            viewmats_list.append(vm)
            Ks_list.append(ks)
            raw_c2w_list.append(raw_c2w_full if action_overlay else None)
        viewmats = torch.stack(viewmats_list, dim=0).to(device=device, dtype=torch.bfloat16)
        Ks = torch.stack(Ks_list, dim=0).to(device=device, dtype=torch.bfloat16)
        inference_kwargs["viewmats"] = viewmats
        inference_kwargs["Ks"] = Ks
        if action_overlay:
            raw_c2w_per_sample = raw_c2w_list

    if lmdb_path:
        # Override any hand-authored trajectory with the test record's exact
        # camera tensors. These are already normalized by
        # CameraLatentLMDBDataset in the same convention used for training.
        lmdb_viewmats = batch["viewmats"][:, :config.num_output_frames]
        lmdb_Ks = batch["Ks"][:, :config.num_output_frames]
        if lmdb_viewmats.shape[1] != config.num_output_frames:
            raise ValueError(
                f"LMDB sample has {lmdb_viewmats.shape[1]} camera frames, but "
                f"inference requests {config.num_output_frames}."
            )
        if lmdb_Ks.shape[1] != config.num_output_frames:
            raise ValueError(
                f"LMDB sample has {lmdb_Ks.shape[1]} intrinsic frames, but "
                f"inference requests {config.num_output_frames}."
            )
        lmdb_viewmats = lmdb_viewmats.to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
        lmdb_Ks = lmdb_Ks.to(
            device=device, dtype=torch.bfloat16, non_blocking=True
        )
        if lmdb_viewmats.shape[0] != config.num_samples:
            lmdb_viewmats = lmdb_viewmats.repeat(config.num_samples, 1, 1, 1)
            lmdb_Ks = lmdb_Ks.repeat(config.num_samples, 1, 1, 1)
        inference_kwargs["viewmats"] = lmdb_viewmats
        inference_kwargs["Ks"] = lmdb_Ks
        if action_overlay and local_rank == 0:
            print("[inference] action_overlay is unavailable for LMDB camera inputs.")

    with torch.inference_mode():
        generated = pipeline.inference(**inference_kwargs)

    if not save_latents_only:
        current_video = rearrange(generated, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)

        # Final output video
        video = 255.0 * torch.cat(all_video, dim=1)

        # Clear VAE cache
        pipeline.vae.model.clear_cache()
    else:
        latents = generated

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        # Determine model type for filename
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"
            
        for seed_idx in range(config.num_samples):
            if reference_image_stem:
                # Keep the reference name unchanged for the common one-sample
                # case; add a seed suffix only when multiple clips are made
                # from the same conditioning image.
                base_name = (
                    reference_image_stem if config.num_samples == 1
                    else f'{reference_image_stem}-{seed_idx}'
                )
            elif config.save_with_index:
                base_name = f'rank{rank}-{idx}-{seed_idx}_{model_type}'
            else:
                base_name = f'rank{rank}-{prompt[:100]}-{seed_idx}_{model_type}'

            if save_latents_only:
                latent_path = os.path.join(config.output_folder, f'{base_name}.pt')
                torch.save(latents[seed_idx].cpu(), latent_path)
            else:
                output_path = os.path.join(config.output_folder, f'{base_name}.mp4')
                fps = 24 if '5B' in config.model_kwargs.model_name else 16
                # Optional: composite WASD + joystick action overlay onto
                # each frame (mirrors scripts/inference/inference_bidir_camera.py).
                if action_overlay and raw_c2w_per_sample is not None:
                    from utils.action_overlay import apply_overlay
                    _vid = video[seed_idx]  # (T, H, W, C) float [0, 255]
                    _vid_thwc = _vid.clamp(0, 255).to(torch.uint8).cpu().numpy()
                    _vid_thwc = apply_overlay(
                        _vid_thwc, raw_c2w_per_sample[seed_idx],
                        corner=overlay_corner,
                    )
                    video[seed_idx] = torch.from_numpy(_vid_thwc).to(_vid.device).float()
                write_video(output_path, video[seed_idx], fps=fps)

            prompt_txt_path = os.path.join(config.output_folder, f'{base_name}_prompts.txt')
            save_prompts_to_txt(
                prompts[seed_idx] if isinstance(prompts[seed_idx], list) else [prompts[seed_idx]],
                prompt_txt_path,
                is_main_process=(rank == 0),
            )

    if config.inference_iter != -1 and i >= config.inference_iter:
        break
