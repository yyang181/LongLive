# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Mapping

import torch
import peft
from peft import get_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig
)


def configure_lora_for_model(transformer, model_name, lora_config, is_main_process=True, all_causal=False):
    """Configure LoRA for a WanDiffusionWrapper model
    
    Args:
        transformer: The transformer model to apply LoRA to
        model_name: 'generator' or 'fake_score'
        lora_config: LoRA configuration
        is_main_process: Whether this is the main process (for logging)
        all_causal: Whether all models use causal attention blocks
    
    Returns:
        lora_model: The LoRA-wrapped model
    """
    target_linear_modules = set()
    
    if model_name == 'generator':
        adapter_target_modules = ['CausalWanAttentionBlock']
    elif model_name == 'fake_score':
        adapter_target_modules = ['CausalWanAttentionBlock'] if all_causal else ['WanAttentionBlock']
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    for name, module in transformer.named_modules():
        if module.__class__.__name__ in adapter_target_modules:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_submodule_name)
    
    target_linear_modules = sorted(target_linear_modules)

    if not target_linear_modules:
        raise ValueError(
            f"No Linear layers were found in the LoRA target blocks for {model_name}. "
            f"Expected one of: {adapter_target_modules}."
        )
    
    if is_main_process:
        print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
        if getattr(lora_config, 'verbose', False):
            for module_name in sorted(target_linear_modules):
                print(f"  - {module_name}")
    
    # Create LoRA config
    adapter_type = lora_config.get('type', 'lora')
    if adapter_type == 'lora':
        peft_config = peft.LoraConfig(
            r=lora_config.get('rank', 16),
            lora_alpha=lora_config.get('alpha', None) or lora_config.get('rank', 16),
            lora_dropout=lora_config.get('dropout', 0.0),
            target_modules=target_linear_modules,
        )
    else:
        raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
    
    # Apply LoRA to the transformer
    lora_model = peft.get_peft_model(transformer, peft_config)

    if is_main_process:
        print('peft_config', peft_config)
        lora_model.print_trainable_parameters()
    
    return lora_model


def lora_state_dict_from_full_generator(
    lora_model,
    full_generator_state_dict: Mapping[str, torch.Tensor],
    *,
    model_prefix: str = "model.",
):
    """Extract a PEFT adapter state from a full FSDP wrapper state dict.

    Diffusion SFT wraps the complete ``WanDiffusionWrapper`` in FSDP, while
    PEFT wraps only ``WanDiffusionWrapper.model``. A full FSDP state dict
    therefore prefixes every PEFT key with ``model.``. Strip exactly that
    wrapper prefix before asking PEFT to filter and normalize adapter keys.
    """
    if not isinstance(full_generator_state_dict, Mapping):
        raise TypeError(
            "full_generator_state_dict must be a mapping, "
            f"got {type(full_generator_state_dict)!r}."
        )

    model_state_dict = {
        name[len(model_prefix):]: value
        for name, value in full_generator_state_dict.items()
        if not model_prefix or name.startswith(model_prefix)
    }
    if not model_state_dict:
        raise ValueError(
            f"Full generator state contains no keys with prefix {model_prefix!r}."
        )

    adapter_state = get_peft_model_state_dict(
        lora_model,
        state_dict=model_state_dict,
    )
    if not adapter_state:
        raise ValueError("No LoRA tensors were found in the full generator state dict.")
    return adapter_state


def gather_lora_state_dict(lora_model):
    with FSDP.state_dict_type(
        lora_model,                  
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
    ):
        full = lora_model.state_dict()
    return get_peft_model_state_dict(lora_model, state_dict=full)


def load_lora_checkpoint(lora_model, lora_state_dict, model_name, is_main_process=True):
    """Load LoRA weights from state dict
    
    Args:
        lora_model: The LoRA-wrapped model
        lora_state_dict: LoRA state dict to load
        model_name: 'generator' or 'critic'
        is_main_process: Whether this is the main process (for logging)
    """
    if is_main_process:
        print(f"Loading LoRA {model_name} weights: {len(lora_state_dict)} keys in checkpoint")
    
    load_result = peft.set_peft_model_state_dict(lora_model, lora_state_dict)
    missing_lora = [
        name for name in load_result.missing_keys if "lora_" in name
    ]
    if missing_lora or load_result.unexpected_keys:
        raise RuntimeError(
            f"Invalid LoRA {model_name} checkpoint: "
            f"missing_lora={missing_lora}, "
            f"unexpected={load_result.unexpected_keys}."
        )
    
    if is_main_process:
        print(f"LoRA {model_name} weights loaded successfully")
