import unittest

import torch

try:
    import peft
    from utils.lora_utils import (
        configure_lora_for_model,
        lora_state_dict_from_full_generator,
    )
except ImportError:
    peft = None


class CausalWanAttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(4, 4)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.SiLU(),
            torch.nn.Linear(8, 4),
        )

    def forward(self, x):
        return self.q(x) + self.ffn(x)


class TinyWanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([CausalWanAttentionBlock()])
        self.head = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.head(self.blocks[0](x))


@unittest.skipIf(peft is None, "peft is not installed")
class TestLoraUtils(unittest.TestCase):
    def _wrap(self, model):
        return configure_lora_for_model(
            model,
            model_name="generator",
            lora_config={
                "type": "lora",
                "rank": 2,
                "alpha": 2,
                "dropout": 0.0,
                "verbose": False,
            },
            is_main_process=False,
        )

    def test_lora_freezes_backbone_but_not_external_memory(self):
        model = TinyWanModel()
        memory = torch.nn.Linear(4, 4)
        object.__setattr__(model, "query_memory_encoder", memory)

        wrapped = self._wrap(model)

        trainable_names = [
            name for name, param in wrapped.named_parameters() if param.requires_grad
        ]
        self.assertTrue(trainable_names)
        self.assertTrue(all("lora_" in name for name in trainable_names))
        self.assertTrue(all(param.requires_grad for param in memory.parameters()))
        self.assertFalse(wrapped.base_model.model.head.weight.requires_grad)

    def test_extracts_adapter_from_full_generator_state(self):
        wrapped = self._wrap(TinyWanModel())
        full_generator_state = {
            f"model.{name}": tensor.detach().clone()
            for name, tensor in wrapped.state_dict().items()
        }

        adapter_state = lora_state_dict_from_full_generator(
            wrapped,
            full_generator_state,
        )

        self.assertTrue(adapter_state)
        self.assertTrue(all("lora_" in name for name in adapter_state))
        target = self._wrap(TinyWanModel())
        result = peft.set_peft_model_state_dict(target, adapter_state)
        self.assertFalse(
            [name for name in result.missing_keys if "lora_" in name]
        )
        self.assertFalse(result.unexpected_keys)


if __name__ == "__main__":
    unittest.main()
