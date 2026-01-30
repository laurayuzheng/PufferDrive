import logging
import math
from typing import Union

import torch
from torch import nn

from .adapters import (
    SkilledLoRALinear,
    CPolyLoRALinear,
    LoRAExperts,
    LoRAExpertsOutput,
    HyperLoRALinear,
    SkilledLTSFTLinear,
    MoV,
)

from .adapter_utils import replace_layers, inform_layers

VARIANT2CLASS = {
    "cpoly": (CPolyLoRALinear, True),
    "mdsi": (SkilledLoRALinear, True),
    "private": (SkilledLoRALinear, False),
    "expert": (LoRAExperts, True),
    "expert_output": (LoRAExpertsOutput, True),
    "hyper": (HyperLoRALinear, True),
    "learned": (SkilledLoRALinear, True),
    "sparse": (SkilledLTSFTLinear, False),
    "mov": (MoV, True),
}


class SkilledMixin(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        n_tasks: int,
        n_skills: int,
        skilled_variant: str = "private",
        freeze: bool = True,
        custom_skills: Union[str, None] = None,
        rank=16,
        state_dict=None,
        attention_only: bool = True,
    ):
        super().__init__()
        self.model = model
        self.n_tasks = n_tasks
        self.n_skills = n_skills
        self.skilled_variant = skilled_variant
        self.lora_rank = rank

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        adapter_class = VARIANT2CLASS.get(skilled_variant, (SkilledLoRALinear, True))[0]
        only_attention = attention_only

        self.adapter_class = adapter_class
        skills = self.get_skills(custom_skills)

        if skilled_variant == "private":
            n_skills = n_tasks + 1

        replace_layers(
            self.model,
            adapter_class,
            n_tasks,
            n_skills,
            skills,
            rank=self.lora_rank,
            only_attention=only_attention,
        )

        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)
            self.model.tie_weights()

    def get_skills(self, custom_skills):
        if self.skilled_variant in ["learned", "hyper", "sparse", "cpoly", "expert", "expert_output", "mov"]:
            # skills are computed inside each module
            skills = None
        elif self.skilled_variant == "shared":
            skills = torch.ones((self.n_tasks, 1))
        elif self.skilled_variant == "private":
            eye = torch.eye(self.n_tasks, self.n_tasks)
            skills = torch.cat([eye, torch.ones((self.n_tasks, 1))], dim=-1)
        elif self.skilled_variant == "custom":
            skills = custom_skills
        elif self.skilled_variant == "mdsi":
            assert self.n_skills == 8
            assert self.n_tasks == 4

            skills = torch.tensor(
                [
                    [0, 0, 1, 0, 1, 0, 0, 0],  # reckless and careless
                    [1, 1, 0, 0, 0, 1, 0, 0],  # anxious
                    [0, 0, 0, 1, 0, 0, 0, 0],  # angry
                    [0, 0, 0, 0, 0, 0, 1, 1],
                ]
            )
        else:
            raise ValueError

        return skills

    def generate(self, *args, task_ids=None, **kwargs):
        inform_layers(self.model, self.adapter_class, task_ids)
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, task_ids=None, add_prior=False, **kwargs):

        if task_ids is not None:
            inform_layers(self.model, self.adapter_class, value=task_ids)
        outputs = self.model.forward(*args, **kwargs)

        return outputs
