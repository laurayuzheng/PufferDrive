import math
import random
import itertools
from typing import Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.nn.init import calculate_gain
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


EPS = 1e-12
# ALPHA = 16

class STEArgmax(torch.autograd.Function):
    '''pass-through argmax. assumes dim to argmax over is last dim.'''
    @staticmethod
    def forward(ctx, input):
        result = torch.argmax(input, dim=-1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    

class SkilledModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._task_ids = None

    @property
    def task_ids(self):
        return self._task_ids

    @task_ids.setter
    def task_ids(self, value):
        self._task_ids = value


class HyperLoRALinear(SkilledModule):
    """Applies a linear function parameterised by a base bias
    and a weighted average of base and task-conditioned weights
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        n_tasks: int,
        n_skills: int,
        skills: Optional[Tensor],
        weight: Tensor,
        bias: Optional[Tensor],
        r: int = 16,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.r = r
        self.alpha = self.r * 2.0
        self.argmax_fn = STEArgmax()

        self.task_embs = nn.Embedding(n_tasks, n_skills)
        self.task_proj = nn.Sequential(
            nn.Linear(n_skills, n_skills),
            nn.ReLU(),
            nn.Linear(n_skills, n_skills),
        )

        self.weight = nn.Parameter(weight.detach().clone())
        self.weight.requires_grad = not freeze

        self.hyper_weight_A = nn.Linear(n_skills, r * self.in_features, bias=False)
        self.hyper_weight_B = nn.Linear(n_skills, self.out_features * r, bias=False)
        self.scaling = self.alpha / self.r

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
            self.bias.requires_grad = not freeze
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.hyper_weight_B.weight)

    def forward(self, input: Tensor) -> Tensor:
        # Provisions for inputs repeated for generation
        assert self.task_ids is not None
        assert input.size()[0] % self.task_ids.size(0) == 0

        task_ids = self.argmax_fn.apply(self.task_ids).reshape(-1)
        input_reshaped = input.clone().reshape(-1, self.in_features) # (b, in_features)

        repeats = input_reshaped.size()[0] // task_ids.size(0)
        if repeats > 1:
            task_ids = torch.repeat_interleave(task_ids, repeats, dim=0)

        task_embs = self.task_embs(task_ids)
        task_embs = self.task_proj(task_embs)

        hyper_weight_A = self.hyper_weight_A(task_embs).view(input_reshaped.size()[0], self.in_features, self.r)
        hyper_weight_B = self.hyper_weight_B(task_embs).view(input_reshaped.size()[0], self.r, self.out_features)
        output = torch.einsum("bi,bir->br", input_reshaped, hyper_weight_A)  
        output = torch.einsum("br,bro->bo", output, hyper_weight_B) 
        output = output.reshape(*input.shape[:-1], self.out_features)
        output = F.linear(input, self.weight, self.bias) + output * self.scaling

        return output


class SkilledLoRALinear(SkilledModule):
    """Applies a linear function parameterised by a base bias
    and a weighted average of base and skill weights
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        n_tasks: int,
        n_skills: int,
        skills: Optional[Tensor],
        weight: Tensor,
        bias: Optional[Tensor],
        r: int = 16,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.r = r
        self.alpha = self.r * 1.0
        self.n_tasks = n_tasks
        self.n_skills = n_skills

        if skills is None:
            self.skill_logits = nn.Parameter(torch.empty((n_tasks, n_skills)).uniform_(-1e-3, 1e-3))
            self.is_learned = True
        else:
            self.register_buffer("skill_logits", skills)
            self.is_learned = False

        self.weight = nn.Parameter(weight.detach().clone())
        self.weight.requires_grad = not freeze

        skills_weight_A = weight.new_empty((n_skills, r * self.in_features))
        skills_weight_B = weight.new_empty((n_skills, self.out_features * r))
        self.skills_weight_A = nn.Parameter(skills_weight_A)
        self.skills_weight_B = nn.Parameter(skills_weight_B)
        self.scaling = self.alpha / self.r

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
            self.bias.requires_grad = not freeze
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = calculate_gain(nonlinearity="leaky_relu")
        bound = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            torch.nn.init.uniform_(self.skills_weight_A, -bound, bound)
            torch.nn.init.zeros_(self.skills_weight_B)

    def forward(self, input: Tensor) -> Tensor:
        # Provisions for inputs repeated for generation
        assert input.size()[0] % self.task_ids.size(0) == 0, f"first dimension of {input.size()} not divisible by first dimension of {self.task_ids.size()}"

        input_leading_dim = input.size()[:-1]

        added_dim = len(input_leading_dim)
        input = input.unsqueeze(added_dim)
        task_ids = self.task_ids.reshape(-1, self.n_tasks)

        repeats = input.size()[0] // task_ids.size(0)
        if repeats > 1:
            task_ids = torch.repeat_interleave(task_ids, repeats, dim=0)

        # (b, n_tasks) @ (n_tasks, n_skills) -> (b, n_skills)
        skill_logits = task_ids @ self.skill_logits

        if self.is_learned:
            if self.training:
                skill_logits = RelaxedBernoulli(temperature=1.0, logits=skill_logits).rsample()
            else:
                skill_logits = torch.sigmoid(skill_logits)
        skill_logits = skill_logits / (skill_logits.sum(dim=-1, keepdim=True) + EPS)
        skills_weight_A = torch.mm(skill_logits, self.skills_weight_A).reshape(
            *input_leading_dim, self.in_features, self.r
        )
        skills_weight_B = torch.mm(skill_logits, self.skills_weight_B).reshape(
            *input_leading_dim, self.r, self.out_features
        )

        output = input @ skills_weight_A  # bsi,bir->bsr
        output = torch.matmul(output, skills_weight_B)  # bsr,bro->bso

        if added_dim:
            input = input.squeeze(added_dim)
            output = output.squeeze(added_dim)

        output = F.linear(input, self.weight, self.bias) + output * self.scaling
        return output


class LoRAExperts(SkilledModule):
    """LoRA experts which mix weights with task id logits.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        n_tasks: int,
        n_skills: int,
        skills: Optional[Tensor],
        weight: Tensor,
        bias: Optional[Tensor],
        r: int = 16,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.r = r
        self.alpha = self.r * 4.0
        self.n_experts = n_tasks

        self.weight = nn.Parameter(weight.detach().clone())
        self.weight.requires_grad = not freeze

        expert_weight_A = weight.new_empty((self.n_experts, r, self.in_features))
        expert_weight_B = weight.new_empty((self.n_experts, self.out_features, r))
        self.expert_weight_A = nn.Parameter(expert_weight_A)
        self.expert_weight_B = nn.Parameter(expert_weight_B)
        self.scaling = self.alpha / self.r
        # self.expert_gates = nn.Parameter(torch.zeros(self.n_experts))
        self.dropout = nn.Dropout(p=0.1)

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
            self.bias.requires_grad = not freeze
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = calculate_gain(nonlinearity="leaky_relu")
        bound = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            torch.nn.init.uniform_(self.expert_weight_A, -bound, bound)
            torch.nn.init.zeros_(self.expert_weight_B)

    def forward(self, input: Tensor) -> Tensor:
        # Provisions for inputs repeated for generation
        assert input.size()[0] % self.task_ids.size(0) == 0
        
        input_reshaped = input.clone().reshape(-1, self.in_features)
        expert_ids = self.task_ids.reshape(-1, self.n_experts)  # (b, n_experts)
        # expert_gates = torch.nn.functional.softplus(self.expert_gates)  # (n_experts,)

        repeats = input_reshaped.size()[0] // expert_ids.size(0)
        if repeats > 1:
            expert_ids = torch.repeat_interleave(expert_ids, repeats, dim=0)
        
        # expert_ids = expert_ids * expert_gates.unsqueeze(0)  # (b, n_experts)
        expert_ids = expert_ids / (expert_ids.sum(dim=-1, keepdim=True) + EPS)
        
        expert_weight_A = (expert_ids.unsqueeze(-1).unsqueeze(-1) * self.expert_weight_A.unsqueeze(0)).sum(
            dim=1
        )  # (b, r, in_features)
        expert_weight_B = (expert_ids.unsqueeze(-1).unsqueeze(-1) * self.expert_weight_B.unsqueeze(0)).sum(
            dim=1
        )  # (b, out_features, r)

        output = torch.einsum(
            "bi,bri->br", input_reshaped, expert_weight_A
        )  # (b, in_features) @ (b, r, in_features) -> (b, r)
        output = torch.einsum(
            "br,bor->bo", output, expert_weight_B
        )  # (b, r) @ (b, out_features, r) -> (b, out_features)
        output = self.dropout(output)  # (b, out_features)
        
        output = F.linear(input_reshaped, self.weight, self.bias) + output * self.scaling
        output = output.reshape(*input.shape[:-1], self.out_features)
        return output


class LoRAExpertsOutput(SkilledModule):
    """LoRA experts which mix outputs with task id logits.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        n_tasks: int,
        n_skills: int,
        skills: Optional[Tensor],
        weight: Tensor,
        bias: Optional[Tensor],
        r: int = 16,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.r = r
        self.alpha = self.r * 4.0
        self.n_experts = n_tasks

        self.weight = nn.Parameter(weight.detach().clone())
        self.weight.requires_grad = not freeze

        expert_weight_A = weight.new_empty((self.n_experts, r, self.in_features))
        expert_weight_B = weight.new_empty((self.n_experts, self.out_features, r))
        self.expert_weight_A = nn.Parameter(expert_weight_A)
        self.expert_weight_B = nn.Parameter(expert_weight_B)
        # self.expert_gates = nn.Parameter(torch.zeros(self.n_experts))
        self.dropout = nn.Dropout(p=0.1)
        self.scaling = self.alpha / self.r

        # self.residual_adapter = ResidualAdapter(self.in_features, self.out_features, self.r)

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
            self.bias.requires_grad = not freeze
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = calculate_gain(nonlinearity="leaky_relu")
        bound = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            torch.nn.init.uniform_(self.expert_weight_A, -bound, bound)
            torch.nn.init.zeros_(self.expert_weight_B)

    def forward(self, input: Tensor) -> Tensor:
        # Provisions for inputs repeated for generation
        assert input.size()[0] % self.task_ids.size(0) == 0

        input_reshaped = input.clone().reshape(-1, self.in_features) # (b, in_features)
        mixing_weights = self.task_ids.reshape(-1, self.n_experts)  # (b, n_experts)
        # expert_gates = torch.nn.functional.softplus(self.expert_gates)  # (n_experts,)

        repeats = input_reshaped.size()[0] // mixing_weights.size(0)
        if repeats > 1:
            mixing_weights = torch.repeat_interleave(mixing_weights, repeats, dim=0)

        # mixing_weights = mixing_weights * expert_gates.unsqueeze(0)
        mixing_weights = mixing_weights / (mixing_weights.sum(dim=-1, keepdim=True) + EPS)
        expert_weight_A = self.expert_weight_A.unsqueeze(0)  # (1, n_experts, r, in_features)
        expert_weight_B = self.expert_weight_B.unsqueeze(0)  # (1, n_experts, out_features, r)

        output = torch.einsum(
            "bi,beri->ber", input_reshaped, expert_weight_A
        )  # (b, in_features) @ (b, n_experts, r, in_features) -> (b, n_experts, r)
        output = torch.einsum(
            "ber,beor->beo", output, expert_weight_B
        )  # (b, n_experts, r) @ (b, n_experts, out_features, r) -> (b, n_experts, out_features)
        # collapse the 1st dimension with the mixing weights

        output = self.dropout(output)  # (b, out_features)
        lora_output = torch.einsum("be,beo->bo", mixing_weights, output)

        base_output = F.linear(input_reshaped, self.weight, self.bias)
        # resid_output = self.residual_adapter(input_reshaped) # layer-specific adjustment

        # output = W0(x) + LoRA(x) + residual(x) + b
        output = base_output + lora_output * self.scaling # + resid_output
        output = output.reshape(*input.shape[:-1], self.out_features)
        return output
    
    @staticmethod
    def gram_schmidt(Y: torch.Tensor, eps=1e-8):
        """
        Classical Gram-Schmidt orthonormalization along the row (expert) dimension.
        Args:
            Y: Tensor of shape (batch_size, num_vectors, dim)
        Returns:
            Q: Orthonormalized tensor of same shape
        """
        batch_size, n_vecs, dim = Y.shape
        Q = []
        
        for i in range(n_vecs):
            v = Y[:, i]  # shape: (batch, dim)
            # Subtract projections onto previous vectors
            for j in range(i):
                qj = Q[j]
                proj = (v * qj).sum(dim=-1, keepdim=True) * qj  # shape: (batch, dim)
                v = v - proj
            # Normalize
            norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
            Q.append(v / norm)

        return torch.stack(Q, dim=1)  # shape: (batch, n_vecs, dim)
    
class MoV(SkilledModule):
    """LoRA experts which mix outputs with task id logits.
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        n_tasks: int,
        n_skills: int,
        skills: Optional[Tensor],
        weight: Tensor,
        bias: Optional[Tensor],
        r: int = 16,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.n_experts = n_tasks
        self.weight = nn.Parameter(weight.detach().clone())
        self.weight.requires_grad = not freeze

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
            self.bias.requires_grad = not freeze
        else:
            self.register_parameter("bias", None)

        # implemented based on https://arxiv.org/pdf/2309.05444
        self.mov_vectors = nn.Parameter(torch.ones(self.n_experts, self.in_features))
        self.mov_router = nn.Sequential(
            nn.Linear(self.in_features, self.n_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, input: Tensor) -> Tensor:
        # task ids is not needed for molora

        router_probs = self.mov_router(input)  # (b, _, n_experts)
        mov_combined = torch.einsum("...e,ed->...d", router_probs, self.mov_vectors) # (b, in_features)
        return F.linear(input * mov_combined, self.weight, self.bias)
    

class ResidualAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, bottleneck_dim):
        super().__init__()
        self.down = nn.Linear(in_dim, bottleneck_dim)
        self.nonlin = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, out_dim)
        self.scale = nn.Parameter(torch.Tensor([0.01]))

    def forward(self, x):
        return self.scale * self.up(self.nonlin(self.down(x)))
    
class CPolyLoRALinear(SkilledModule):
    """Applies a linear function parameterised by a base bias
    and a weighted average of base and skill weights
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        n_tasks: int,
        n_skills: int,
        skills: Optional[Tensor],
        weight: Tensor,
        bias: Optional[Tensor],
        r: int = 16,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.r = r
        self.n_skills = n_skills
        self.n_tasks = n_tasks
        self.n_experts = n_tasks
        self.alpha = self.r * 2.0
        self.argmax_fn = STEArgmax()

        if skills is None:
            self.skill_logits_A = weight.new_empty((n_tasks, n_skills)).uniform_(-1e-3, 1e-3)
            self.skill_logits_B = torch.eye(n_tasks, device=weight.device)
            self.skill_logits = nn.Parameter(torch.cat([self.skill_logits_A, self.skill_logits_B], dim=-1))
            self.is_learned = True
        else:
            self.register_buffer("skill_logits", skills)
            self.is_learned = False

        self.weight = nn.Parameter(weight.detach().clone())
        self.weight.requires_grad = not freeze

        skills_weight_A = weight.new_empty((n_skills + n_tasks, r * self.in_features))
        skills_weight_B = weight.new_empty((n_skills + n_tasks, self.out_features * r))
        self.skills_weight_A = nn.Parameter(skills_weight_A)
        self.skills_weight_B = nn.Parameter(skills_weight_B)
        self.scaling = self.alpha / self.r

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
            self.bias.requires_grad = not freeze
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = calculate_gain(nonlinearity="leaky_relu")
        bound = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            torch.nn.init.uniform_(self.skills_weight_A, -bound, bound)
            torch.nn.init.zeros_(self.skills_weight_B)

    def forward(self, input: Tensor) -> Tensor:
        # Provisions for inputs repeated for generation

        assert self.task_ids is not None
        assert input.size()[0] % self.task_ids.size(0) == 0

        # treat all dims except last like batch dimension
        task_ids = self.argmax_fn.apply(self.task_ids).reshape(-1)
        input_reshaped = input.clone().reshape(-1, self.in_features) # (b, in_features)

        repeats = input_reshaped.size()[0] // task_ids.size(0)
        if repeats > 1:
            task_ids = torch.repeat_interleave(task_ids, repeats, dim=0)

        assert task_ids.size(0) == input_reshaped.size(0)

        skill_logits = self.skill_logits[task_ids]

        if self.is_learned:
            if self.training:
                skill_logits[:, : self.n_skills] = RelaxedBernoulli(
                    temperature=1.0, logits=skill_logits[:, : self.n_skills]
                ).rsample()
            else:
                skill_logits[:, : self.n_skills] = torch.sigmoid(skill_logits.clone()[:, : self.n_skills])

        skill_logits[:, : self.n_skills] = skill_logits.clone()[:, : self.n_skills] / (
            skill_logits.clone()[:, : self.n_skills].sum(dim=-1, keepdim=True) + EPS
        )

        skills_weight_A = torch.mm(skill_logits, self.skills_weight_A).reshape(
            input_reshaped.size()[0], self.in_features, self.r
        )
        skills_weight_B = torch.mm(skill_logits, self.skills_weight_B).reshape(
            input_reshaped.size()[0], self.r, self.out_features
        )
        output = torch.einsum("bi,bir->br", input_reshaped, skills_weight_A)  
        output = torch.einsum("br,bro->bo", output, skills_weight_B) 
        output = output.reshape(*input.shape[:-1], self.out_features)
        output_original = F.linear(input, self.weight, self.bias)
        output = output_original + output * self.scaling

        return output


class SkilledLTSFTLinear(SkilledModule):
    """Applies a linear function parameterised by a base bias
    and a weighted average of base and skill weights
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        n_tasks: int,
        n_skills: int,
        skills: Optional[Tensor],
        weight: Tensor,
        bias: Optional[Tensor],
        r: int = 16,
        density: float = 0.1,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape

        if skills is None:
            self.skill_logits = nn.Parameter(torch.empty((n_tasks, n_skills)).uniform_(-1e-3, 1e-3))
            self.is_learned = True
        else:
            self.register_buffer("skill_logits", skills)
            self.is_learned = False

        self.weight = nn.Parameter(weight.detach().clone())
        self.weight.requires_grad = not freeze

        indices = itertools.product(range(self.out_features * self.in_features), range(n_skills))
        k = int(self.out_features * self.in_features * n_skills * density)
        indices = random.sample(list(indices), k=k)
        indices = torch.LongTensor(indices).T
        values = torch.zeros((k,))
        skills_weight = torch.sparse_coo_tensor(indices, values, (self.out_features * self.in_features, n_skills))
        self.skills_weight = nn.Parameter(skills_weight.coalesce())
        self.argmax_fn = STEArgmax()

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
            self.bias.requires_grad = not freeze
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        # Provisions for inputs repeated for generation
        assert self.task_ids is not None
        assert input.size()[0] % self.task_ids.size(0) == 0

        # treat all dims except last like batch dimension
        task_ids = self.argmax_fn.apply(self.task_ids).reshape(-1)
        input_reshaped = input.clone().reshape(-1, self.in_features) # (b, in_features)

        repeats = input_reshaped.size()[0] // task_ids
        if repeats > 1:
            task_ids = torch.repeat_interleave(task_ids, repeats, dim=0)

        skill_logits = self.skill_logits[task_ids]
        if self.is_learned:
            if self.training:
                skill_logits = RelaxedBernoulli(temperature=1.0, logits=skill_logits).rsample()
            else:
                skill_logits = torch.sigmoid(skill_logits)
        skill_logits = skill_logits / (skill_logits.sum(dim=-1, keepdim=True) + EPS)

        skills_weight = torch.sparse.mm(self.skills_weight, skill_logits.T).T.view(
            input_reshaped.size()[0], self.in_features, self.out_features
        )
        output = torch.matmul(input_reshaped, skills_weight)  # bsi,bio->bso
        output = F.linear(input, self.weight, self.bias) + output.reshape(*input.shape[:-1], self.out_features)

        return output
