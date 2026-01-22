"""
Mixture-of-Experts adapters for PufferDrive closed-loop RL.

Adapted from pc_driving/polysona/utils/adapters.py for use with PPO training.
"""

import math
from typing import Optional, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import calculate_gain


EPS = 1e-12


class LoRAExpertsRL(nn.Module):
    """LoRA experts that mix weights based on router outputs.

    This is the 'expert' variant from Polysona - weights are mixed before
    computation, making it more parameter-efficient than mixing outputs.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_experts: Number of expert modules (K)
        rank: LoRA rank (low-rank factorization dimension)
        alpha: LoRA scaling factor
        bias: Whether to include bias
        freeze_base: Whether to freeze base weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 3,
        rank: int = 8,
        alpha: float = 4.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.rank = rank
        self.scaling = alpha / rank

        # Base weights (optionally frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.requires_grad = not freeze_base

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.bias.requires_grad = not freeze_base
        else:
            self.register_parameter("bias", None)

        # Expert-specific LoRA matrices
        # A: (num_experts, rank, in_features) - down projection
        # B: (num_experts, out_features, rank) - up projection
        self.expert_A = nn.Parameter(torch.empty(num_experts, rank, in_features))
        self.expert_B = nn.Parameter(torch.empty(num_experts, out_features, rank))

        self.dropout = nn.Dropout(p=0.1)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize base weights with Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Initialize LoRA weights
        gain = calculate_gain(nonlinearity="leaky_relu")
        bound = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            nn.init.uniform_(self.expert_A, -bound, bound)
            nn.init.zeros_(self.expert_B)  # Start with zero contribution

    def forward(self, x: Tensor, expert_weights: Tensor) -> Tensor:
        """
        Forward pass with expert weight mixing.

        Args:
            x: Input tensor (batch, in_features)
            expert_weights: Router output probabilities (batch, num_experts)

        Returns:
            Output tensor (batch, out_features)
        """
        batch_size = x.shape[0]

        # Normalize expert weights
        expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + EPS)

        # Mix expert LoRA weights based on router output
        # expert_weights: (batch, num_experts)
        # expert_A: (num_experts, rank, in_features)
        # mixed_A: (batch, rank, in_features)
        mixed_A = torch.einsum("be,eri->bri", expert_weights, self.expert_A)
        mixed_B = torch.einsum("be,eor->bor", expert_weights, self.expert_B)

        # Compute LoRA output: x @ A.T @ B.T
        # x: (batch, in_features)
        # mixed_A: (batch, rank, in_features) -> need (batch, in_features, rank)
        lora_out = torch.einsum("bi,bri->br", x, mixed_A)  # (batch, rank)
        lora_out = torch.einsum("br,bor->bo", lora_out, mixed_B)  # (batch, out_features)
        lora_out = self.dropout(lora_out)

        # Base output + scaled LoRA output
        base_out = F.linear(x, self.weight, self.bias)
        return base_out + lora_out * self.scaling


class PersonaRouter(nn.Module):
    """Router that predicts expert assignment probabilities.

    Uses Gumbel-Softmax for differentiable discrete routing during training.

    Args:
        input_dim: Input feature dimension
        num_experts: Number of experts (K)
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_experts),
        )

        # Temperature for Gumbel-Softmax (will be updated during training)
        self.register_buffer("temperature", torch.tensor(1.0))

    def forward(
        self, features: Tensor, hard: bool = False
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass to compute expert probabilities.

        Args:
            features: Input features (batch, input_dim)
            hard: If True, use hard (one-hot) routing

        Returns:
            Tuple of (expert_probs, logits)
            - expert_probs: (batch, num_experts) - routing probabilities
            - logits: (batch, num_experts) - raw logits for auxiliary losses
        """
        logits = self.classifier(features)

        if self.training:
            # Use Gumbel-Softmax during training for differentiable discrete routing
            expert_probs = F.gumbel_softmax(
                logits, tau=self.temperature.item(), hard=hard, dim=-1
            )
        else:
            # Use softmax during evaluation (or hard selection)
            if hard:
                # Hard selection: one-hot encoding of argmax
                indices = logits.argmax(dim=-1)
                expert_probs = F.one_hot(indices, num_classes=self.num_experts).float()
            else:
                expert_probs = F.softmax(logits, dim=-1)

        return expert_probs, logits

    def set_temperature(self, temperature: float):
        """Update Gumbel-Softmax temperature."""
        self.temperature.fill_(temperature)


def kl_divergence_loss(
    persona_probs: Tensor, prior: Optional[Tensor] = None
) -> Tensor:
    """
    KL divergence loss to encourage uniform expert usage.

    Computes KL(empirical || prior) where empirical is the batch-average
    expert assignment distribution.

    Args:
        persona_probs: Expert assignment probabilities (batch, num_experts)
        prior: Prior distribution over experts. If None, uses uniform.

    Returns:
        KL divergence scalar
    """
    num_experts = persona_probs.shape[-1]

    if prior is None:
        prior = torch.ones(num_experts, device=persona_probs.device) / num_experts

    # Compute empirical distribution (average over batch)
    empirical = persona_probs.mean(dim=0) + EPS
    empirical = empirical / empirical.sum()  # Normalize

    # KL divergence: sum(empirical * log(empirical / prior))
    return F.kl_div(empirical.log(), prior, reduction="sum")


def entropy_loss(persona_probs: Tensor) -> Tensor:
    """
    Negative entropy loss to encourage sharp (confident) routing.

    Lower entropy means more confident predictions (closer to one-hot).
    We return negative entropy so that minimizing this loss increases sharpness.

    Args:
        persona_probs: Expert assignment probabilities (batch, num_experts)

    Returns:
        Negative entropy scalar (mean over batch)
    """
    # Add small epsilon for numerical stability
    probs = persona_probs + EPS
    # Entropy: -sum(p * log(p))
    entropy = -(probs * probs.log()).sum(dim=-1)
    # Return negative entropy (we want to minimize this to get sharp predictions)
    return -entropy.mean()


def expert_cosine_loss(expert_A: Tensor, expert_B: Tensor) -> Tensor:
    """
    Cosine similarity loss to encourage diverse experts.

    Computes the delta_W = B @ A for each expert and penalizes
    similarity between experts (off-diagonal cosine similarity).

    Args:
        expert_A: LoRA A matrices (num_experts, rank, in_features)
        expert_B: LoRA B matrices (num_experts, out_features, rank)

    Returns:
        Cosine similarity loss scalar
    """
    num_experts = expert_A.shape[0]

    # Compute delta_W for each expert: B @ A -> (num_experts, out_features, in_features)
    delta_W = torch.einsum("eor,eri->eoi", expert_B, expert_A)

    # Flatten to (num_experts, out_features * in_features)
    delta_W_flat = delta_W.reshape(num_experts, -1)

    # Normalize each expert's delta_W
    normalized = F.normalize(delta_W_flat, p=2, dim=-1)

    # Compute pairwise cosine similarity
    cos_sim = normalized @ normalized.T  # (num_experts, num_experts)

    # Penalize off-diagonal similarity (want experts to be orthogonal)
    identity = torch.eye(num_experts, device=cos_sim.device)
    off_diagonal = cos_sim - identity

    # Mean squared off-diagonal similarity
    return (off_diagonal ** 2).sum() / (num_experts * (num_experts - 1))


def compute_temperature(
    progress: float, tau_max: float = 2.0, tau_min: float = 0.1
) -> float:
    """
    Compute Gumbel-Softmax temperature based on training progress.

    Uses linear annealing from tau_max to tau_min.

    Args:
        progress: Training progress in [0, 1]
        tau_max: Initial temperature (soft routing)
        tau_min: Final temperature (hard routing)

    Returns:
        Current temperature value
    """
    return tau_max - (tau_max - tau_min) * progress
