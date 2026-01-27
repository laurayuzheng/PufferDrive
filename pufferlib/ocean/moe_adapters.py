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
        # Standard LoRA scaling
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

        # Use softmax for soft routing (consistent between train and eval)
        # This avoids train/eval mismatch where Gumbel noise masks mode collapse
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


class SocialForcesRouter(nn.Module):
    """Router that uses social forces for expert assignment.

    Computes repulsive forces from neighboring vehicles and uses these
    physics-based features to determine which driving style expert to use.
    This is similar to the social forces routing in Polysona.

    The intuition is that the social situation (crowded vs open, fast neighbors
    vs slow) should inform which driving style to use.

    Args:
        num_experts: Number of experts (K)
        hidden_dim: Hidden layer dimension for classifier
        dropout: Dropout probability
        use_context_features: If True, also use encoded context features (fused mode)
        context_dim: Dimension of context features (only used if use_context_features=True)
        social_force_dim: Dimension of social force features (default 6: mean/max/std of 2D forces)
    """

    def __init__(
        self,
        num_experts: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        use_context_features: bool = False,
        context_dim: int = 192,
        social_force_dim: int = 6,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.use_context_features = use_context_features
        self.social_force_dim = social_force_dim

        # Social force normalization
        self.social_force_norm = nn.LayerNorm(social_force_dim)

        # Determine input dimension for classifier
        if use_context_features:
            # Fused mode: combine social forces with context features
            self.fusion = nn.Sequential(
                nn.Linear(context_dim + social_force_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            classifier_input_dim = hidden_dim
        else:
            # Social forces only mode
            classifier_input_dim = social_force_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_experts),
        )

        # Temperature for Gumbel-Softmax
        self.register_buffer("temperature", torch.tensor(1.0))

    def compute_social_forces(
        self,
        partner_obs: Tensor,
        A: float = 1.0,
        B: float = 1.0,
        D: float = 5.0,
        eps: float = 1e-6,
    ) -> Tensor:
        """Compute social force features from partner observations.

        Adapted from Polysona's compute_social_forces_batch.
        Social forces model repulsive interactions between vehicles.

        Args:
            partner_obs: Partner observations (batch, max_partners, 7)
                Features per partner: [rel_x, rel_y, width, length, heading_x, heading_y, speed]
                Note: rel_x/rel_y are pre-scaled by 0.02 in observations
            A: Force magnitude parameter
            B: Force decay parameter
            D: Characteristic distance parameter
            eps: Small constant for numerical stability

        Returns:
            social_forces: Aggregated force features (batch, 6)
                [sum_force_x, sum_force_y, mean_mag, max_mag, std_mag, num_neighbors]
        """
        # Extract relative positions (undo the 0.02 scaling to get actual distances)
        # rel_x, rel_y are in ego frame, scaled by 0.02
        rel_x = partner_obs[:, :, 0] * 50.0  # Undo scaling: 1/0.02 = 50
        rel_y = partner_obs[:, :, 1] * 50.0

        # Compute distances
        dist = torch.sqrt(rel_x**2 + rel_y**2 + eps)  # (batch, max_partners)

        # Create mask for valid partners (non-zero observations)
        partner_magnitude = partner_obs.abs().sum(dim=-1)  # (batch, max_partners)
        valid_mask = (partner_magnitude > eps).float()  # (batch, max_partners)

        # Compute unit direction vectors (pointing away from neighbor = repulsive)
        dir_x = -rel_x / (dist + eps)  # (batch, max_partners)
        dir_y = -rel_y / (dist + eps)

        # Social force magnitude: A * exp((D - dist) / B) with clipping (Polysona formula)
        # Larger force when closer (dist < D)
        force_magnitude = A * torch.exp(torch.clamp((D - dist) / B, max=10.0))  # (batch, max_partners)

        # Apply valid mask (zero out forces from padding partners)
        force_magnitude = force_magnitude * valid_mask

        # Compute force vectors
        force_x = force_magnitude * dir_x  # (batch, max_partners)
        force_y = force_magnitude * dir_y

        # Sum forces over all partners (like Polysona line 584)
        sum_force_x = force_x.sum(dim=-1)  # (batch,)
        sum_force_y = force_y.sum(dim=-1)  # (batch,)

        # Aggregate force magnitudes: mean, max, std (like Polysona temporal pooling)
        # But we pool over partners instead of time
        num_valid = valid_mask.sum(dim=-1).clamp(min=1)  # (batch,)

        mean_mag = force_magnitude.sum(dim=-1) / num_valid  # (batch,)

        # For max, set invalid to 0 (force_magnitude is already masked)
        max_mag = force_magnitude.max(dim=-1).values  # (batch,)

        # Std using PyTorch's approach (Polysona uses .std())
        # Compute variance manually with masking
        mean_mag_expanded = mean_mag.unsqueeze(-1)  # (batch, 1)
        sq_diff = (force_magnitude - mean_mag_expanded * valid_mask) ** 2 * valid_mask
        var_mag = sq_diff.sum(dim=-1) / num_valid.clamp(min=1)
        std_mag = torch.sqrt(var_mag + eps)  # (batch,)

        # Normalized neighbor count (0-1 range for stability)
        max_partners = partner_obs.shape[1]
        norm_num_neighbors = num_valid / max_partners  # (batch,)

        # Concatenate all features
        social_forces = torch.stack(
            [sum_force_x, sum_force_y, mean_mag, max_mag, std_mag, norm_num_neighbors],
            dim=-1,
        )  # (batch, 6)

        return social_forces

    def forward(
        self,
        partner_obs: Tensor,
        context_features: Optional[Tensor] = None,
        hard: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass to compute expert probabilities.

        Args:
            partner_obs: Raw partner observations (batch, max_partners, 7)
            context_features: Optional encoded context features (batch, context_dim)
                Required if use_context_features=True
            hard: If True, use hard (one-hot) routing

        Returns:
            Tuple of (expert_probs, logits)
        """
        # Compute social force features
        social_forces = self.compute_social_forces(partner_obs)
        social_forces = self.social_force_norm(social_forces)

        # Build input for classifier
        if self.use_context_features:
            if context_features is None:
                raise ValueError("context_features required when use_context_features=True")
            fused = self.fusion(torch.cat([context_features, social_forces], dim=-1))
            input_features = fused
        else:
            input_features = social_forces

        # Classify to get expert logits
        logits = self.classifier(input_features)

        # Use softmax for soft routing (consistent between train and eval)
        # This avoids train/eval mismatch where Gumbel noise masks mode collapse
        if hard:
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


def reconstruction_loss(
    reconstructed: Tensor,
    target: Tensor,
) -> Tensor:
    """
    Reconstruction loss for social forces in MTR-MoE.

    This loss encourages the latent expert representation to capture
    meaningful information about the driving context (social forces).

    Args:
        reconstructed: Reconstructed social forces from router (batch, 6)
        target: Target social forces computed from observations (batch, 6)

    Returns:
        MSE loss scalar
    """
    return F.mse_loss(reconstructed, target.detach())


def expert_output_diversity_loss(
    x: Tensor,
    actor: "LoRAExpertsRL",
    atn_dim: list,
    temperature: float = 1.0,
) -> Tensor:
    """
    Output-space diversity loss to encourage experts to produce different action distributions.

    Unlike the cosine loss which operates in weight space, this loss directly penalizes
    similarity between the action distributions produced by different experts.

    The loss computes the negative pairwise KL divergence between expert action distributions,
    encouraging experts to make different decisions.

    Args:
        x: Hidden state from encoder (batch, hidden_size)
        actor: LoRAExpertsRL layer with expert_A, expert_B, weight, bias, scaling
        atn_dim: Action dimension list (for splitting logits)
        temperature: Softmax temperature for action distributions

    Returns:
        Negative mean pairwise KL divergence (minimize to maximize diversity)
    """
    batch_size = x.shape[0]
    num_experts = actor.num_experts

    # Compute base output once
    base_out = F.linear(x, actor.weight, actor.bias)  # (batch, out_features)

    # Compute all expert LoRA outputs in parallel using einsum
    # x: (batch, in_features)
    # expert_A: (num_experts, rank, in_features)
    # expert_B: (num_experts, out_features, rank)

    # Step 1: x @ A.T for each expert -> (batch, num_experts, rank)
    lora_intermediate = torch.einsum("bi,eri->ber", x, actor.expert_A)

    # Step 2: intermediate @ B.T for each expert -> (batch, num_experts, out_features)
    lora_out = torch.einsum("ber,eor->beo", lora_intermediate, actor.expert_B)

    # Total output for each expert: base_out + lora_out * scaling
    # base_out: (batch, out_features) -> (batch, 1, out_features)
    all_expert_outputs = base_out.unsqueeze(1) + lora_out * actor.scaling  # (batch, num_experts, out_features)

    # Split into action logits (handle multi-head actions like [7, 13] for accel/steer)
    # For simplicity, take the first action head if there are multiple
    first_action_dim = atn_dim[0]
    expert_logits = all_expert_outputs[:, :, :first_action_dim]  # (batch, num_experts, action_dim)

    # Convert to probabilities with temperature
    expert_probs = F.softmax(expert_logits / temperature, dim=-1)  # (batch, num_experts, action_dim)

    # Compute pairwise KL divergence between experts
    # KL(P || Q) = sum(P * log(P / Q))
    # We want symmetric divergence: 0.5 * (KL(P||Q) + KL(Q||P))

    total_kl = 0.0
    num_pairs = 0

    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            p = expert_probs[:, i, :] + EPS  # (batch, action_dim)
            q = expert_probs[:, j, :] + EPS

            # Symmetric KL divergence
            kl_pq = (p * (p.log() - q.log())).sum(dim=-1)  # (batch,)
            kl_qp = (q * (q.log() - p.log())).sum(dim=-1)  # (batch,)
            symmetric_kl = 0.5 * (kl_pq + kl_qp)

            total_kl = total_kl + symmetric_kl.mean()
            num_pairs += 1

    # Return negative mean KL (minimize this loss = maximize divergence)
    if num_pairs > 0:
        return -total_kl / num_pairs
    else:
        return torch.tensor(0.0, device=x.device)
