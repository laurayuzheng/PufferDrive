from torch import nn
import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401


Recurrent = pufferlib.models.LSTMWrapper


class Drive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6  # 6 is the number of one-hot encoded categories

        # Determine ego dimension from environment's dynamics model
        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)  # Shape: [batch, ROAD_MAX_OBJECTS, 7]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
        return embedding

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self.value_fn(flat_hidden)

        return action, value


class DriveMoE(nn.Module):
    """
    Mixture-of-Experts variant of the Drive policy for latent variable modeling.

    Uses the same encoder structure as Drive but adds:
    - A router that predicts K expert assignment probabilities
    - LoRA-based expert adapters on the actor head
    - Auxiliary losses to prevent mode collapse

    This enables learning K discrete driving style buckets.
    """

    def __init__(
        self,
        env,
        input_size=64,
        hidden_size=256,
        num_experts=3,
        lora_rank=8,
        lora_alpha=4.0,
        router_hidden_dim=64,
        freeze_base=True,
        expert_prior=None,
        **kwargs,
    ):
        super().__init__()
        from pufferlib.ocean.moe_adapters import (
            LoRAExpertsRL,
            PersonaRouter,
            kl_divergence_loss,
            entropy_loss,
            expert_cosine_loss,
            compute_temperature,
        )

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6

        # Store loss functions as methods
        self._kl_divergence_loss = kl_divergence_loss
        self._entropy_loss = entropy_loss
        self._expert_cosine_loss = expert_cosine_loss
        self._compute_temperature = compute_temperature

        # Expert prior for KL loss (e.g., [0.3, 0.6, 0.1] from open-loop)
        # If None, uniform prior is used
        if expert_prior is not None:
            self.register_buffer("expert_prior", torch.tensor(expert_prior, dtype=torch.float32))
        else:
            self.expert_prior = None

        # Determine ego dimension from environment's dynamics model
        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7

        # Encoders (same as Drive)
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        # Shared embedding (same as Drive)
        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, hidden_size)),
        )

        # Router: predicts expert probabilities from concatenated encoder features
        self.router = PersonaRouter(
            input_dim=3 * input_size,
            num_experts=num_experts,
            hidden_dim=router_hidden_dim,
        )

        # Action space setup
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        # Actor with LoRA experts
        self.actor = LoRAExpertsRL(
            in_features=hidden_size,
            out_features=sum(self.atn_dim),
            num_experts=num_experts,
            rank=lora_rank,
            alpha=lora_alpha,
            bias=True,
            freeze_base=freeze_base,
        )

        # Value function (no experts - shared across all styles)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

        # Freeze base model weights if specified (but NOT value_fn - it must adapt)
        if freeze_base:
            for param in self.ego_encoder.parameters():
                param.requires_grad = False
            for param in self.road_encoder.parameters():
                param.requires_grad = False
            for param in self.partner_encoder.parameters():
                param.requires_grad = False
            for param in self.shared_embedding.parameters():
                param.requires_grad = False
            # NOTE: value_fn is NOT frozen - it must learn to evaluate the new policy

        # Storage for auxiliary losses (computed during forward pass)
        self._aux_losses = {}
        self._expert_probs = None
        self._router_logits = None

        # Expert usage tracking for logging
        self._expert_usage_counts = None

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)

        # Router prediction (before shared embedding for richer features)
        expert_probs, router_logits = self.router(concat_features, hard=not self.training)

        # Store for auxiliary losses
        self._expert_probs = expert_probs
        self._router_logits = router_logits

        # Compute auxiliary losses during training
        if self.training:
            self._aux_losses = {
                "kl": self._kl_divergence_loss(expert_probs, prior=self.expert_prior),
                "entropy": self._entropy_loss(expert_probs),
                "cosine": self._expert_cosine_loss(
                    self.actor.expert_A, self.actor.expert_B
                ),
            }

        # Track expert usage
        with torch.no_grad():
            expert_assignments = expert_probs.argmax(dim=-1)
            self._expert_usage_counts = torch.bincount(
                expert_assignments, minlength=self.num_experts
            ).float()

        # Shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))

        return embedding

    def decode_actions(self, flat_hidden, expert_probs=None):
        if expert_probs is None:
            expert_probs = self._expert_probs

        if self.is_continuous:
            parameters = self.actor(flat_hidden, expert_probs)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action_logits = self.actor(flat_hidden, expert_probs)
            action = torch.split(action_logits, self.atn_dim, dim=1)

        value = self.value_fn(flat_hidden)

        return action, value

    def get_auxiliary_losses(self):
        """Return auxiliary losses for training integration."""
        return self._aux_losses

    def get_expert_stats(self):
        """Return expert usage statistics for logging."""
        if self._expert_usage_counts is None:
            return {}

        total = self._expert_usage_counts.sum()
        if total == 0:
            return {}

        stats = {}
        for i in range(self.num_experts):
            stats[f"expert_{i}_usage"] = (self._expert_usage_counts[i] / total).item()

        return stats

    def update_temperature(self, progress, config):
        """Update Gumbel-Softmax temperature based on training progress."""
        tau_max = config.get("moe_tau_max", 2.0)
        tau_min = config.get("moe_tau_min", 0.1)
        new_temp = self._compute_temperature(progress, tau_max, tau_min)
        self.router.set_temperature(new_temp)
