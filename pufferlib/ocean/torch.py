from torch import nn
import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401

# MTR-based policies (import for policy discovery)
from pufferlib.ocean.torch_mtr import DriveMTR, DriveMTRMoE  # noqa: F401


Recurrent = pufferlib.models.LSTMWrapper


class Drive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, predict_goal=False, goal_hidden_dim=64, **kwargs):
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

        # Goal prediction config
        self.predict_goal = predict_goal
        self._goal_predictions = None

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

        # Goal prediction head (predicts goal in ego-centric frame)
        if predict_goal:
            self.goal_head = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(hidden_size, goal_hidden_dim)),
                nn.ReLU(),
                pufferlib.pytorch.layer_init(nn.Linear(goal_hidden_dim, 2), std=0.01),  # (x, y) in ego frame
            )

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

        # Compute goal predictions if enabled
        if self.predict_goal:
            self._goal_predictions = self.goal_head(embedding)

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

    def get_goal_predictions(self):
        """Return the most recent goal predictions.

        Returns:
            Tensor of shape (batch, 2) with predicted (x, y) in ego-centric frame,
            or None if predict_goal is False.
        """
        return self._goal_predictions

    def get_goal_prediction_loss(self, gt_goals, gt_valid):
        """Compute MSE loss between predicted and logged goals.

        Args:
            gt_goals: Ground truth goals in ego-centric frame (meters), shape (batch, 2)
            gt_valid: Boolean mask for valid goals, shape (batch,)

        Returns:
            Scalar MSE loss tensor, or 0 if no valid samples.
        """
        if self._goal_predictions is None or not self.predict_goal:
            return torch.tensor(0.0, device=gt_goals.device)

        valid_mask = gt_valid.bool()
        if not valid_mask.any():
            return torch.tensor(0.0, device=gt_goals.device)

        # Scale ground truth goals to match observation space (0.005 scale factor from drive.h)
        # This makes predictions and targets in the same scale as the goal observations
        gt_goals_scaled = gt_goals * 0.005

        return F.mse_loss(self._goal_predictions[valid_mask], gt_goals_scaled[valid_mask])


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
        use_social_forces_routing=False,
        social_forces_only=False,
        predict_goal=False,
        goal_hidden_dim=64,
        **kwargs,
    ):
        """
        Args:
            use_social_forces_routing: If True, use social forces for routing instead of
                learned features. Social forces capture the "social situation" (crowding,
                neighbor velocities) to inform expert selection.
            social_forces_only: If True (and use_social_forces_routing=True), use only
                social forces for routing. If False, fuse social forces with encoded
                context features.
            predict_goal: If True, add a goal prediction head. The policy will predict
                its own goal from context (road geometry, other agents).
            goal_hidden_dim: Hidden dimension for goal prediction head.
        """
        super().__init__()
        from pufferlib.ocean.moe_adapters import (
            LoRAExpertsRL,
            PersonaRouter,
            SocialForcesRouter,
            SocialForcesRouterWithReconstruction,
            kl_divergence_loss,
            entropy_loss,
            expert_cosine_loss,
            expert_output_diversity_loss,
            reconstruction_loss,
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

        # Social forces routing config
        self.use_social_forces_routing = use_social_forces_routing
        self.social_forces_only = social_forces_only

        # Store loss functions as methods
        self._kl_divergence_loss = kl_divergence_loss
        self._entropy_loss = entropy_loss
        self._expert_cosine_loss = expert_cosine_loss
        self._expert_output_diversity_loss = expert_output_diversity_loss
        self._reconstruction_loss = reconstruction_loss
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

        # Router: predicts expert probabilities
        if use_social_forces_routing:
            # Social forces router with VAE-style reconstruction (Polysona)
            self.router = SocialForcesRouterWithReconstruction(
                num_experts=num_experts,
                hidden_dim=router_hidden_dim,
                use_context_features=not social_forces_only,
                context_dim=3 * input_size,  # concat_features dimension
            )
        else:
            # Standard router uses learned features
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

        # Goal prediction with LoRA experts
        self.predict_goal = predict_goal
        self._goal_predictions = None
        self.goal_hidden_dim = goal_hidden_dim

        if predict_goal:
            # Base goal head (shared across experts)
            self.goal_head_fc1 = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, goal_hidden_dim))
            self.goal_head_fc2 = pufferlib.pytorch.layer_init(nn.Linear(goal_hidden_dim, 2), std=0.01)

            # LoRA adapters for goal prediction (experts predict different goals)
            # A matrices: (num_experts, lora_rank, goal_hidden_dim)
            # B matrices: (num_experts, 2, lora_rank)
            self.goal_expert_A = nn.Parameter(
                torch.zeros(num_experts, lora_rank, goal_hidden_dim)
            )
            self.goal_expert_B = nn.Parameter(
                torch.zeros(num_experts, 2, lora_rank)
            )
            # Initialize A with small random values, B with zeros
            nn.init.kaiming_uniform_(self.goal_expert_A, a=5**0.5)
            nn.init.zeros_(self.goal_expert_B)
            self.goal_lora_alpha = lora_alpha
            self.goal_lora_rank = lora_rank

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
        # Reconstruction outputs (for VAE-style loss)
        self._social_forces = None
        self._reconstructed_social_forces = None

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
        if self.use_social_forces_routing:
            # Social forces router with reconstruction (returns 4 values)
            expert_probs, router_logits, social_forces, reconstructed_sf = self.router(
                partner_obs=partner_objects,
                context_features=concat_features if not self.social_forces_only else None,
                hard=not self.training,
            )
            # Store reconstruction outputs for VAE loss
            self._social_forces = social_forces
            self._reconstructed_social_forces = reconstructed_sf
        else:
            # Standard router uses encoded features (returns 2 values)
            expert_probs, router_logits = self.router(concat_features, hard=not self.training)
            self._social_forces = None
            self._reconstructed_social_forces = None

        # Store for auxiliary losses
        self._expert_probs = expert_probs
        self._router_logits = router_logits

        # Track expert usage
        with torch.no_grad():
            expert_assignments = expert_probs.argmax(dim=-1)
            self._expert_usage_counts = torch.bincount(
                expert_assignments, minlength=self.num_experts
            ).float()

        # Shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))

        # Compute goal predictions with LoRA experts if enabled
        if self.predict_goal:
            # Base goal prediction
            goal_hidden = F.relu(self.goal_head_fc1(embedding))
            base_goal = self.goal_head_fc2(goal_hidden)

            # Apply LoRA delta weighted by expert probabilities
            # expert_probs: (batch, num_experts)
            # goal_expert_A: (num_experts, lora_rank, goal_hidden_dim)
            # goal_expert_B: (num_experts, 2, lora_rank)
            # goal_hidden: (batch, goal_hidden_dim)

            # Compute LoRA output for each expert
            # goal_hidden @ A^T: (batch, goal_hidden_dim) @ (num_experts, goal_hidden_dim, lora_rank)^T
            # = (batch, goal_hidden_dim) @ (num_experts, lora_rank, goal_hidden_dim)
            # For each expert: (batch, goal_hidden_dim) @ (goal_hidden_dim, lora_rank) -> (batch, lora_rank)
            # Combined: (batch, num_experts, lora_rank)
            lora_intermediate = torch.einsum('bh,erh->ber', goal_hidden, self.goal_expert_A)
            # lora_intermediate @ B^T: (batch, num_experts, lora_rank) @ (num_experts, 2, lora_rank)^T
            # = (batch, num_experts, lora_rank) @ (num_experts, lora_rank, 2)
            # -> (batch, num_experts, 2)
            lora_outputs = torch.einsum('ber,eor->beo', lora_intermediate, self.goal_expert_B)

            # Weight by expert probabilities and sum
            # expert_probs: (batch, num_experts), lora_outputs: (batch, num_experts, 2)
            lora_delta = torch.einsum('be,beo->bo', expert_probs, lora_outputs)

            # Scale by alpha/rank
            lora_delta = lora_delta * (self.goal_lora_alpha / self.goal_lora_rank)

            self._goal_predictions = base_goal + lora_delta
        else:
            self._goal_predictions = None

        # Compute auxiliary losses during training (after embedding is ready for output_div loss)
        if self.training:
            # expert_probs are soft (from softmax) during training, so use directly
            self._aux_losses = {
                "kl": self._kl_divergence_loss(expert_probs, prior=self.expert_prior),
                "entropy": self._entropy_loss(expert_probs),
                "cosine": self._expert_cosine_loss(
                    self.actor.expert_A, self.actor.expert_B
                ),
                # Output diversity loss (gated by coefficient in pufferl.py)
                # Set to None here - it will be computed lazily if coefficient > 0
                "output_div": None,
                # Reconstruction loss (VAE-style, from Polysona)
                "reconstruction": (
                    self._reconstruction_loss(self._reconstructed_social_forces, self._social_forces)
                    if self._social_forces is not None
                    else None
                ),
            }
            # Store embedding for lazy output_div computation
            self._cached_embedding = embedding
        else:
            self._cached_embedding = None

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

    def get_auxiliary_losses(self, config=None):
        """Return auxiliary losses for training integration.

        Args:
            config: Optional config dict. If provided and aux_output_div_coef > 0,
                    computes the output diversity loss lazily.
        """
        # Compute output diversity loss lazily if coefficient is set and > 0
        if (
            config is not None
            and self._aux_losses.get("output_div") is None
            and self._cached_embedding is not None
        ):
            output_div_coef = config.get("aux_output_div_coef", 0.0)
            if output_div_coef > 0:
                self._aux_losses["output_div"] = self._expert_output_diversity_loss(
                    self._cached_embedding,
                    self.actor,
                    self.atn_dim,
                )

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

    def get_goal_predictions(self):
        """Return the most recent goal predictions.

        Returns:
            Tensor of shape (batch, 2) with predicted (x, y) in ego-centric frame,
            or None if predict_goal is False.
        """
        return self._goal_predictions

    def get_goal_prediction_loss(self, gt_goals, gt_valid):
        """Compute MSE loss between predicted and logged goals.

        Args:
            gt_goals: Ground truth goals in ego-centric frame (meters), shape (batch, 2)
            gt_valid: Boolean mask for valid goals, shape (batch,)

        Returns:
            Scalar MSE loss tensor, or 0 if no valid samples.
        """
        if self._goal_predictions is None or not self.predict_goal:
            return torch.tensor(0.0, device=gt_goals.device)

        valid_mask = gt_valid.bool()
        if not valid_mask.any():
            return torch.tensor(0.0, device=gt_goals.device)

        # Scale ground truth goals to match observation space (0.005 scale factor from drive.h)
        # This makes predictions and targets in the same scale as the goal observations
        gt_goals_scaled = gt_goals * 0.005

        return F.mse_loss(self._goal_predictions[valid_mask], gt_goals_scaled[valid_mask])

    def update_temperature(self, progress, config):
        """Update Gumbel-Softmax temperature based on training progress."""
        tau_max = config.get("moe_tau_max", 2.0)
        tau_min = config.get("moe_tau_min", 0.1)
        new_temp = self._compute_temperature(progress, tau_max, tau_min)
        self.router.set_temperature(new_temp)

    def forward_with_forced_expert(self, observations, expert_idx, state=None):
        """Forward pass with a specific expert forced (one-hot).

        Args:
            observations: Input observations.
            expert_idx: Index of the expert to force (0 to num_experts-1).
            state: Optional hidden state (unused, for API compatibility).

        Returns:
            (actions, value) tuple.
        """
        batch_size = observations.shape[0]

        # Create one-hot expert weights
        forced_probs = torch.zeros(batch_size, self.num_experts, device=observations.device)
        forced_probs[:, expert_idx] = 1.0

        # Encode observations (this sets self._expert_probs but we'll override)
        hidden = self.encode_observations(observations, state)

        # Decode with forced expert weights
        actions, value = self.decode_actions(hidden, expert_probs=forced_probs)

        return actions, value

    def get_expert_assignments(self):
        """Return expert assignments (argmax of expert_probs) from the last forward pass.

        Returns:
            Tensor of shape (batch,) with expert indices, or None if not available.
        """
        if self._expert_probs is None:
            return None
        return self._expert_probs.argmax(dim=-1)
