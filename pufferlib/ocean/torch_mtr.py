"""
MTR-based policies for PufferDrive closed-loop RL.

This module provides:
- DriveMTR: Baseline MTR policy with GMM action head
- DriveMTRMoE: MTR policy with frozen backbone + LoRA experts

Uses the full MTR (Motion Transformer) architecture from mtr_encoder.py,
with MPC-style GMM action output for closed-loop control.

Key GMM Action Head Format (7 values per timestep):
- [0] accel: acceleration (direct)
- [1] steer_raw: steering before tanh transform
- [2] log_σ_accel: log std of acceleration
- [3] log_σ_steer: log std of steering
- [4] ρ: correlation coefficient
- [5] vx: predicted velocity x
- [6] vy: predicted velocity y

Transforms:
- mu_accel = pred[:, :, :, 0]  # direct
- mu_steer = tanh(pred[:, :, :, 1]) * π/3  # scaled to [-60°, +60°]
"""

from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pufferlib
import pufferlib.pytorch

from pufferlib.ocean.mtr_encoder import (
    MTREncoder,
    MTRDecoder,
    PersonaRouterWithReconstruction,
    EPS,
)
from pufferlib.ocean.moe_adapters import (
    LoRAExpertsRL,
    kl_divergence_loss,
    entropy_loss,
    expert_cosine_loss,
    expert_output_diversity_loss,
    compute_temperature,
)
from pufferlib.ocean.inverse_dynamics import (
    ACCELERATION_VALUES,
    STEERING_VALUES,
)


Recurrent = pufferlib.models.LSTMWrapper


class DriveMTR(nn.Module):
    """
    Baseline MTR policy for PufferDrive with GMM action head.

    Uses the full MTR architecture:
    - PointNet polyline encoders for agents and map
    - Transformer self-attention encoder
    - Transformer decoder with cross-attention to agents and maps
    - Query-based decoding with learnable intention queries
    - GMM action output with kinematic integration

    Training:
    1. RL loss on discretized first-timestep actions
    2. Trajectory loss on full predicted trajectory vs ground truth
    3. Optional imitation loss on expert actions

    Outputs discrete actions (7 accel × 13 steer = 91 total).
    """

    def __init__(
        self,
        env,
        d_model: int = 256,
        hidden_size: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_queries: int = 6,
        num_future_frames: int = 80,
        dropout: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            env: PufferDrive environment
            d_model: Transformer hidden dimension
            hidden_size: Output hidden dimension (for LSTM)
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder self-attention layers
            num_decoder_layers: Number of decoder cross-attention layers
            num_queries: Number of intention/action queries
            num_future_frames: Number of future frames to predict
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_future_frames = num_future_frames
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6

        # Determine ego dimension from environment's dynamics model
        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7

        # MTR Encoder
        self.encoder = MTREncoder(
            num_input_attr_agent=self.partner_features,
            num_input_attr_map=self.road_features_after_onehot,
            d_model=d_model,
            num_attn_head=nhead,
            num_attn_layers=num_encoder_layers,
            hidden_dim_agent=128,
            hidden_dim_map=64,
            dropout=dropout,
        )

        # MTR Decoder
        self.decoder = MTRDecoder(
            in_channels=d_model,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            num_queries=num_queries,
            num_future_frames=num_future_frames,
            dropout=dropout,
        )

        # Projection to hidden_size (for LSTM input)
        self.output_proj = nn.Linear(d_model, hidden_size)

        # Action space setup
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        # Value function
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1
        )

        # Register action discretization values as buffers
        self.register_buffer(
            "accel_values",
            torch.tensor(ACCELERATION_VALUES, dtype=torch.float32),
        )
        self.register_buffer(
            "steer_values",
            torch.tensor(STEERING_VALUES, dtype=torch.float32),
        )

        # Storage for trajectory loss computation
        self._pred_trajs = None
        self._pred_scores = None

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_eval(self, observations, state=None):
        """Forward pass for evaluation (no state needed for non-RNN policy)."""
        return self.forward(observations, state)

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        """Encode observations using MTR architecture."""
        batch_size = observations.shape[0]
        device = observations.device

        # Parse observations
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features

        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        # Reshape
        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)
        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)

        # One-hot encode road type
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        # Prepare MTR inputs
        # Agent trajectories: ego + partners
        ego_as_agent = torch.zeros(batch_size, 1, 1, self.partner_features, device=device)
        ego_as_agent[:, 0, 0, :min(ego_dim, self.partner_features)] = ego_obs[:, :min(ego_dim, self.partner_features)]

        partner_trajs = partner_objects.unsqueeze(2)  # (batch, max_partners, 1, features)
        obj_trajs = torch.cat([ego_as_agent, partner_trajs], dim=1)

        # Masks
        ego_mask = torch.ones(batch_size, 1, 1, dtype=torch.bool, device=device)
        partner_mag = partner_objects.abs().sum(dim=-1)
        partner_mask = (partner_mag > EPS).unsqueeze(2)
        obj_trajs_mask = torch.cat([ego_mask, partner_mask], dim=1)

        # Positions
        ego_pos = torch.zeros(batch_size, 1, 2, device=device)
        partner_pos = partner_objects[:, :, :2] * 50.0  # Undo scaling
        obj_trajs_last_pos = torch.cat([ego_pos, partner_pos], dim=1)

        # Map data
        road_mag = road_objects.abs().sum(dim=-1)
        map_polylines = road_objects.unsqueeze(2)
        map_polylines_mask = (road_mag > EPS).unsqueeze(2)
        map_polylines_center = road_objects[:, :, :2] * 50.0

        # Track index (ego = 0)
        track_index_to_predict = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Encode
        enc_dict = self.encoder(
            obj_trajs=obj_trajs,
            obj_trajs_mask=obj_trajs_mask,
            obj_trajs_last_pos=obj_trajs_last_pos,
            map_polylines=map_polylines,
            map_polylines_mask=map_polylines_mask,
            map_polylines_center=map_polylines_center,
            track_index_to_predict=track_index_to_predict,
        )

        # Decode (returns dict with decoded_feature, pred_trajs, pred_scores)
        dec_dict = self.decoder(enc_dict)

        # Store for trajectory loss computation
        self._pred_trajs = dec_dict["pred_trajs"]
        self._pred_scores = dec_dict["pred_scores"]

        # Project to hidden_size
        hidden = F.relu(self.output_proj(dec_dict["decoded_feature"]))

        return hidden

    def decode_actions(self, flat_hidden):
        """
        Decode hidden state to actions and value.

        Extracts first-timestep actions from GMM predictions and discretizes
        to the action space.
        """
        pred_trajs = self._pred_trajs  # (batch, K, T, 7)
        pred_scores = self._pred_scores  # (batch, K)
        batch_size = flat_hidden.shape[0]

        if pred_trajs is None or pred_scores is None:
            # Fallback: return zeros if no predictions (shouldn't happen in normal use)
            if self.is_continuous:
                loc = torch.zeros(batch_size, self.atn_dim[0], device=flat_hidden.device)
                std = torch.ones(batch_size, self.atn_dim[1], device=flat_hidden.device) * 0.1
                action = torch.distributions.Normal(loc, std)
            else:
                # Discrete: return flat logits matching atn_dim (e.g., [91])
                action = tuple(
                    torch.zeros(batch_size, dim, device=flat_hidden.device)
                    for dim in self.atn_dim
                )
            value = self.value_fn(flat_hidden)
            return action, value

        # Select best query based on confidence scores
        if self.training:
            # Soft selection during training: weighted combination
            weights = F.softmax(pred_scores, dim=-1)  # (batch, K)
            # First timestep raw actions: (batch, K, 2) -> [accel, steer_raw]
            first_step_raw = pred_trajs[:, :, 0, :2]
            # Weighted combination
            raw_actions = torch.einsum('bk,bka->ba', weights, first_step_raw)  # (batch, 2)
        else:
            # Hard selection during inference
            best_idx = pred_scores.argmax(dim=-1)  # (batch,)
            raw_actions = pred_trajs[torch.arange(batch_size, device=pred_trajs.device), best_idx, 0, :2]

        # Apply transforms (from mtr_actions.py:425-428)
        accel = raw_actions[:, 0]  # Direct
        steer = torch.tanh(raw_actions[:, 1]) * (torch.pi / 3)  # Scale to [-π/3, π/3]

        # Convert steering angle to discrete action space
        # STEERING_VALUES are in [-1, 1], so map [-π/3, π/3] → [-1, 1]
        steer_normalized = steer / (torch.pi / 3)  # Now in [-1, 1]

        if self.is_continuous:
            # Continuous action space
            loc = torch.stack([accel, steer_normalized], dim=-1)
            std = torch.ones_like(loc) * 0.1
            action = torch.distributions.Normal(loc, std)
        else:
            # Discrete action space: create logits for flattened 91-action space
            # action_idx = accel_idx * 13 + steer_idx
            NUM_ACCEL = len(self.accel_values)  # 7
            NUM_STEER = len(self.steer_values)  # 13
            NUM_ACTIONS = NUM_ACCEL * NUM_STEER  # 91

            # Compute negative L1 distance as logits
            accel_logits = -torch.abs(accel.unsqueeze(-1) - self.accel_values)  # (batch, 7)
            steer_logits = -torch.abs(steer_normalized.unsqueeze(-1) - self.steer_values)  # (batch, 13)

            # Temperature scaling to make distributions sharper
            temperature = 0.5
            accel_logits = accel_logits / temperature
            steer_logits = steer_logits / temperature

            # Combine into flattened action logits: log(P(accel) * P(steer)) = log P(accel) + log P(steer)
            # Shape: (batch, 7, 1) + (batch, 1, 13) -> (batch, 7, 13) -> (batch, 91)
            combined_logits = accel_logits.unsqueeze(-1) + steer_logits.unsqueeze(-2)
            flat_logits = combined_logits.view(batch_size, NUM_ACTIONS)  # (batch, 91)

            # Return as tuple matching atn_dim format
            action = torch.split(flat_logits, self.atn_dim, dim=1)

        value = self.value_fn(flat_hidden)

        return action, value

    def get_trajectory_loss(
        self,
        gt_future_traj: torch.Tensor,
        gt_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GMM NLL loss on predicted trajectory vs ground truth.

        Args:
            gt_future_traj: (batch, T, 2) ground truth positions
            gt_valid_mask: (batch, T) validity mask

        Returns:
            Trajectory loss scalar
        """
        if self._pred_trajs is None or self._pred_scores is None:
            return torch.tensor(0.0, device=gt_future_traj.device)

        pred_trajs = self._pred_trajs  # (batch, K, T, 7)
        pred_scores = self._pred_scores  # (batch, K)
        batch_size, K, T, _ = pred_trajs.shape

        # Truncate to match lengths
        T_gt = gt_future_traj.shape[1]
        T_min = min(T, T_gt)
        pred_trajs = pred_trajs[:, :, :T_min, :]
        gt_future_traj = gt_future_traj[:, :T_min, :]
        gt_valid_mask = gt_valid_mask[:, :T_min]

        # Extract predicted positions (first 2 values after kinematic integration)
        # Note: pred_trajs has raw GMM output, not integrated positions
        # For simplicity, use the velocity components to approximate position delta
        pred_pos = pred_trajs[:, :, :, :2]  # Raw output (not integrated)

        # Compute L2 distance between predictions and GT
        # gt_future_traj: (batch, T, 2) -> (batch, 1, T, 2)
        gt_expanded = gt_future_traj.unsqueeze(1)

        # Distance: (batch, K, T)
        dist = torch.norm(pred_pos - gt_expanded, dim=-1)

        # Apply validity mask
        gt_valid_expanded = gt_valid_mask.unsqueeze(1)  # (batch, 1, T)
        dist_masked = dist * gt_valid_expanded

        # Sum over timesteps, average over batch
        num_valid = gt_valid_mask.sum(dim=-1).clamp(min=1)  # (batch,)
        ade_per_query = dist_masked.sum(dim=-1) / num_valid.unsqueeze(1)  # (batch, K)

        # Winner-takes-all: use best query
        min_ade = ade_per_query.min(dim=-1).values  # (batch,)

        return min_ade.mean()


class DriveMTRMoE(nn.Module):
    """
    MTR policy with Mixture-of-Experts for PufferDrive.

    Uses the full MTR architecture with:
    - Frozen MTR Encoder (when freeze_base=True)
    - MTR Decoder with optional LoRA adapters
    - Persona Router: Expert routing with social forces + reconstruction loss
    - LoRA Actor: Parameter-efficient expert adapters

    Training:
    1. RL loss on discretized actions
    2. Router reconstruction loss
    3. Expert diversity losses (KL, cosine, entropy)
    """

    def __init__(
        self,
        env,
        d_model: int = 256,
        hidden_size: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_queries: int = 6,
        num_future_frames: int = 80,
        num_experts: int = 3,
        lora_rank: int = 8,
        lora_alpha: float = 4.0,
        router_hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_base: bool = False,
        base_checkpoint: str = None,
        expert_prior=None,
        **kwargs,
    ):
        """
        Args:
            env: PufferDrive environment
            d_model: Transformer hidden dimension
            hidden_size: Output hidden dimension (for LSTM)
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder self-attention layers
            num_decoder_layers: Number of decoder cross-attention layers
            num_queries: Number of intention/action queries
            num_future_frames: Number of future frames to predict
            num_experts: Number of expert modules (K)
            lora_rank: LoRA rank for expert adapters
            lora_alpha: LoRA scaling factor
            router_hidden_dim: Hidden dimension for router
            dropout: Dropout probability
            freeze_base: Whether to freeze MTR encoder (train LoRA only)
            base_checkpoint: Path to pretrained baseline checkpoint
            expert_prior: Prior distribution for KL loss
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_queries = num_queries
        self.num_future_frames = num_future_frames
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
        self._expert_output_diversity_loss = expert_output_diversity_loss
        self._compute_temperature = compute_temperature

        # Expert prior for KL loss
        if expert_prior is not None:
            self.register_buffer("expert_prior", torch.tensor(expert_prior, dtype=torch.float32))
        else:
            self.expert_prior = None

        # Determine ego dimension from environment's dynamics model
        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7

        # MTR Encoder
        self.encoder = MTREncoder(
            num_input_attr_agent=self.partner_features,
            num_input_attr_map=self.road_features_after_onehot,
            d_model=d_model,
            num_attn_head=nhead,
            num_attn_layers=num_encoder_layers,
            hidden_dim_agent=128,
            hidden_dim_map=64,
            dropout=dropout,
        )

        # MTR Decoder
        self.decoder = MTRDecoder(
            in_channels=d_model,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            num_queries=num_queries,
            num_future_frames=num_future_frames,
            dropout=dropout,
        )

        # Load pretrained weights if provided
        if base_checkpoint is not None:
            self._load_base_checkpoint(base_checkpoint)

        # Projection to hidden_size (for LSTM input)
        self.output_proj = nn.Linear(d_model, hidden_size)

        # Persona Router with reconstruction
        self.router = PersonaRouterWithReconstruction(
            context_dim=d_model,
            social_force_dim=6,
            num_experts=num_experts,
            hidden_dim=router_hidden_dim,
            dropout=dropout,
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

        # Value function (shared across all styles)
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1
        )

        # Register action discretization values as buffers
        self.register_buffer(
            "accel_values",
            torch.tensor(ACCELERATION_VALUES, dtype=torch.float32),
        )
        self.register_buffer(
            "steer_values",
            torch.tensor(STEERING_VALUES, dtype=torch.float32),
        )

        # Optionally freeze MTR encoder
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Router is NOT frozen - it learns to route

        # Storage for auxiliary losses
        self._aux_losses = {}
        self._expert_probs = None
        self._router_logits = None
        self._social_forces = None
        self._reconstructed_sf = None
        self._pred_trajs = None
        self._pred_scores = None

        # Expert usage tracking
        self._expert_usage_counts = None
        self._cached_embedding = None

    def _load_base_checkpoint(self, checkpoint_path: str):
        """Load pretrained encoder/decoder weights from a DriveMTR checkpoint."""
        import os
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Base checkpoint not found at {checkpoint_path}, starting from scratch")
            return

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Filter for encoder and decoder weights
        encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
        decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}

        if encoder_state:
            self.encoder.load_state_dict(encoder_state, strict=False)
            print(f"Loaded encoder weights from {checkpoint_path}")
        if decoder_state:
            self.decoder.load_state_dict(decoder_state, strict=False)
            print(f"Loaded decoder weights from {checkpoint_path}")

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_eval(self, observations, state=None):
        """Forward pass for evaluation (no state needed for non-RNN policy)."""
        return self.forward(observations, state)

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        """Encode observations using MTR architecture with routing."""
        batch_size = observations.shape[0]
        device = observations.device

        # Parse observations
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features

        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        # Reshape
        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)
        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)

        # One-hot encode road type
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        # Prepare MTR inputs
        ego_as_agent = torch.zeros(batch_size, 1, 1, self.partner_features, device=device)
        ego_as_agent[:, 0, 0, :min(ego_dim, self.partner_features)] = ego_obs[:, :min(ego_dim, self.partner_features)]

        partner_trajs = partner_objects.unsqueeze(2)
        obj_trajs = torch.cat([ego_as_agent, partner_trajs], dim=1)

        ego_mask = torch.ones(batch_size, 1, 1, dtype=torch.bool, device=device)
        partner_mag = partner_objects.abs().sum(dim=-1)
        partner_mask = (partner_mag > EPS).unsqueeze(2)
        obj_trajs_mask = torch.cat([ego_mask, partner_mask], dim=1)

        ego_pos = torch.zeros(batch_size, 1, 2, device=device)
        partner_pos = partner_objects[:, :, :2] * 50.0
        obj_trajs_last_pos = torch.cat([ego_pos, partner_pos], dim=1)

        road_mag = road_objects.abs().sum(dim=-1)
        map_polylines = road_objects.unsqueeze(2)
        map_polylines_mask = (road_mag > EPS).unsqueeze(2)
        map_polylines_center = road_objects[:, :, :2] * 50.0

        track_index_to_predict = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Encode
        enc_dict = self.encoder(
            obj_trajs=obj_trajs,
            obj_trajs_mask=obj_trajs_mask,
            obj_trajs_last_pos=obj_trajs_last_pos,
            map_polylines=map_polylines,
            map_polylines_mask=map_polylines_mask,
            map_polylines_center=map_polylines_center,
            track_index_to_predict=track_index_to_predict,
        )

        # Decode
        dec_dict = self.decoder(enc_dict)
        decoded_feature = dec_dict["decoded_feature"]

        # Store predictions
        self._pred_trajs = dec_dict["pred_trajs"]
        self._pred_scores = dec_dict["pred_scores"]

        # Route using social forces
        expert_probs, router_logits, social_forces, reconstructed_sf = self.router(
            decoded_feature, partner_objects, hard=not self.training
        )

        # Store routing information
        self._expert_probs = expert_probs
        self._router_logits = router_logits
        self._social_forces = social_forces
        self._reconstructed_sf = reconstructed_sf

        # Track expert usage
        with torch.no_grad():
            expert_assignments = expert_probs.argmax(dim=-1)
            self._expert_usage_counts = torch.bincount(
                expert_assignments, minlength=self.num_experts
            ).float()

        # Project to hidden_size
        hidden = F.relu(self.output_proj(decoded_feature))

        # Compute auxiliary losses during training
        if self.training:
            self._aux_losses = {
                "kl": self._kl_divergence_loss(expert_probs, prior=self.expert_prior),
                "entropy": self._entropy_loss(expert_probs),
                "cosine": self._expert_cosine_loss(
                    self.actor.expert_A, self.actor.expert_B
                ),
                "reconstruction": F.mse_loss(
                    reconstructed_sf, social_forces.detach()
                ),
                "output_div": None,  # Computed lazily if coefficient > 0
            }
            self._cached_embedding = hidden
        else:
            self._cached_embedding = None

        return hidden

    def decode_actions(self, flat_hidden, expert_probs=None):
        """Decode hidden state to actions and value using LoRA experts."""
        if expert_probs is None:
            expert_probs = self._expert_probs

        batch_size = flat_hidden.shape[0]

        if self.is_continuous:
            parameters = self.actor(flat_hidden, expert_probs)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = F.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action_logits = self.actor(flat_hidden, expert_probs)
            action = torch.split(action_logits, self.atn_dim, dim=1)

        value = self.value_fn(flat_hidden)

        return action, value

    def get_auxiliary_losses(self, config=None) -> Dict[str, torch.Tensor]:
        """
        Return auxiliary losses for training integration.

        Includes:
        - kl: KL divergence loss for uniform expert usage
        - entropy: Negative entropy for confident routing
        - cosine: Expert weight orthogonality
        - reconstruction: Social forces reconstruction
        - output_div: Output diversity (computed lazily if coefficient > 0)
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

    def get_expert_stats(self) -> Dict[str, float]:
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

    def update_temperature(self, progress: float, config: Dict):
        """Update Gumbel-Softmax temperature based on training progress."""
        tau_max = config.get("moe_tau_max", 2.0)
        tau_min = config.get("moe_tau_min", 0.1)
        new_temp = self._compute_temperature(progress, tau_max, tau_min)
        self.router.set_temperature(new_temp)

    def forward_with_forced_expert(self, observations, expert_idx: int, state=None):
        """
        Forward pass with a specific expert forced (one-hot).

        Useful for multiverse visualization to compare expert behaviors.
        """
        batch_size = observations.shape[0]

        # Create one-hot expert weights
        forced_probs = torch.zeros(batch_size, self.num_experts, device=observations.device)
        forced_probs[:, expert_idx] = 1.0

        # Encode observations (this sets routing but we'll override)
        hidden = self.encode_observations(observations, state)

        # Decode with forced expert weights
        actions, value = self.decode_actions(hidden, expert_probs=forced_probs)

        return actions, value
