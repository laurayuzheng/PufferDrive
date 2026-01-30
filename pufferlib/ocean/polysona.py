import torch
import torch.nn as nn
import torch.nn.functional as F
import pufferlib

class LoRALayer(nn.Module):
    """Low-Rank Adapter for a linear transformation."""
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        
        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # Returns the *delta* to be added to original output
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class MixtureOfLoRAExperts(nn.Module):
    """Mixture of LoRA experts with soft routing (CAT-style mixing)."""
    def __init__(self, in_dim, out_dim, num_experts=3, rank=8, alpha=16):
        super().__init__()
        self.num_experts = num_experts
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Create expert LoRAs
        self.experts = nn.ModuleList([
            LoRALayer(in_dim, out_dim, rank, alpha) 
            for _ in range(num_experts)
        ])
    
    def forward(self, x, routing_weights):
        """
        Args:
            x: Input tensor [batch, in_dim]
            routing_weights: Soft routing [batch, num_experts] (should sum to 1)
        Returns:
            Delta to add to base output [batch, out_dim]
        """
        batch_size = x.shape[0]
        
        # Compute each expert's contribution
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # expert_outputs: [batch, num_experts, out_dim]
        
        # Weighted combination (CAT-style)
        routing_weights = routing_weights.unsqueeze(-1)  # [batch, num_experts, 1]
        mixed_output = (expert_outputs * routing_weights).sum(dim=1)  # [batch, out_dim]
        
        return mixed_output


class SocialForcesExtractor(nn.Module):
    """Extract social forces features from partner observations."""
    def __init__(self, max_partners, partner_features, hidden_dim=64):
        super().__init__()
        self.max_partners = max_partners
        self.partner_features = partner_features
        
        # Social forces are computed from relative positions/velocities
        # Assuming partner features include: [rel_x, rel_y, rel_vx, rel_vy, ...]
        self.force_proj = nn.Sequential(
            nn.Linear(partner_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, partner_obs, ego_obs):
        """
        Compute social forces features.
        
        Args:
            partner_obs: [batch, max_partners, partner_features]
            ego_obs: [batch, ego_dim]
        
        Returns:
            social_forces: [batch, hidden_dim]
        """
        batch_size = partner_obs.shape[0]
        
        # Mask out invalid partners (assuming zeros indicate invalid)
        valid_mask = (partner_obs.abs().sum(dim=-1) > 1e-6).float()  # [batch, max_partners]
        
        # Project partner features
        partner_features = self.force_proj(partner_obs)  # [batch, max_partners, hidden_dim]
        
        # Compute repulsion-like features (simplified social forces)
        # In full implementation: use exponential decay based on distance
        rel_pos = partner_obs[:, :, :2]  # Assuming first 2 features are relative position
        distances = torch.norm(rel_pos, dim=-1, keepdim=True).clamp(min=0.1)  # [batch, max_partners, 1]
        
        # Exponential decay weighting (closer = stronger force)
        force_weights = torch.exp(-distances / 5.0) * valid_mask.unsqueeze(-1)
        
        # Weighted aggregation of partner features
        weighted_features = partner_features * force_weights
        social_forces = weighted_features.sum(dim=1)  # [batch, hidden_dim]
        
        return social_forces


class StyleRouter(nn.Module):
    """Routes to experts based on social forces and context (VAE-style)."""
    def __init__(self, context_dim, social_dim, num_experts=3, hidden_dim=64):
        super().__init__()
        self.num_experts = num_experts
        
        # Encoder (posterior estimator)
        self.encoder = nn.Sequential(
            nn.Linear(context_dim + social_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),  # Logits for Gumbel-Softmax
        )
        
        # Reconstructor (for VAE-like training)
        self.reconstructor = nn.Sequential(
            nn.Linear(num_experts + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, social_dim),
        )
        
    def forward(self, context_features, social_forces, temperature=1.0, hard=False):
        """
        Args:
            context_features: Hidden state from encoder [batch, context_dim]
            social_forces: Social forces features [batch, social_dim]
            temperature: Gumbel-Softmax temperature
            hard: Whether to use hard (one-hot) or soft routing
        
        Returns:
            routing_weights: [batch, num_experts]
            routing_logits: [batch, num_experts] (for loss computation)
            reconstructed_sf: [batch, social_dim]
        """
        # Concatenate context and social forces
        combined = torch.cat([context_features, social_forces], dim=-1)
        
        # Get routing logits
        routing_logits = self.encoder(combined)
        
        # Gumbel-Softmax for differentiable discrete sampling
        routing_weights = F.gumbel_softmax(routing_logits, tau=temperature, hard=hard)
        
        # Reconstruct social forces (VAE decoder)
        recon_input = torch.cat([routing_weights, context_features], dim=-1)
        reconstructed_sf = self.reconstructor(recon_input)
        
        return routing_weights, routing_logits, reconstructed_sf


class Polysona(nn.Module):
    """PufferDrive policy with Mixture-of-LoRA style experts."""
    
    def __init__(self, env, input_size=128, hidden_size=128, 
                 num_experts=3, lora_rank=8, lora_alpha=16,
                 social_force_dim=64, freeze_base=True, **kwargs):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.freeze_base = freeze_base
        
        # === Base Model Components (from original Drive) ===
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6
        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7

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

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, hidden_size)),
        )
        
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        # Base actor (will be frozen, MoL adds to this)
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01
        )
        self.value_fn = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1
        )
        
        # === Style Components (trainable in post-training) ===
        self.social_forces_extractor = SocialForcesExtractor(
            self.max_partner_objects, 
            self.partner_features,
            hidden_dim=social_force_dim
        )
        
        self.style_router = StyleRouter(
            context_dim=hidden_size,
            social_dim=social_force_dim,
            num_experts=num_experts,
        )
        
        # MoL layer: modifies hidden state before actor
        self.mol_layer = MixtureOfLoRAExperts(
            in_dim=hidden_size,
            out_dim=hidden_size,
            num_experts=num_experts,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        
        # Style discriminator (for DIAYN-style training)
        self.style_discriminator = nn.Sequential(
            nn.Linear(hidden_size + social_force_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
        )
        
        # Temperature for Gumbel-Softmax (can anneal during training)
        self.register_buffer('temperature', torch.tensor(1.0))
        
    def load_base_model(self, checkpoint_path):
        """Load pre-trained base model and optionally freeze."""
        state_dict = torch.load(checkpoint_path)
        # Load only base model components
        base_keys = ['ego_encoder', 'road_encoder', 'partner_encoder', 
                     'shared_embedding', 'actor', 'value_fn']
        filtered_state = {k: v for k, v in state_dict.items() 
                         if any(k.startswith(bk) for bk in base_keys)}
        self.load_state_dict(filtered_state, strict=False)
        
        if self.freeze_base:
            self._freeze_base_model()
    
    def _freeze_base_model(self):
        """Freeze base model parameters."""
        for name, param in self.named_parameters():
            if not any(x in name for x in ['social_forces', 'style_router', 
                                            'mol_layer', 'style_discriminator']):
                param.requires_grad = False
                
    def _parse_observations(self, observations):
        """Parse flat observation into components."""
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim:ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim:ego_dim + partner_dim + road_dim]
        
        partner_obs = partner_obs.view(-1, self.max_partner_objects, self.partner_features)
        road_obs = road_obs.view(-1, self.max_road_objects, self.road_features)
        
        return ego_obs, partner_obs, road_obs

    def encode_observations(self, observations, state=None):
        """Encode observations (same as base model)."""
        ego_obs, partner_obs, road_obs = self._parse_observations(observations)
        
        # One-hot encode road categories
        road_continuous = road_obs[:, :, :self.road_features - 1]
        road_categorical = road_obs[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_obs = torch.cat([road_continuous, road_onehot], dim=2)
        
        # Encode each modality
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_obs).max(dim=1)
        road_features, _ = self.road_encoder(road_obs).max(dim=1)
        
        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)
        embedding = F.relu(self.shared_embedding(concat_features))
        
        return embedding

    def forward(self, observations, state=None, style_override=None):
        """
        Forward pass with style-conditioned actions.
        
        Args:
            observations: Environment observations
            state: LSTM state (if using recurrence)
            style_override: If provided, use this style index instead of routing
        """
        # Parse observations for social forces
        ego_obs, partner_obs, road_obs = self._parse_observations(observations)
        
        # Get base encoding
        hidden = self.encode_observations(observations, state)
        
        # Compute social forces
        social_forces = self.social_forces_extractor(partner_obs, ego_obs)
        
        # Route to experts
        if style_override is not None:
            # Use specified style (for counterfactual simulation)
            routing_weights = F.one_hot(
                style_override, num_classes=self.num_experts
            ).float()
            routing_logits = None
            reconstructed_sf = None
        else:
            routing_weights, routing_logits, reconstructed_sf = self.style_router(
                hidden, social_forces, 
                temperature=self.temperature.item(),
                hard=False
            )
        
        # Apply MoL transformation
        style_delta = self.mol_layer(hidden, routing_weights)
        styled_hidden = hidden + style_delta  # Residual connection
        
        # Decode actions
        actions, value = self.decode_actions(styled_hidden)
        
        # Store auxiliary outputs for training
        self._last_routing_weights = routing_weights
        self._last_routing_logits = routing_logits
        self._last_reconstructed_sf = reconstructed_sf
        self._last_social_forces = social_forces
        self._last_styled_hidden = styled_hidden
        
        return actions, value
    
    def decode_actions(self, flat_hidden):
        """Decode actions from hidden state."""
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = F.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)
        
        value = self.value_fn(flat_hidden)
        return action, value
    
    def compute_style_losses(self):
        """Compute auxiliary losses for style training."""
        losses = {}
        
        # 1. Reconstruction loss (VAE-like)
        if self._last_reconstructed_sf is not None:
            losses['recon_loss'] = F.mse_loss(
                self._last_reconstructed_sf, 
                self._last_social_forces.detach()
            )
        
        # 2. Discriminator loss (DIAYN-style)
        # Train discriminator to predict style from trajectory features
        disc_input = torch.cat([
            self._last_styled_hidden.detach(), 
            self._last_social_forces.detach()
        ], dim=-1)
        disc_logits = self.style_discriminator(disc_input)
        style_labels = self._last_routing_weights.argmax(dim=-1)
        losses['disc_loss'] = F.cross_entropy(disc_logits, style_labels)
        
        # 3. Intrinsic reward for policy (maximize discriminability)
        # This should be added to the environment reward during training
        with torch.no_grad():
            disc_probs = F.softmax(disc_logits, dim=-1)
            # DIAYN reward: log q(z|s) - log p(z)
            prior = torch.ones(self.num_experts, device=disc_probs.device) / self.num_experts
            intrinsic_reward = (
                torch.log(disc_probs.gather(1, style_labels.unsqueeze(1)) + 1e-8) 
                - torch.log(prior[style_labels].unsqueeze(1))
            ).squeeze()
            losses['intrinsic_reward'] = intrinsic_reward
        
        # 4. Entropy regularization (encourage exploration across styles)
        if self._last_routing_logits is not None:
            router_probs = F.softmax(self._last_routing_logits, dim=-1)
            router_entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean()
            losses['router_entropy'] = router_entropy
        
        return losses