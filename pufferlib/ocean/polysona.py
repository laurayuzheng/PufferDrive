from torch import nn
import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401
from pufferlib.ocean.polysona_utils import polytropon
from pufferlib.ocean.polysona_utils import adapter_utils

Recurrent = pufferlib.models.LSTMWrapper


class DrivePolysona(nn.Module):
    def __init__(self, 
                 env, 
                 num_personas=3, 
                 prior=None, 
                 input_size=128, 
                 hidden_size=128, 
                 **kwargs):
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

        self.actor = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01),
        )
        # pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

        # Polysona related attributes
        self.num_personas = num_personas
        self.persona_prior = (
            torch.ones((num_personas,), dtype=torch.float) / self.num_personas if prior is None else prior
        )

        self.skilled_variant = 'private'  # standard mixture of LoRA
        self.mixin_class = polytropon.VARIANT2CLASS[self.skilled_variant][0]
        self.persona_classifier = None

        # Initialize router
        self.persona_classifier = (
            nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),  # in channels and pooled social forces
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                pufferlib.pytorch.layer_init(nn.Linear(hidden_size, self.num_personas))
            )
        )

        # Applies LoRA wrapping to ego, partner, and actor module.
        self.wrap_model_with_mixins()


    def _wrap_mixin(self, model: nn.Module, attention_only=True):
        """Generic helper to wrap a module with the manual skilled mixin class."""
        return (
            polytropon.SkilledMixin(
                model=model,
                n_tasks=self.num_personas,  # conservative, moderate, aggressive
                n_skills=self.num_personas,
                skilled_variant=self.skilled_variant,
                freeze=True,
                state_dict=None,
                rank=self.lora_rank,
                attention_only=attention_only,
            )
            if model is not None
            else None
        )

    def wrap_model_with_mixins(self):
        """Only wrap the actor for now; we use encoder to predict the 'style'."""
        # self.ego_encoder = self._wrap_mixin(self.ego_encoder)
        # self.partner_encoder = self._wrap_mixin(self.partner_encoder)
        # self.shared_embedding = self._wrap_mixin(self.shared_embedding)
        self.actor = self._wrap_mixin(self.actor)
    
    def broadcast_expert_indices(self, z):
        """Informs all Mixin layers of current expert indices."""

        if z is not None:
            # adapter_utils.inform_layers(self.ego_encoder, adapter_class=self.mixin_class, value=z)
            # adapter_utils.inform_layers(self.partner_encoder, adapter_class=self.mixin_class, value=z)
            # adapter_utils.inform_layers(self.shared_embedding, adapter_class=self.mixin_class, value=z)
            adapter_utils.inform_layers(self.actor, adapter_class=self.mixin_class, value=z)


    def forward(self, observations, state=None, z=None):
        # sampled prior z should be of shape (batch, num_personas)
        self.broadcast_expert_indices(z)

        hidden = self.encode_observations(observations)
        latent_prediction = self.compute_q_z_given_s(hidden)
        actions, value = self.decode_actions(hidden)
        return actions, value, latent_prediction

    def forward_train(self, x, state=None, z=None):
        return self.forward(x, state, z)
    
    def forward_eval(self, x, state=None, z=None):

        if z is not None:
            return self.forward(x, state, z)
        else:
            # Use inferred latent, if not provided
            hidden = self.encode_observations(x)
            latent_prediction = self.compute_q_z_given_s(hidden)

            self.broadcast_expert_indices(latent_prediction)
            actions, value = self.decode_actions(hidden)
            return actions, value, latent_prediction

    
    def compute_q_z_given_s(self, s):
        return self.persona_classifier(s)
    
    def compute_skill_reward(self, predicted_z_logits, sampled_z):
        """Compute DIAYN intrinsic reward: r = log q_φ(z | s) - log p(z)
        
        Get log q_φ(z | s) for specific skill indices.
        
        Args:
            predicted_z_logits: [batch, num_personas]
            sampled_z: [batch] - integer skill indices
        Returns:
            log_prob: [batch]
        """

        p_z = self.persona_prior.gather(-1, sampled_z)  # zs is [batch, 1] skill indices
        log_q_z_s = F.log_softmax(predicted_z_logits, dim=-1)

        return log_q_z_s.gather(-1, sampled_z).detach() - torch.log(p_z + 1e-6)


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

    def decode_actions(self, flat_hidden, z=None):
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
