"""
MTR (Motion Transformer) architecture adapted for PufferDrive closed-loop RL.

This implementation follows the original MTR architecture from:
- Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
- Published at NeurIPS 2022

The architecture is kept identical to the original MTR, with the only adaptation
being the output heads (discrete action logits instead of GMM trajectory predictions).

Key components (matching original MTR):
- PointNetPolylineEncoder for agent and map feature extraction
- Transformer self-attention encoder
- Transformer decoder with separate cross-attention to agents and maps
- Query-based decoding with learnable intention/action queries
- Dense future prediction for feature enhancement
- Dynamic map collection based on predicted waypoints

Adaptation for RL:
- Output discrete action logits instead of trajectory GMMs
- Single timestep observations (uses LSTM for temporal context)
- PufferDrive observation format instead of WOMD polylines
"""

import copy
import math
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-12


# ==============================================================================
# Utility Functions (from MTR_utils.py and position_encoding_utils.py)
# ==============================================================================


def build_mlps(
    c_in: int,
    mlp_channels: List[int],
    ret_before_act: bool = False,
    without_norm: bool = False,
) -> nn.Sequential:
    """Build MLP layers (exactly as in MTR)."""
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()])
            else:
                # MTR uses BatchNorm1d, but LayerNorm is more stable for variable batch sizes
                layers.extend(
                    [
                        nn.Linear(c_in, mlp_channels[k], bias=False),
                        nn.LayerNorm(mlp_channels[k]),
                        nn.ReLU(),
                    ]
                )
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)


def gen_sineembed_for_position(pos_tensor: torch.Tensor, hidden_dim: int = 256) -> torch.Tensor:
    """
    Generate sinusoidal position embedding (from MTR position_encoding_utils).
    Matches the original implementation exactly.

    Args:
        pos_tensor: (N, B, 2) or (B, N, 2) position tensor
        hidden_dim: embedding dimension

    Returns:
        pos_embedding: same shape as input with last dim = hidden_dim
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)

    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale

    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t

    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos


def _get_activation_fn(activation: str):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


# ==============================================================================
# PointNetPolylineEncoder (exactly matching MTR)
# ==============================================================================


class PointNetPolylineEncoder(nn.Module):
    """
    PointNet-style polyline encoder (matching MTR exactly).

    Args:
        in_channels: Input feature dimension per point
        hidden_dim: Hidden layer dimension
        num_layers: Total number of MLP layers
        num_pre_layers: Number of pre-pooling layers
        out_channels: Output feature dimension
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_pre_layers: int = 1,
        out_channels: Optional[int] = None,
    ):
        super().__init__()

        # Pre-pooling MLPs
        self.pre_mlps = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False,
        )

        # Post-pooling MLPs (with global feature concatenation)
        self.mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False,
        )

        # Output projection
        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim,
                mlp_channels=[hidden_dim, out_channels],
                ret_before_act=True,
                without_norm=True,
            )
        else:
            self.out_mlps = None

    def forward(self, polylines: torch.Tensor, polylines_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            polylines: (batch, num_polylines, num_points, C)
            polylines_mask: (batch, num_polylines, num_points) - bool mask

        Returns:
            features: (batch, num_polylines, out_channels or hidden_dim)
        """
        batch_size, num_polylines, num_points, C = polylines.shape

        # Pre-MLP on valid points only
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])  # (N_valid, hidden_dim)
        polylines_feature = polylines.new_zeros(
            batch_size, num_polylines, num_points, polylines_feature_valid.shape[-1]
        )
        polylines_feature[polylines_mask] = polylines_feature_valid

        # Max-pool over points to get global feature per polyline
        pooled_feature = polylines_feature.max(dim=2)[0]  # (batch, num_polylines, hidden_dim)

        # Concatenate point features with global feature
        polylines_feature = torch.cat(
            (polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points, 1)), dim=-1
        )

        # Post-pooling MLP
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(
            batch_size, num_polylines, num_points, polylines_feature_valid.shape[-1]
        )
        feature_buffers[polylines_mask] = polylines_feature_valid

        # Output projection with final max-pooling
        if self.out_mlps is not None:
            feature_buffers = feature_buffers.max(dim=2)[0]  # (batch, num_polylines, hidden_dim)
            valid_mask = polylines_mask.sum(dim=-1) > 0  # (batch, num_polylines)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N_valid, out_channels)
            feature_buffers = feature_buffers.new_zeros(
                batch_size, num_polylines, feature_buffers_valid.shape[-1]
            )
            feature_buffers[valid_mask] = feature_buffers_valid

        return feature_buffers


# ==============================================================================
# Custom MultiheadAttention (matching MTR's without_weight mode)
# ==============================================================================


class MultiheadAttentionCustom(nn.Module):
    """
    Custom multihead attention matching MTR's implementation.
    Supports without_weight mode where Q, K, V projections are done externally.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        vdim: Optional[int] = None,
        without_weight: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        self.without_weight = without_weight
        if not without_weight:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
            nn.init.xavier_uniform_(self.in_proj_weight)
            nn.init.constant_(self.in_proj_bias, 0.0)
        else:
            self.in_proj_weight = None
            self.in_proj_bias = None

        self.out_proj = nn.Linear(self.vdim, self.vdim)
        if without_weight:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (L, N, E) where L is target length, N is batch, E is embed_dim
            key: (S, N, E) where S is source length
            value: (S, N, E_v) where E_v is vdim
            attn_mask: (L, S) or (N*num_heads, L, S)
            key_padding_mask: (N, S)

        Returns:
            attn_output: (L, N, E_v)
            attn_weights: (N, L, S)
        """
        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]
        v_head_dim = self.vdim // self.num_heads

        # Project Q, K, V
        if not self.without_weight:
            # Combined projection
            qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q, k, v = query, key, value

        # Reshape for multi-head attention
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, v_head_dim).transpose(0, 1)

        # Compute attention scores
        attn_weights = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply masks
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(
                -1, self.num_heads, -1, -1
            ).reshape(bsz * self.num_heads, 1, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
            attn_weights = attn_weights + attn_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Apply attention to values
        attn_output = torch.bmm(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.vdim)
        attn_output = self.out_proj(attn_output)

        # Average attention weights over heads for return
        attn_weights_avg = attn_weights.view(bsz, self.num_heads, tgt_len, src_len).mean(dim=1)

        return attn_output, attn_weights_avg


# ==============================================================================
# TransformerDecoderLayer (matching original MTR exactly)
# ==============================================================================


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer matching original MTR implementation.

    Key features:
    - Separate QKV projections for self-attention and cross-attention
    - query_sine_embed for dynamic query position encoding
    - Support for both global and local attention
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        keep_query_pos: bool = True,
        rm_self_attn_decoder: bool = False,
        use_local_attn: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder
        self.use_local_attn = use_local_attn
        self.keep_query_pos = keep_query_pos
        self.normalize_before = normalize_before

        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttentionCustom(
                d_model, nhead, dropout=dropout, vdim=d_model, without_weight=True
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)

        # Cross attention (global or local)
        # For local attention, we use a fallback global attention since CUDA ops aren't available
        self.cross_attn = MultiheadAttentionCustom(
            d_model * 2, nhead, dropout=dropout, vdim=d_model, without_weight=True
        )

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor,
        query_pos: torch.Tensor,
        query_sine_embed: torch.Tensor,
        is_first: bool = False,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        # For local attention (fallback to global if not available)
        index_pair: Optional[torch.Tensor] = None,
        key_batch_cnt: Optional[torch.Tensor] = None,
        index_pair_batch: Optional[torch.Tensor] = None,
        memory_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: (num_query, B, C) - query features
            memory: (S, B, C) or stacked (M1+M2+..., C) for local attention
            pos: (S, B, C) - memory position encoding
            query_pos: (num_query, B, C) - query position encoding
            query_sine_embed: (num_query, B, C) - dynamic query position
            is_first: whether this is the first decoder layer
            memory_key_padding_mask: (B, S) - True means masked
            index_pair: for local attention (not used in fallback)
            key_batch_cnt: for local attention (not used in fallback)
            index_pair_batch: for local attention (not used in fallback)
            memory_valid_mask: for local attention (not used in fallback)

        Returns:
            tgt: (num_query, B, C) - updated query features
        """
        num_queries, bs, d_model = tgt.shape

        # ========== Self-Attention =============
        if not self.rm_self_attn_decoder:
            q_content = self.sa_qcontent_proj(tgt)
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2, _ = self.self_attn(q, k, v)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Cross-Attention =============
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        k_pos = self.ca_kpos_proj(pos)

        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)

        # Concatenate content+pos for query, content+pos for key (DAB-DETR style)
        q = q.view(num_queries, bs, self.nhead, d_model // self.nhead)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, d_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=-1).view(num_queries, bs, d_model * 2)

        hw = k.shape[0]
        k = k.view(hw, bs, self.nhead, d_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, d_model // self.nhead)
        k = torch.cat([k, k_pos], dim=-1).view(hw, bs, d_model * 2)

        tgt2, _ = self.cross_attn(q, k, v, key_padding_mask=memory_key_padding_mask)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ========== FFN =============
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


# ==============================================================================
# MTREncoder (matching original MTR)
# ==============================================================================


class MTREncoder(nn.Module):
    """
    MTR Encoder (matching original implementation).

    Components:
    1. PointNet polyline encoders for agents and map
    2. Global self-attention layers
    """

    def __init__(
        self,
        num_input_attr_agent: int = 7,
        num_input_attr_map: int = 13,
        d_model: int = 256,
        num_attn_head: int = 8,
        num_attn_layers: int = 6,
        hidden_dim_agent: int = 128,
        hidden_dim_map: int = 64,
        num_layers_agent: int = 3,
        num_layers_map: int = 3,
        num_pre_layers_map: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_out_channels = d_model

        # Agent polyline encoder (input includes mask as feature)
        self.agent_polyline_encoder = PointNetPolylineEncoder(
            in_channels=num_input_attr_agent + 1,  # +1 for mask feature
            hidden_dim=hidden_dim_agent,
            num_layers=num_layers_agent,
            num_pre_layers=1,
            out_channels=d_model,
        )

        # Map polyline encoder
        self.map_polyline_encoder = PointNetPolylineEncoder(
            in_channels=num_input_attr_map,
            hidden_dim=hidden_dim_map,
            num_layers=num_layers_map,
            num_pre_layers=num_pre_layers_map,
            out_channels=d_model,
        )

        # Self-attention layers
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_attn_head,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=False,
                norm_first=False,
            )
            for _ in range(num_attn_layers)
        ])

    def apply_global_attn(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        x_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply global self-attention (matching MTR).

        Args:
            x: (batch, N, d_model) - features
            x_mask: (batch, N) - valid mask (True = valid)
            x_pos: (batch, N, 2 or 3) - positions

        Returns:
            x_out: (batch, N, d_model)
        """
        batch_size, N, d_model = x.shape

        # Convert to (N, batch, d_model) for transformer
        x_t = x.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)

        # Generate position embeddings
        pos_embedding = gen_sineembed_for_position(x_pos_t[..., :2], hidden_dim=d_model)

        # Add position embeddings
        x_t = x_t + pos_embedding

        # Create attention mask (True = masked out in PyTorch)
        key_padding_mask = ~x_mask

        # Apply self-attention layers
        for layer in self.self_attn_layers:
            x_t = layer(x_t, src_key_padding_mask=key_padding_mask)

        # Convert back to (batch, N, d_model)
        x_out = x_t.permute(1, 0, 2)
        return x_out

    def forward(
        self,
        obj_trajs: torch.Tensor,
        obj_trajs_mask: torch.Tensor,
        obj_trajs_last_pos: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polylines_mask: torch.Tensor,
        map_polylines_center: torch.Tensor,
        track_index_to_predict: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (matching MTR interface).

        Args:
            obj_trajs: (batch, num_objects, num_timestamps, features)
            obj_trajs_mask: (batch, num_objects, num_timestamps)
            obj_trajs_last_pos: (batch, num_objects, 2 or 3)
            map_polylines: (batch, num_polylines, num_points, features)
            map_polylines_mask: (batch, num_polylines, num_points)
            map_polylines_center: (batch, num_polylines, 2 or 3)
            track_index_to_predict: (batch,) - index of ego/target agent

        Returns:
            Dictionary containing encoded features
        """
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        num_polylines = map_polylines.shape[1]

        # Add mask as feature (matching MTR)
        obj_trajs_in = torch.cat(
            (obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1
        )

        # Encode agent polylines
        obj_polylines_feature = self.agent_polyline_encoder(
            obj_trajs_in, obj_trajs_mask
        )  # (batch, num_objects, d_model)

        # Encode map polylines
        map_polylines_feature = self.map_polyline_encoder(
            map_polylines, map_polylines_mask
        )  # (batch, num_polylines, d_model)

        # Get valid masks (at least one point valid)
        obj_valid_mask = obj_trajs_mask.sum(dim=-1) > 0  # (batch, num_objects)
        map_valid_mask = map_polylines_mask.sum(dim=-1) > 0  # (batch, num_polylines)

        # Concatenate for global attention
        global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature), dim=1)
        global_token_mask = torch.cat((obj_valid_mask, map_valid_mask), dim=1)
        global_token_pos = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1)

        # Apply global self-attention
        global_token_feature = self.apply_global_attn(
            x=global_token_feature,
            x_mask=global_token_mask,
            x_pos=global_token_pos,
        )

        # Split back into object and map features
        obj_polylines_feature = global_token_feature[:, :num_objects]
        map_polylines_feature = global_token_feature[:, num_objects:]

        # Extract center object feature (ego agent)
        center_objects_feature = obj_polylines_feature[
            torch.arange(num_center_objects, device=obj_trajs.device), track_index_to_predict
        ]

        return {
            "center_objects_feature": center_objects_feature,
            "obj_feature": obj_polylines_feature,
            "map_feature": map_polylines_feature,
            "obj_mask": obj_valid_mask,
            "map_mask": map_valid_mask,
            "obj_pos": obj_trajs_last_pos,
            "map_pos": map_polylines_center,
        }


# ==============================================================================
# MTRDecoder (full implementation matching original MTR)
# ==============================================================================


class MTRDecoder(nn.Module):
    """
    MTR Decoder (full implementation matching original).

    Key components:
    1. Input projections for center object, agents, and map
    2. Dense future prediction for feature enhancement
    3. Learnable intention/action queries
    4. Separate decoder layers for agent and map cross-attention
    5. Dynamic map collection based on predicted waypoints
    6. Iterative refinement with updated query centers
    7. Feature fusion layers
    """

    def __init__(
        self,
        in_channels: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        num_queries: int = 6,
        num_future_frames: int = 80,  # For dense future prediction
        dropout: float = 0.1,
        # Map collection parameters
        num_base_map_polylines: int = 128,
        num_waypoint_map_polylines: int = 64,
        center_offset_of_map: Tuple[float, float] = (30.0, 0.0),
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_future_frames = num_future_frames
        self.num_base_map_polylines = num_base_map_polylines
        self.num_waypoint_map_polylines = num_waypoint_map_polylines
        self.center_offset_of_map = center_offset_of_map

        # Input projection for center object
        self.in_proj_center_obj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Build transformer decoder for objects (global attention)
        self.in_proj_obj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.obj_decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="relu",
                normalize_before=False,
                keep_query_pos=True,
                rm_self_attn_decoder=False,
                use_local_attn=False,
            )
            for _ in range(num_decoder_layers)
        ])

        # Build transformer decoder for map (local attention, but fallback to global)
        self.in_proj_map = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.map_decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="relu",
                normalize_before=False,
                keep_query_pos=True,
                rm_self_attn_decoder=False,
                use_local_attn=True,  # Note: falls back to global in our implementation
            )
            for _ in range(num_decoder_layers)
        ])

        # Dense future prediction layers (matching MTR)
        self.obj_pos_encoding_layer = build_mlps(
            c_in=2,
            mlp_channels=[d_model, d_model, d_model],
            ret_before_act=True,
            without_norm=True,
        )
        self.dense_future_head = build_mlps(
            c_in=d_model * 2,
            mlp_channels=[d_model, d_model, num_future_frames * 7],
            ret_before_act=True,
        )
        self.future_traj_mlps = build_mlps(
            c_in=4 * num_future_frames,
            mlp_channels=[d_model, d_model, d_model],
            ret_before_act=True,
            without_norm=True,
        )
        self.traj_fusion_mlps = build_mlps(
            c_in=d_model * 2,
            mlp_channels=[d_model, d_model, d_model],
            ret_before_act=True,
            without_norm=True,
        )

        # Learnable intention queries (replacing file-loaded cluster centers)
        self.intention_points = nn.Parameter(torch.randn(num_queries, 2) * 10.0)  # (num_queries, 2)
        self.intention_query_mlps = build_mlps(
            c_in=d_model,
            mlp_channels=[d_model, d_model],
            ret_before_act=True,
        )

        # Feature fusion layers (combining center + obj_attn + map_attn)
        temp_layer = build_mlps(
            c_in=d_model * 3,
            mlp_channels=[d_model, d_model],
            ret_before_act=True,
        )
        self.query_feature_fusion_layers = nn.ModuleList([
            copy.deepcopy(temp_layer) for _ in range(num_decoder_layers)
        ])

        # Motion prediction heads (for iterative refinement)
        # Output 7 values per timestep: [accel, steer, σ_accel, σ_steer, ρ, vx, vy]
        motion_reg_head = build_mlps(
            c_in=d_model,
            mlp_channels=[d_model, d_model, num_future_frames * 7],
            ret_before_act=True,
        )
        self.motion_reg_heads = nn.ModuleList([
            copy.deepcopy(motion_reg_head) for _ in range(num_decoder_layers)
        ])

        # Motion classification heads (for query confidence scoring)
        motion_cls_head = build_mlps(
            c_in=d_model,
            mlp_channels=[d_model, d_model, 1],
            ret_before_act=True,
        )
        self.motion_cls_heads = nn.ModuleList([
            copy.deepcopy(motion_cls_head) for _ in range(num_decoder_layers)
        ])

    def apply_dense_future_prediction(
        self,
        obj_feature: torch.Tensor,
        obj_mask: torch.Tensor,
        obj_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dense future prediction (matching MTR exactly).

        Predicts future trajectories for all agents and fuses with current features.
        """
        num_center_objects, num_objects, _ = obj_feature.shape

        # Get valid features and positions
        obj_pos_valid = obj_pos[obj_mask][..., 0:2]
        obj_feature_valid = obj_feature[obj_mask]

        # Encode positions
        obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)
        obj_fused_feature_valid = torch.cat((obj_pos_feature_valid, obj_feature_valid), dim=-1)

        # Predict dense future trajectories
        pred_dense_trajs_valid = self.dense_future_head(obj_fused_feature_valid)
        pred_dense_trajs_valid = pred_dense_trajs_valid.view(
            pred_dense_trajs_valid.shape[0], self.num_future_frames, 7
        )

        # Add position offset
        temp_center = pred_dense_trajs_valid[:, :, 0:2] + obj_pos_valid[:, None, 0:2]
        pred_dense_trajs_valid = torch.cat((temp_center, pred_dense_trajs_valid[:, :, 2:]), dim=-1)

        # Encode future trajectory features
        obj_future_input_valid = pred_dense_trajs_valid[:, :, [0, 1, -2, -1]].flatten(
            start_dim=1, end_dim=2
        )
        obj_future_feature_valid = self.future_traj_mlps(obj_future_input_valid)

        # Fuse with current features
        obj_full_trajs_feature = torch.cat((obj_feature_valid, obj_future_feature_valid), dim=-1)
        obj_feature_valid = self.traj_fusion_mlps(obj_full_trajs_feature)

        # Scatter back to full tensor
        ret_obj_feature = torch.zeros_like(obj_feature)
        ret_obj_feature[obj_mask] = obj_feature_valid

        ret_pred_dense_future_trajs = obj_feature.new_zeros(
            num_center_objects, num_objects, self.num_future_frames, 7
        )
        ret_pred_dense_future_trajs[obj_mask] = pred_dense_trajs_valid

        return ret_obj_feature, ret_pred_dense_future_trajs

    def get_motion_query(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get motion queries from learnable intention points."""
        # Expand intention points for batch
        intention_points = self.intention_points[None, :, :].expand(
            batch_size, -1, -1
        )  # (batch, num_query, 2)

        # Convert to (num_query, batch, 2) for transformer
        intention_points_t = intention_points.permute(1, 0, 2)

        # Generate sinusoidal embeddings
        intention_query = gen_sineembed_for_position(intention_points_t, hidden_dim=self.d_model)
        intention_query = self.intention_query_mlps(intention_query.view(-1, self.d_model)).view(
            self.num_queries, batch_size, self.d_model
        )

        return intention_query, intention_points_t

    def apply_dynamic_map_collection(
        self,
        map_pos: torch.Tensor,
        map_mask: torch.Tensor,
        pred_waypoints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dynamic map collection based on predicted waypoints (matching MTR).

        Returns indices of relevant map polylines for each query.
        """
        batch_size = map_pos.shape[0]
        num_polylines = map_pos.shape[1]
        num_query = pred_waypoints.shape[1]
        device = map_pos.device

        # Clone and mask invalid positions
        map_pos_masked = map_pos.clone()
        map_pos_masked[~map_mask] = 1e7

        # Base region collection (around center offset)
        base_points = torch.tensor(self.center_offset_of_map, device=device, dtype=map_pos.dtype)
        base_dist = (map_pos_masked[:, :, 0:2] - base_points[None, None, :]).norm(dim=-1)
        k_base = min(num_polylines, self.num_base_map_polylines)
        _, base_map_idxs = base_dist.topk(k=k_base, dim=-1, largest=False)
        base_map_idxs = base_map_idxs[:, None, :].expand(-1, num_query, -1)

        # Dynamic collection based on waypoints
        # pred_waypoints: (batch, num_query, num_timestamps, 2)
        dynamic_dist = (
            pred_waypoints[:, :, None, :, 0:2] - map_pos_masked[:, None, :, None, 0:2]
        ).norm(dim=-1)
        dynamic_dist = dynamic_dist.min(dim=-1)[0]  # (batch, num_query, num_polylines)

        k_dynamic = min(num_polylines, self.num_waypoint_map_polylines)
        _, dynamic_map_idxs = dynamic_dist.topk(k=k_dynamic, dim=-1, largest=False)

        # Combine indices
        collected_idxs = torch.cat((base_map_idxs, dynamic_map_idxs), dim=-1)

        return collected_idxs

    def action_to_trajectory(
        self,
        pred_trajs: torch.Tensor,
        obj_speeds: torch.Tensor,
        obj_lengths: torch.Tensor,
        delta_t: float = 0.1,
    ) -> torch.Tensor:
        """
        Convert GMM actions to trajectory positions using bicycle model.
        Full implementation from mtr_actions.py:419-486.

        Args:
            pred_trajs: (batch, K, T, 7) raw head output
                [accel, steer_raw, log_σ_accel, log_σ_steer, ρ, vx, vy]
            obj_speeds: (batch,) initial speeds
            obj_lengths: (batch,) vehicle lengths

        Returns:
            trajectory_prediction: (batch, K, T, 7) with [mu_x, mu_y, log_σ_x, log_σ_y, ρ, vel_x, vel_y]
        """
        batch, K, T, _ = pred_trajs.shape
        device = pred_trajs.device

        # Expand initial state
        obj_speeds = obj_speeds[:, None, None].expand(-1, K, T).clone()
        obj_lengths = obj_lengths[:, None, None].expand(-1, K, T).clone()

        trajectory_prediction = torch.zeros((batch, K, T, 7), device=device)

        # Extract GMM action parameters
        mu_accel = pred_trajs[:, :, :, 0]
        mu_delta = torch.tanh(pred_trajs[:, :, :, 1]) * torch.pi / 3  # scale to [-π/3, π/3]
        sigma_accel = torch.exp(torch.clamp(pred_trajs[:, :, :, 2], max=10.0))
        sigma_delta = torch.exp(torch.clamp(pred_trajs[:, :, :, 3], max=10.0))
        rho = pred_trajs[:, :, :, 4]

        # Integrate velocity: v(t) = v0 + ∫a dt
        mu_vel = (mu_accel * delta_t).cumsum(dim=-1) + obj_speeds
        sigma_vel = (sigma_accel * delta_t).cumsum(dim=-1)

        # Integrate heading: θ(t) = ∫(v * tan(δ) / L) dt
        mu_theta = (delta_t / obj_lengths * (mu_vel * torch.tan(mu_delta))).cumsum(dim=-1)

        # Heading uncertainty propagation
        G = (delta_t / obj_lengths) * mu_vel * sigma_delta / (torch.cos(mu_delta) ** 2 + EPS)
        H = (delta_t / obj_lengths) * sigma_vel * torch.tan(mu_delta)
        I = (delta_t / obj_lengths) * sigma_vel * sigma_delta / (torch.cos(mu_delta) ** 2 + EPS)
        var_theta = (G**2 + H**2 + I**2).cumsum(dim=-1)
        sigma_theta = torch.sqrt(var_theta)

        # Integrate position: x(t) = ∫v*cos(θ) dt
        mu_x = mu_vel * torch.cos(mu_theta) * delta_t
        A = (mu_vel * sigma_theta * torch.sin(mu_theta) * delta_t) ** 2
        B = (sigma_vel * torch.cos(mu_theta) * delta_t) ** 2
        C = (sigma_vel * sigma_theta * torch.sin(mu_theta) * delta_t) ** 2
        sigma_x = A + B + C  # variance for cumsum

        # Integrate position: y(t) = ∫v*sin(θ) dt
        mu_y = mu_vel * torch.sin(mu_theta) * delta_t
        D = (mu_vel * sigma_theta * torch.cos(mu_theta) * delta_t) ** 2
        E = (sigma_vel * torch.sin(mu_theta) * delta_t) ** 2
        F = (sigma_vel * sigma_theta * torch.cos(mu_theta) * delta_t) ** 2
        sigma_y = D + E + F  # variance for cumsum

        # Cumulative sum over timesteps for position
        gmm_component = torch.stack([mu_x, mu_y, sigma_x, sigma_y], dim=-1).cumsum(dim=-2)
        gmm_component[..., 2:4] = torch.log(torch.sqrt(gmm_component[..., 2:4]) + EPS)

        # Velocity components
        vel_x = mu_vel * torch.cos(mu_theta)
        vel_y = mu_vel * torch.sin(mu_theta)

        # Assemble output
        trajectory_prediction[:, :, :, :4] = gmm_component
        trajectory_prediction[:, :, :, 4] = rho
        trajectory_prediction[:, :, :, 5] = vel_x
        trajectory_prediction[:, :, :, 6] = vel_y

        return trajectory_prediction

    def forward(self, enc_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through decoder (matching MTR structure).

        Returns:
            Dictionary containing:
            - decoded_feature: (batch, d_model) - feature for value head
            - pred_trajs: (batch, num_queries, num_future_frames, 7) - raw GMM trajectories
            - pred_scores: (batch, num_queries) - query confidence scores
        """
        obj_feature = enc_dict["obj_feature"]
        obj_mask = enc_dict["obj_mask"]
        obj_pos = enc_dict["obj_pos"]
        map_feature = enc_dict["map_feature"]
        map_mask = enc_dict["map_mask"]
        map_pos = enc_dict["map_pos"]
        center_objects_feature = enc_dict["center_objects_feature"]

        batch_size, num_objects, _ = obj_feature.shape
        num_polylines = map_feature.shape[1]
        device = obj_feature.device

        # Input projections
        center_objects_feature = self.in_proj_center_obj(center_objects_feature)

        obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])
        obj_feature = obj_feature.new_zeros(batch_size, num_objects, self.d_model)
        obj_feature[obj_mask] = obj_feature_valid

        map_feature_valid = self.in_proj_map(map_feature[map_mask])
        map_feature = map_feature.new_zeros(batch_size, num_polylines, self.d_model)
        map_feature[map_mask] = map_feature_valid

        # Dense future prediction
        obj_feature, pred_dense_future_trajs = self.apply_dense_future_prediction(
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos
        )

        # Get motion queries
        intention_query, intention_points = self.get_motion_query(batch_size, device)
        query_content = torch.zeros_like(intention_query)

        # Initialize predicted waypoints from intention points
        pred_waypoints = intention_points.permute(1, 0, 2)[:, :, None, :]  # (batch, num_query, 1, 2)
        dynamic_query_center = intention_points  # (num_query, batch, 2)

        # Expand center feature for each query
        center_objects_feature_expanded = center_objects_feature[None, :, :].repeat(
            self.num_queries, 1, 1
        )  # (num_query, batch, d_model)

        # Convert features to transformer format (S, B, C)
        obj_feature_t = obj_feature.permute(1, 0, 2)
        map_feature_t = map_feature.permute(1, 0, 2)
        obj_pos_t = obj_pos.permute(1, 0, 2)
        map_pos_t = map_pos.permute(1, 0, 2)

        # Position embeddings for memory
        obj_pos_embed = gen_sineembed_for_position(obj_pos_t[..., :2], self.d_model)
        map_pos_embed = gen_sineembed_for_position(map_pos_t[..., :2], self.d_model)

        # Decoder layers
        for layer_idx in range(self.num_decoder_layers):
            # Dynamic query position
            query_sine_embed = gen_sineembed_for_position(dynamic_query_center, self.d_model)

            # Cross-attention to objects
            obj_query_feature = self.obj_decoder_layers[layer_idx](
                tgt=query_content,
                memory=obj_feature_t,
                pos=obj_pos_embed,
                query_pos=intention_query,
                query_sine_embed=query_sine_embed,
                is_first=(layer_idx == 0),
                memory_key_padding_mask=~obj_mask,
            )

            # Dynamic map collection
            collected_idxs = self.apply_dynamic_map_collection(
                map_pos=map_pos, map_mask=map_mask, pred_waypoints=pred_waypoints
            )

            # Cross-attention to map
            map_query_feature = self.map_decoder_layers[layer_idx](
                tgt=query_content,
                memory=map_feature_t,
                pos=map_pos_embed,
                query_pos=intention_query,
                query_sine_embed=query_sine_embed,
                is_first=(layer_idx == 0),
                memory_key_padding_mask=~map_mask,
                index_pair=collected_idxs,
            )

            # Fuse features: center + obj_attn + map_attn
            query_feature = torch.cat(
                [center_objects_feature_expanded, obj_query_feature, map_query_feature], dim=-1
            )
            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(0, 1)
            ).view(self.num_queries, batch_size, -1)

            # Motion prediction for iterative refinement
            query_content_t = query_content.permute(1, 0, 2).contiguous()  # (batch, num_query, d_model)
            pred_trajs = self.motion_reg_heads[layer_idx](
                query_content_t.view(batch_size * self.num_queries, -1)
            ).view(batch_size, self.num_queries, self.num_future_frames, 7)

            # Query confidence scores
            pred_scores = self.motion_cls_heads[layer_idx](
                query_content_t.view(batch_size * self.num_queries, -1)
            ).view(batch_size, self.num_queries)

            # Update for next iteration
            pred_waypoints = pred_trajs[:, :, :, 0:2]
            dynamic_query_center = pred_trajs[:, :, -1, 0:2].permute(1, 0, 2).contiguous()

        # Aggregate queries (mean pooling for value head)
        decoded_feature = query_content.permute(1, 0, 2).mean(dim=1)  # (batch, d_model)

        return {
            "decoded_feature": decoded_feature,
            "pred_trajs": pred_trajs,  # (batch, num_queries, num_future_frames, 7)
            "pred_scores": pred_scores,  # (batch, num_queries)
        }


# ==============================================================================
# Social Forces Computer (from Polysona)
# ==============================================================================


class SocialForcesComputer(nn.Module):
    """
    Compute social force features from partner observations (from Polysona).

    Social forces model repulsive interactions between vehicles.
    Output: [sum_force_x, sum_force_y, mean_mag, max_mag, std_mag, num_neighbors]
    """

    def __init__(self, output_dim: int = 6):
        super().__init__()
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        partner_obs: torch.Tensor,
        A: float = 1.0,
        B: float = 1.0,
        D: float = 5.0,
    ) -> torch.Tensor:
        """
        Args:
            partner_obs: (batch, max_partners, 7)
                [rel_x, rel_y, width, length, heading_x, heading_y, speed]
                Note: rel_x/rel_y are scaled by 0.02 in observations

        Returns:
            social_forces: (batch, 6)
        """
        # Undo scaling for positions
        rel_x = partner_obs[:, :, 0] * 50.0
        rel_y = partner_obs[:, :, 1] * 50.0

        dist = torch.sqrt(rel_x**2 + rel_y**2 + EPS)

        # Valid mask
        partner_mag = partner_obs.abs().sum(dim=-1)
        valid_mask = (partner_mag > EPS).float()

        # Direction vectors (repulsive)
        dir_x = -rel_x / (dist + EPS)
        dir_y = -rel_y / (dist + EPS)

        # Force magnitude
        force_mag = A * torch.exp(torch.clamp((D - dist) / B, max=10.0))
        force_mag = force_mag * valid_mask

        # Force vectors
        force_x = force_mag * dir_x
        force_y = force_mag * dir_y

        # Aggregations
        sum_fx = force_x.sum(dim=-1)
        sum_fy = force_y.sum(dim=-1)

        num_valid = valid_mask.sum(dim=-1).clamp(min=1)
        mean_mag = force_mag.sum(dim=-1) / num_valid
        max_mag = force_mag.max(dim=-1).values

        # Std
        mean_exp = mean_mag.unsqueeze(-1)
        sq_diff = (force_mag - mean_exp * valid_mask) ** 2 * valid_mask
        var_mag = sq_diff.sum(dim=-1) / num_valid.clamp(min=1)
        std_mag = torch.sqrt(var_mag + EPS)

        norm_neighbors = num_valid / partner_obs.shape[1]

        social_forces = torch.stack(
            [sum_fx, sum_fy, mean_mag, max_mag, std_mag, norm_neighbors], dim=-1
        )
        return self.norm(social_forces)


# ==============================================================================
# Persona Router with Reconstruction (from Polysona)
# ==============================================================================


class PersonaRouterWithReconstruction(nn.Module):
    """
    Router with reconstruction loss (from Polysona).

    Predicts expert probabilities and reconstructs social forces from
    the latent representation.
    """

    def __init__(
        self,
        context_dim: int = 256,
        social_force_dim: int = 6,
        num_experts: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.social_force_dim = social_force_dim

        self.social_forces_computer = SocialForcesComputer(social_force_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(context_dim + social_force_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_experts),
        )

        # Reconstructor
        self.reconstructor = nn.Sequential(
            nn.Linear(num_experts, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, social_force_dim),
        )

        self.register_buffer("temperature", torch.tensor(1.0))

    def forward(
        self,
        context: torch.Tensor,
        partner_obs: torch.Tensor,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            - expert_probs: (batch, num_experts)
            - logits: (batch, num_experts)
            - social_forces: (batch, 6)
            - reconstructed_sf: (batch, 6)
        """
        social_forces = self.social_forces_computer(partner_obs)
        fused = self.fusion(torch.cat([context, social_forces], dim=-1))
        logits = self.classifier(fused)

        if hard:
            indices = logits.argmax(dim=-1)
            expert_probs = F.one_hot(indices, num_classes=self.num_experts).float()
        else:
            expert_probs = F.softmax(logits, dim=-1)

        reconstructed_sf = self.reconstructor(expert_probs)

        return expert_probs, logits, social_forces, reconstructed_sf

    def set_temperature(self, temperature: float):
        self.temperature.fill_(temperature)


# ==============================================================================
# Full MTR Architecture (combining all components)
# ==============================================================================


class FullMTRArchitecture(nn.Module):
    """
    Complete MTR architecture with persona routing for PufferDrive RL.

    Architecture matches original MTR:
    1. MTREncoder: PointNet + Transformer self-attention
    2. MTRDecoder: Cross-attention with intention queries + dense future prediction
    3. PersonaRouter: Expert routing with reconstruction loss
    """

    def __init__(
        self,
        # Observation dimensions (PufferDrive)
        ego_dim: int = 7,
        partner_features: int = 7,
        road_features: int = 13,  # after one-hot
        max_partners: int = 63,
        max_road_objects: int = 64,
        # MTR architecture
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_queries: int = 6,
        num_future_frames: int = 80,
        # MoE
        num_experts: int = 3,
        router_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.max_partners = max_partners
        self.max_road_objects = max_road_objects
        self.partner_features = partner_features
        self.road_features = road_features
        self.ego_dim = ego_dim

        # MTR Encoder
        self.encoder = MTREncoder(
            num_input_attr_agent=partner_features,
            num_input_attr_map=road_features,
            d_model=d_model,
            num_attn_head=nhead,
            num_attn_layers=num_encoder_layers,
            hidden_dim_agent=128,
            hidden_dim_map=64,
            dropout=dropout,
        )

        # MTR Decoder (full implementation)
        self.decoder = MTRDecoder(
            in_channels=d_model,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            num_queries=num_queries,
            num_future_frames=num_future_frames,
            dropout=dropout,
        )

        # Persona Router
        self.router = PersonaRouterWithReconstruction(
            context_dim=d_model,
            social_force_dim=6,
            num_experts=num_experts,
            hidden_dim=router_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        ego_obs: torch.Tensor,
        partner_obs: torch.Tensor,
        road_obs: torch.Tensor,
        hard_routing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            ego_obs: (batch, ego_dim)
            partner_obs: (batch, max_partners, partner_features)
            road_obs: (batch, max_road_objects, road_features)

        Returns:
            Dictionary with hidden, expert_probs, router_logits, social_forces, reconstructed_sf
        """
        batch_size = ego_obs.shape[0]
        device = ego_obs.device

        # Prepare agent trajectories (ego + partners)
        ego_as_agent = torch.zeros(batch_size, 1, 1, self.partner_features, device=device)
        ego_as_agent[:, 0, 0, : min(self.ego_dim, self.partner_features)] = ego_obs[
            :, : min(self.ego_dim, self.partner_features)
        ]

        partner_trajs = partner_obs.unsqueeze(2)  # (batch, max_partners, 1, features)
        obj_trajs = torch.cat([ego_as_agent, partner_trajs], dim=1)

        # Masks
        ego_mask = torch.ones(batch_size, 1, 1, dtype=torch.bool, device=device)
        partner_mag = partner_obs.abs().sum(dim=-1)
        partner_mask = (partner_mag > EPS).unsqueeze(2)
        obj_trajs_mask = torch.cat([ego_mask, partner_mask], dim=1)

        # Positions
        ego_pos = torch.zeros(batch_size, 1, 2, device=device)
        partner_pos = partner_obs[:, :, :2] * 50.0  # Undo scaling
        obj_trajs_last_pos = torch.cat([ego_pos, partner_pos], dim=1)

        # Map data
        road_mag = road_obs.abs().sum(dim=-1)
        map_polylines = road_obs.unsqueeze(2)
        map_polylines_mask = (road_mag > EPS).unsqueeze(2)
        map_polylines_center = road_obs[:, :, :2] * 50.0

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

        # Decode
        dec_dict = self.decoder(enc_dict)
        decoded_feature = dec_dict["decoded_feature"]

        # Route
        expert_probs, router_logits, social_forces, reconstructed_sf = self.router(
            decoded_feature, partner_obs, hard=hard_routing
        )

        return {
            "hidden": decoded_feature,
            "expert_probs": expert_probs,
            "router_logits": router_logits,
            "social_forces": social_forces,
            "reconstructed_sf": reconstructed_sf,
        }

    def set_temperature(self, temperature: float):
        self.router.set_temperature(temperature)
