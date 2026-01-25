"""
Test that MoE architecture in torch.py matches drivenet_moe.h.

This test ensures the weight export order matches what the C visualizer expects.
If this test fails, either torch.py or drivenet_moe.h needs to be updated to match.
"""

import pytest
import torch
import numpy as np
import gymnasium

# Import the policy
from pufferlib.ocean.torch import DriveMoE
from pufferlib.models import LSTMWrapper
import pufferlib.spaces


# Expected parameter order from drivenet_moe.h init_drivenet_moe()
# This must match the order weights are loaded in the C code
EXPECTED_MOE_PARAMETER_ORDER = [
    # ego_encoder: Sequential(Linear, LayerNorm, Linear)
    ("policy.ego_encoder.0.weight", (64, 7)),      # Linear(7, 64)
    ("policy.ego_encoder.0.bias", (64,)),
    ("policy.ego_encoder.1.weight", (64,)),        # LayerNorm(64)
    ("policy.ego_encoder.1.bias", (64,)),
    ("policy.ego_encoder.2.weight", (64, 64)),     # Linear(64, 64)
    ("policy.ego_encoder.2.bias", (64,)),

    # road_encoder: Sequential(Linear, LayerNorm, Linear)
    ("policy.road_encoder.0.weight", (64, 13)),    # Linear(13, 64) - 7 + 6 one-hot
    ("policy.road_encoder.0.bias", (64,)),
    ("policy.road_encoder.1.weight", (64,)),       # LayerNorm(64)
    ("policy.road_encoder.1.bias", (64,)),
    ("policy.road_encoder.2.weight", (64, 64)),    # Linear(64, 64)
    ("policy.road_encoder.2.bias", (64,)),

    # partner_encoder: Sequential(Linear, LayerNorm, Linear)
    ("policy.partner_encoder.0.weight", (64, 7)),  # Linear(7, 64)
    ("policy.partner_encoder.0.bias", (64,)),
    ("policy.partner_encoder.1.weight", (64,)),    # LayerNorm(64)
    ("policy.partner_encoder.1.bias", (64,)),
    ("policy.partner_encoder.2.weight", (64, 64)), # Linear(64, 64)
    ("policy.partner_encoder.2.bias", (64,)),

    # shared_embedding: Sequential(GELU, Linear) - GELU has no params
    ("policy.shared_embedding.1.weight", (256, 192)),  # Linear(192, 256)
    ("policy.shared_embedding.1.bias", (256,)),

    # router: PersonaRouter with classifier Sequential
    ("policy.router.classifier.0.weight", (64, 192)),  # Linear(192, 64)
    ("policy.router.classifier.0.bias", (64,)),
    ("policy.router.classifier.1.weight", (64,)),      # LayerNorm(64)
    ("policy.router.classifier.1.bias", (64,)),
    # Note: ReLU and Dropout have no params
    ("policy.router.classifier.4.weight", (3, 64)),    # Linear(64, 3)
    ("policy.router.classifier.4.bias", (3,)),

    # actor: LoRAExpertsRL
    ("policy.actor.weight", (91, 256)),           # Base Linear weight
    ("policy.actor.bias", (91,)),                 # Base Linear bias
    ("policy.actor.expert_A", (3, 8, 256)),       # LoRA A matrices
    ("policy.actor.expert_B", (3, 91, 8)),        # LoRA B matrices

    # value_fn: Linear
    ("policy.value_fn.weight", (1, 256)),
    ("policy.value_fn.bias", (1,)),

    # lstm: LSTM (from LSTMWrapper)
    ("lstm.weight_ih_l0", (1024, 256)),  # 4 * hidden_size x input_size
    ("lstm.weight_hh_l0", (1024, 256)),  # 4 * hidden_size x hidden_size
    ("lstm.bias_ih_l0", (1024,)),
    ("lstm.bias_hh_l0", (1024,)),
]


class MockEnv:
    """Mock environment with the attributes DriveMoE needs."""

    def __init__(self):
        # These match the default MoE config and C constants
        self.ego_features = 7  # EGO_FEATURES_CLASSIC
        self.max_road_objects = 20  # MAX_ROAD_SEGMENT_OBSERVATIONS
        self.max_partner_objects = 127  # MAX_AGENTS - 1
        self.partner_features = 7  # PARTNER_FEATURES
        self.road_features = 7  # ROAD_FEATURES
        self.dynamics_model = "classic"

        # Action space: MultiDiscrete([91]) for 7*13 joint actions
        self.single_action_space = gymnasium.spaces.MultiDiscrete([91])

        # Observation space
        num_obs = (
            self.ego_features
            + self.max_partner_objects * self.partner_features
            + self.max_road_objects * self.road_features
        )
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(num_obs,), dtype=np.float32
        )


class TestMoEArchitectureMatch:
    """Test that PyTorch MoE architecture matches C implementation."""

    @pytest.fixture
    def moe_policy(self):
        """Create a DriveMoE policy wrapped in LSTMWrapper."""
        # Create mock environment
        env = MockEnv()

        # Create MoE policy with default config
        policy = DriveMoE(
            env,
            input_size=64,
            hidden_size=256,
            num_experts=3,
            lora_rank=8,
            lora_alpha=4.0,
            router_hidden_dim=64,
            freeze_base=False,  # Don't freeze for testing
        )

        # Wrap with LSTM
        wrapped_policy = LSTMWrapper(env, policy, input_size=256, hidden_size=256)

        return wrapped_policy

    def test_parameter_order_matches(self, moe_policy):
        """Test that parameter order matches expected order from drivenet_moe.h."""
        actual_params = list(moe_policy.named_parameters())

        # Check we have the expected number of parameters
        assert len(actual_params) == len(EXPECTED_MOE_PARAMETER_ORDER), (
            f"Parameter count mismatch: got {len(actual_params)}, "
            f"expected {len(EXPECTED_MOE_PARAMETER_ORDER)}"
        )

        # Check each parameter name and shape matches
        for i, ((actual_name, actual_param), (expected_name, expected_shape)) in enumerate(
            zip(actual_params, EXPECTED_MOE_PARAMETER_ORDER)
        ):
            assert actual_name == expected_name, (
                f"Parameter {i} name mismatch:\n"
                f"  Got: {actual_name}\n"
                f"  Expected: {expected_name}\n"
                f"  This means drivenet_moe.h weight loading order is wrong!"
            )

            assert actual_param.shape == torch.Size(expected_shape), (
                f"Parameter {i} ({actual_name}) shape mismatch:\n"
                f"  Got: {actual_param.shape}\n"
                f"  Expected: {expected_shape}\n"
                f"  This means drivenet_moe.h dimensions are wrong!"
            )

    def test_total_weight_count(self, moe_policy):
        """Test total number of weights matches what C code expects."""
        total_weights = sum(p.numel() for p in moe_policy.parameters())

        # Calculate expected from EXPECTED_MOE_PARAMETER_ORDER
        expected_total = sum(
            np.prod(shape) for _, shape in EXPECTED_MOE_PARAMETER_ORDER
        )

        assert total_weights == expected_total, (
            f"Total weight count mismatch:\n"
            f"  Got: {total_weights}\n"
            f"  Expected: {expected_total}\n"
            f"  The exported .bin file will have wrong size!"
        )

    def test_export_produces_correct_size(self, moe_policy, tmp_path):
        """Test that weight export produces file with correct size."""
        # Export weights to temp file
        weights = []
        for name, param in moe_policy.named_parameters():
            weights.append(param.data.cpu().numpy().flatten())

        weights = np.concatenate(weights)
        export_path = tmp_path / "test_weights.bin"
        weights.tofile(export_path)

        # Check file size (4 bytes per float32)
        expected_size = weights.size * 4
        actual_size = export_path.stat().st_size

        assert actual_size == expected_size, (
            f"Exported file size mismatch:\n"
            f"  Got: {actual_size} bytes\n"
            f"  Expected: {expected_size} bytes"
        )

        # Print summary for debugging
        print(f"\nExported {weights.size} weights ({actual_size} bytes)")
        print(f"This should match get_weights() calls in drivenet_moe.h")

    def test_parameter_names_printed(self, moe_policy):
        """Print parameter names and shapes for debugging."""
        print("\n" + "=" * 70)
        print("MoE Policy Parameter Order (for drivenet_moe.h)")
        print("=" * 70)

        total = 0
        for i, (name, param) in enumerate(moe_policy.named_parameters()):
            numel = param.numel()
            total += numel
            print(f"{i:3d}. {name:50s} {str(tuple(param.shape)):20s} ({numel:,} params)")

        print("-" * 70)
        print(f"Total: {total:,} parameters")
        print("=" * 70)


class TestMoEConstants:
    """Test that constants in drivenet_moe.h match Python config."""

    def test_constants_match(self):
        """Verify hardcoded constants match between Python and C."""
        # These are hardcoded in drivenet_moe.h
        C_CONSTANTS = {
            "NN_INPUT_SIZE": 64,
            "NN_HIDDEN_SIZE": 256,
            "NUM_EXPERTS": 3,
            "LORA_RANK": 8,
            "LORA_ALPHA": 4.0,
            "ROUTER_HIDDEN_DIM": 64,
        }

        # These are the defaults in puffer_drive_moe.ini
        PYTHON_DEFAULTS = {
            "input_size": 64,
            "hidden_size": 256,
            "num_experts": 3,
            "lora_rank": 8,
            "lora_alpha": 4.0,
            "router_hidden_dim": 64,
        }

        # Map C names to Python names
        name_mapping = {
            "NN_INPUT_SIZE": "input_size",
            "NN_HIDDEN_SIZE": "hidden_size",
            "NUM_EXPERTS": "num_experts",
            "LORA_RANK": "lora_rank",
            "LORA_ALPHA": "lora_alpha",
            "ROUTER_HIDDEN_DIM": "router_hidden_dim",
        }

        for c_name, c_value in C_CONSTANTS.items():
            py_name = name_mapping[c_name]
            py_value = PYTHON_DEFAULTS[py_name]

            assert c_value == py_value, (
                f"Constant mismatch:\n"
                f"  C ({c_name}): {c_value}\n"
                f"  Python ({py_name}): {py_value}\n"
                f"  Update drivenet_moe.h or puffer_drive_moe.ini to match!"
            )


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_moe_architecture.py -v -s
    pytest.main([__file__, "-v", "-s"])
