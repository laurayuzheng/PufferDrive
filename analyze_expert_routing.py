"""
Analyze expert routing distribution in MoE checkpoint.

Checks if the router assigns any samples to experts 0, 1, 2
(user called them experts 1, 2, 3 using 1-indexing).
"""

import argparse
import ast
import configparser
import os
import sys

import torch
import numpy as np
from collections import defaultdict


def find_config_path(config_name):
    """Find config file by name or path."""
    import glob

    if config_name.endswith(".ini") and os.path.exists(config_name):
        return config_name

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(script_dir, "pufferlib")):
        pufferlib_dir = os.path.join(script_dir, "pufferlib")
    else:
        pufferlib_dir = script_dir

    config_pattern = os.path.join(pufferlib_dir, "config", "**", "*.ini")
    default_config = os.path.join(pufferlib_dir, "config", "default.ini")

    for path in glob.glob(config_pattern, recursive=True):
        if path == default_config:
            continue
        p = configparser.ConfigParser()
        p.read(path)
        if "base" in p and "env_name" in p["base"]:
            if config_name in p["base"]["env_name"].split():
                return path

    raise FileNotFoundError(f"Config '{config_name}' not found.")


def load_ini_config(config_name):
    """Load configuration from an INI file."""
    config_path = find_config_path(config_name)
    print(f"Using config: {config_path}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(script_dir, "pufferlib")):
        pufferlib_dir = os.path.join(script_dir, "pufferlib")
    else:
        pufferlib_dir = os.path.dirname(config_path)

    default_config = os.path.join(pufferlib_dir, "config", "default.ini")

    p = configparser.ConfigParser()
    configs_to_load = [default_config] if os.path.exists(default_config) else []
    configs_to_load.append(config_path)
    p.read(configs_to_load)

    def parse_value(value):
        try:
            return ast.literal_eval(value)
        except:
            return value

    config = {}
    for section in p.sections():
        config[section] = {}
        for key in p[section]:
            config[section][key] = parse_value(p[section][key])

    return config


def analyze_router_distribution(checkpoint_path, num_scenarios=100, num_steps_per_scenario=50):
    """Analyze router expert assignments across many scenarios."""
    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.ocean.torch import DriveMoE
    from pufferlib.models import LSTMWrapper

    # Load config
    config = load_ini_config("puffer_drive_moe")
    env_config = config.get("env", {})
    policy_config = config.get("policy", {})
    rnn_config = config.get("rnn", {})

    # Create environment
    print("\nCreating environment...")
    env = Drive(
        num_agents=10,
        map_dir=env_config.get("map_dir", "pufferlib/resources/drive/train_maps"),
        num_maps=50,  # Load multiple maps for variety
        episode_length=env_config.get("episode_length", 91),
        action_type=env_config.get("action_type", "discrete"),
        dynamics_model=env_config.get("dynamics_model", "classic"),
        dt=env_config.get("dt", 0.1),
    )

    # Create base policy
    print("Creating policy model...")
    input_size = policy_config.get("input_size", 64)
    hidden_size = policy_config.get("hidden_size", 256)

    base_policy = DriveMoE(
        env,
        input_size=input_size,
        hidden_size=hidden_size,
        num_experts=policy_config.get("num_experts", 3),
        lora_rank=policy_config.get("lora_rank", 8),
        lora_alpha=policy_config.get("lora_alpha", 4.0),
        router_hidden_dim=policy_config.get("router_hidden_dim", 64),
        freeze_base=policy_config.get("freeze_base", True),
        use_social_forces_routing=policy_config.get("use_social_forces_routing", False),
        social_forces_only=policy_config.get("social_forces_only", False),
    )

    # Wrap with LSTM
    print("Wrapping with LSTMWrapper...")
    policy = LSTMWrapper(
        env,
        base_policy,
        input_size=hidden_size,  # LSTM input is the policy's hidden_size
        hidden_size=hidden_size,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        state_dict = checkpoint

    # Checkpoint already has correct structure: lstm.*, cell.*, policy.*
    # Load directly without remapping
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"  {k}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"  {k}")
        if len(unexpected) > 5:
            print(f"  ... and {len(unexpected) - 5} more")

    policy.eval()

    # Track expert assignments
    expert_counts = defaultdict(int)
    expert_probs_all = []
    samples_by_expert = defaultdict(list)
    router_logits_all = []

    num_experts = base_policy.num_experts

    print(f"\nAnalyzing {num_scenarios} scenarios with {num_steps_per_scenario} steps each...")
    print(f"Total samples: {num_scenarios * num_steps_per_scenario}")
    print(f"Router type: {'SocialForcesRouter' if base_policy.use_social_forces_routing else 'PersonaRouter'}")

    for scenario_idx in range(num_scenarios):
        obs, _ = env.reset()

        # Initialize LSTM state
        lstm_state = {
            "lstm_h": None,
            "lstm_c": None,
        }

        for step in range(num_steps_per_scenario):
            with torch.no_grad():
                # Get observation for first agent only (obs may be [num_agents, obs_size])
                if len(obs.shape) > 1:
                    single_obs = obs[0]
                else:
                    single_obs = obs
                obs_tensor = torch.tensor(single_obs, dtype=torch.float32).unsqueeze(0)

                # Forward pass through full model (with LSTM)
                actions, _ = policy.forward_eval(obs_tensor, lstm_state)

                # Get expert probs from the inner policy (set during encode_observations)
                expert_probs = base_policy._expert_probs[0].numpy()
                router_logits = base_policy._router_logits[0].numpy()
                expert_assignment = np.argmax(expert_probs)

                expert_counts[expert_assignment] += 1
                expert_probs_all.append(expert_probs)
                router_logits_all.append(router_logits)

                # Store sample info for each expert
                samples_by_expert[expert_assignment].append({
                    'scenario': scenario_idx,
                    'step': step,
                    'probs': expert_probs.copy(),
                    'logits': router_logits.copy(),
                })

                # Simple action selection for single agent, then tile for all agents
                if isinstance(actions, tuple):
                    single_action = np.array([a.argmax().item() for a in actions])
                else:
                    single_action = actions[0].numpy()

                # Tile action for all agents in the environment
                action = np.tile(single_action, (env.num_agents, 1))

                obs, _, done, truncated, _ = env.step(action)

                # Reset if any agent is done (for multi-agent env, done is an array)
                if np.any(done) or np.any(truncated):
                    obs, _ = env.reset()
                    # Reset LSTM state on episode boundary
                    lstm_state = {
                        "lstm_h": None,
                        "lstm_c": None,
                    }

        if (scenario_idx + 1) % 20 == 0:
            print(f"  Progress: {scenario_idx + 1}/{num_scenarios} scenarios")

    # Print results
    total_samples = sum(expert_counts.values())
    print("\n" + "="*60)
    print("EXPERT ASSIGNMENT DISTRIBUTION")
    print("="*60)

    for expert_idx in range(num_experts):
        count = expert_counts[expert_idx]
        pct = 100.0 * count / total_samples if total_samples > 0 else 0
        print(f"Expert {expert_idx}: {count:6d} samples ({pct:5.1f}%)")

    # Analyze expert probabilities
    expert_probs_all = np.array(expert_probs_all)
    router_logits_all = np.array(router_logits_all)

    print("\n" + "="*60)
    print("EXPERT PROBABILITY STATISTICS")
    print("="*60)

    for expert_idx in range(num_experts):
        probs = expert_probs_all[:, expert_idx]
        print(f"Expert {expert_idx}: mean={probs.mean():.4f}, std={probs.std():.4f}, "
              f"min={probs.min():.4f}, max={probs.max():.4f}")

    print("\n" + "="*60)
    print("ROUTER LOGITS STATISTICS (before softmax)")
    print("="*60)

    for expert_idx in range(num_experts):
        logits = router_logits_all[:, expert_idx]
        print(f"Expert {expert_idx}: mean={logits.mean():.4f}, std={logits.std():.4f}, "
              f"min={logits.min():.4f}, max={logits.max():.4f}")

    # Check for router collapse
    print("\n" + "="*60)
    print("ROUTER COLLAPSE CHECK")
    print("="*60)

    entropy = -np.sum(expert_probs_all * np.log(expert_probs_all + 1e-10), axis=1)
    print(f"Mean routing entropy: {entropy.mean():.4f} (max possible: {np.log(num_experts):.4f})")
    print(f"Entropy std: {entropy.std():.4f}")

    if entropy.mean() < 0.1:
        print("WARNING: Very low entropy indicates router collapse (always same expert)")
    elif entropy.mean() < 0.5:
        print("NOTE: Relatively low entropy - router has strong preferences")
    else:
        print("OK: Router shows good diversity in expert selection")

    # Show some examples for underrepresented experts
    print("\n" + "="*60)
    print("SAMPLE EXAMPLES BY EXPERT")
    print("="*60)

    for expert_idx in range(num_experts):
        samples = samples_by_expert[expert_idx]
        print(f"\nExpert {expert_idx}: {len(samples)} samples")
        if samples:
            # Show first 3 examples
            for i, sample in enumerate(samples[:3]):
                print(f"  Example {i+1}: scenario={sample['scenario']}, step={sample['step']}")
                print(f"    probs={sample['probs']}")
                print(f"    logits={sample['logits']}")
        else:
            print("  NO SAMPLES - Router never selects this expert!")

    # Analyze router weights
    print("\n" + "="*60)
    print("ROUTER WEIGHT ANALYSIS")
    print("="*60)

    if hasattr(base_policy.router, 'classifier'):
        # Get the final linear layer
        final_layer = None
        for module in base_policy.router.classifier.modules():
            if isinstance(module, torch.nn.Linear):
                final_layer = module

        if final_layer is not None:
            weights = final_layer.weight.data.numpy()
            biases = final_layer.bias.data.numpy()

            print("Final classifier layer:")
            print(f"  Weight shape: {weights.shape}")
            print(f"  Weight norm per expert: {np.linalg.norm(weights, axis=1)}")
            print(f"  Biases: {biases}")

            # Check if biases heavily favor one expert
            bias_softmax = np.exp(biases - biases.max()) / np.exp(biases - biases.max()).sum()
            print(f"  Bias softmax (baseline routing): {bias_softmax}")

            if max(bias_softmax) > 0.9:
                print("  WARNING: Bias strongly favors one expert - may indicate mode collapse")

    env.close()
    return expert_counts, expert_probs_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str,
                       default="experiments/puffer_drive_moe_932h6awx.pt",
                       help="Path to MoE checkpoint")
    parser.add_argument("--num-scenarios", "-n", type=int, default=200,
                       help="Number of scenarios to test")
    parser.add_argument("--num-steps", "-s", type=int, default=30,
                       help="Steps per scenario")

    args = parser.parse_args()

    analyze_router_distribution(
        args.checkpoint,
        num_scenarios=args.num_scenarios,
        num_steps_per_scenario=args.num_steps,
    )
