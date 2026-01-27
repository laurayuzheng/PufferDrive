#!/usr/bin/env python
"""Test script to verify MoE experts produce different actions."""

import numpy as np
import torch
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.torch import DriveMoE
from pufferlib.models import LSTMWrapper


def main():
    # Create a minimal environment
    print("Creating environment...")
    env = Drive(
        num_agents=4,
        map_dir="resources/drive/binaries/validation",
        num_maps=1,
        episode_length=91,
        goal_behavior=2,  # Stop at goal
    )
    obs, *_ = env.reset()
    print(f"Observation shape: {obs.shape}")

    # Load checkpoint
    checkpoint_path = "experiments/puffer_drive_moe_social_forces.pt"
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        policy_config = checkpoint.get("policy_config", {})
    else:
        state_dict = checkpoint
        policy_config = {}

    # Get model params
    input_size = policy_config.get("input_size", 64)
    hidden_size = policy_config.get("hidden_size", 256)
    num_experts = policy_config.get("num_experts", 3)
    lora_rank = policy_config.get("lora_rank", 8)
    lora_alpha = policy_config.get("lora_alpha", 8.0)

    # Detect router type
    use_social_forces_routing = any("social_force_norm" in k for k in state_dict.keys())
    social_forces_only = use_social_forces_routing and not any("fusion" in k for k in state_dict.keys())

    print(f"Model config: input_size={input_size}, hidden_size={hidden_size}")
    print(f"              num_experts={num_experts}, lora_rank={lora_rank}, lora_alpha={lora_alpha}")
    print(f"              use_social_forces_routing={use_social_forces_routing}, social_forces_only={social_forces_only}")

    # Create policy
    print("\nCreating DriveMoE policy...")
    base_policy = DriveMoE(
        env,
        input_size=input_size,
        hidden_size=hidden_size,
        num_experts=num_experts,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        freeze_base=False,
        use_social_forces_routing=use_social_forces_routing,
        social_forces_only=social_forces_only,
    )

    # Check if LSTM
    has_lstm = any("lstm" in k for k in state_dict.keys())
    print(f"Has LSTM: {has_lstm}")

    if has_lstm:
        policy = LSTMWrapper(env, base_policy, input_size=hidden_size, hidden_size=hidden_size)
    else:
        policy = base_policy

    # Load weights
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    print("Weights loaded successfully.")

    # Test with a single observation
    print("\n" + "="*60)
    print("TEST 1: Single observation, compare experts")
    print("="*60)

    obs_tensor = torch.tensor(obs, dtype=torch.float32)

    # Get base policy
    base_policy = policy.policy if has_lstm else policy

    # Encode observations
    with torch.no_grad():
        hidden = base_policy.encode_observations(obs_tensor)

        if has_lstm:
            hidden, _ = policy.cell(hidden, None)

        # Get router's inferred probs
        inferred_probs = base_policy._expert_probs.clone()
        print(f"\nRouter inferred probs (agent 0): {inferred_probs[0].numpy()}")
        print(f"Router inferred probs (agent 1): {inferred_probs[1].numpy()}")

        # Test each expert
        print("\n--- Forcing each expert and comparing logits ---")
        all_logits = []
        all_actions = []

        for expert_idx in range(num_experts):
            # Force expert
            forced_probs = torch.zeros_like(inferred_probs)
            forced_probs[:, expert_idx] = 1.0

            # Decode with forced expert
            actions_tuple, value = base_policy.decode_actions(hidden, expert_probs=forced_probs)
            logits = actions_tuple[0] if isinstance(actions_tuple, tuple) else actions_tuple

            # Get argmax action
            action = logits.argmax(dim=-1)

            all_logits.append(logits[0].numpy())  # Agent 0's logits
            all_actions.append(action[0].item())  # Agent 0's action

            print(f"\nExpert {expert_idx}:")
            print(f"  Action (agent 0): {action[0].item()}")
            print(f"  Logits (first 10): {logits[0, :10].numpy()}")
            print(f"  Logits mean: {logits[0].mean().item():.4f}, std: {logits[0].std().item():.4f}")
            print(f"  Logits max: {logits[0].max().item():.4f} at idx {logits[0].argmax().item()}")

        # Compare
        print("\n--- Comparison ---")
        print(f"Actions by expert: {all_actions}")
        if len(set(all_actions)) == 1:
            print("WARNING: All experts produce the SAME action!")
        else:
            print("OK: Experts produce different actions.")

        # Check logits difference
        for i in range(num_experts):
            for j in range(i+1, num_experts):
                diff = np.abs(all_logits[i] - all_logits[j]).mean()
                print(f"Mean logit diff between expert {i} and {j}: {diff:.6f}")

    # Test 2: Check LoRA contribution magnitude
    print("\n" + "="*60)
    print("TEST 2: LoRA contribution magnitude")
    print("="*60)

    if hasattr(base_policy, 'actor') and hasattr(base_policy.actor, 'expert_A'):
        # Get the actor layer
        actor = base_policy.actor

        # Compute base output and LoRA output separately
        with torch.no_grad():
            x = hidden[0:1]  # Single sample

            # Base output
            base_out = torch.nn.functional.linear(x, actor.weight, actor.bias)

            # LoRA output for each expert
            print(f"\nLoRA scaling factor: {actor.scaling} (alpha={lora_alpha}, rank={lora_rank})")
            print(f"Base output: mean={base_out.mean().item():.4f}, std={base_out.std().item():.4f}, norm={base_out.norm().item():.4f}")

            for expert_idx in range(num_experts):
                # Get this expert's LoRA weights
                A = actor.expert_A[expert_idx]  # (rank, in_features)
                B = actor.expert_B[expert_idx]  # (out_features, rank)

                # Compute LoRA output: x @ A.T @ B.T
                lora_out = x @ A.T @ B.T * actor.scaling

                print(f"Expert {expert_idx} LoRA: mean={lora_out.mean().item():.4f}, std={lora_out.std().item():.4f}, norm={lora_out.norm().item():.4f}")
                print(f"           LoRA/Base ratio: {lora_out.norm().item() / base_out.norm().item():.4f}")

        print("\n--- LoRA Weight Inspection ---")
        expert_A = base_policy.actor.expert_A.detach().numpy()
        expert_B = base_policy.actor.expert_B.detach().numpy()

        print(f"expert_A shape: {expert_A.shape}")  # (num_experts, lora_rank, in_features)
        print(f"expert_B shape: {expert_B.shape}")  # (num_experts, out_features, lora_rank)

        # Check if experts are different
        for i in range(num_experts):
            for j in range(i+1, num_experts):
                diff_A = np.abs(expert_A[i] - expert_A[j]).mean()
                diff_B = np.abs(expert_B[i] - expert_B[j]).mean()
                print(f"Expert {i} vs {j}: mean |diff_A|={diff_A:.6f}, mean |diff_B|={diff_B:.6f}")

        # Check magnitudes
        for i in range(num_experts):
            print(f"Expert {i} A norm: {np.linalg.norm(expert_A[i]):.4f}, B norm: {np.linalg.norm(expert_B[i]):.4f}")
    else:
        print("Could not find LoRA weights (expert_A, expert_B)")

    # Test 3: Multiple steps
    print("\n" + "="*60)
    print("TEST 3: Multi-step rollout comparison")
    print("="*60)

    num_steps = 10
    obs, *_ = env.reset()

    actions_by_expert = {i: [] for i in range(num_experts)}

    for step in range(num_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            hidden = base_policy.encode_observations(obs_tensor)
            if has_lstm:
                hidden, _ = policy.cell(hidden, None)

            for expert_idx in range(num_experts):
                forced_probs = torch.zeros(obs_tensor.shape[0], num_experts)
                forced_probs[:, expert_idx] = 1.0

                actions_tuple, _ = base_policy.decode_actions(hidden, expert_probs=forced_probs)
                logits = actions_tuple[0] if isinstance(actions_tuple, tuple) else actions_tuple
                action = logits[0].argmax().item()
                actions_by_expert[expert_idx].append(action)

        # Step env with expert 0's action (just to get new obs)
        action_array = np.full((env.num_agents, 1), actions_by_expert[0][-1])
        obs, *_ = env.step(action_array)

    print(f"\nActions over {num_steps} steps (agent 0):")
    for expert_idx in range(num_experts):
        print(f"  Expert {expert_idx}: {actions_by_expert[expert_idx]}")

    # Check if identical
    all_same = all(
        actions_by_expert[0] == actions_by_expert[i]
        for i in range(1, num_experts)
    )
    if all_same:
        print("\nWARNING: All experts produce IDENTICAL action sequences!")
        print("This indicates expert collapse - the LoRA adapters are not differentiating behavior.")
    else:
        print("\nOK: Experts produce different action sequences.")

    # Test 4: Try higher lora_alpha scaling
    print("\n" + "="*60)
    print("TEST 4: Effect of increasing lora_alpha at inference")
    print("="*60)

    obs, *_ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32)

    with torch.no_grad():
        hidden = base_policy.encode_observations(obs_tensor)
        if has_lstm:
            hidden, _ = policy.cell(hidden, None)

        original_scaling = base_policy.actor.scaling

        for test_alpha in [8, 16, 32, 64]:
            test_scaling = test_alpha / lora_rank
            base_policy.actor.scaling = test_scaling

            actions_by_expert = []
            for expert_idx in range(num_experts):
                forced_probs = torch.zeros(obs_tensor.shape[0], num_experts)
                forced_probs[:, expert_idx] = 1.0
                actions_tuple, _ = base_policy.decode_actions(hidden, expert_probs=forced_probs)
                logits = actions_tuple[0] if isinstance(actions_tuple, tuple) else actions_tuple
                action = logits[0].argmax().item()
                actions_by_expert.append(action)

            unique_actions = len(set(actions_by_expert))
            print(f"lora_alpha={test_alpha} (scaling={test_scaling}): actions={actions_by_expert}, unique={unique_actions}")

        # Restore original
        base_policy.actor.scaling = original_scaling

    # Test 5: Check current cosine diversity loss value
    print("\n" + "="*60)
    print("TEST 5: Current expert diversity (cosine loss)")
    print("="*60)

    from pufferlib.ocean.moe_adapters import expert_cosine_loss, expert_output_diversity_loss

    expert_A = base_policy.actor.expert_A
    expert_B = base_policy.actor.expert_B

    cosine_loss = expert_cosine_loss(expert_A, expert_B)
    print(f"Cosine diversity loss: {cosine_loss.item():.6f}")
    print(f"(Lower = more similar experts, Higher = more diverse)")

    # Compute pairwise cosine similarities manually for inspection
    delta_W = torch.einsum("eor,eri->eoi", expert_B, expert_A)  # (K, out, in)
    delta_W_flat = delta_W.view(num_experts, -1)
    normalized = torch.nn.functional.normalize(delta_W_flat, p=2, dim=-1)
    cos_sim = normalized @ normalized.T

    print(f"\nPairwise cosine similarity matrix:")
    print(cos_sim.detach().numpy())
    print(f"\nOff-diagonal mean: {(cos_sim - torch.eye(num_experts)).abs().sum().item() / (num_experts * (num_experts - 1)):.4f}")

    # Test 6: Output diversity loss
    print("\n" + "="*60)
    print("TEST 6: Output diversity loss (new)")
    print("="*60)

    obs, *_ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32)

    with torch.no_grad():
        hidden = base_policy.encode_observations(obs_tensor)
        if has_lstm:
            hidden, _ = policy.cell(hidden, None)

        # Compute output diversity loss
        output_div_loss = expert_output_diversity_loss(
            hidden,
            base_policy.actor,
            base_policy.atn_dim,
        )
        print(f"Output diversity loss: {output_div_loss.item():.6f}")
        print(f"(More negative = more diverse outputs)")

        # Also compute the expert action distributions to visualize
        print("\n--- Expert action probability distributions (first 10 actions) ---")
        base_out = torch.nn.functional.linear(hidden[0:1], base_policy.actor.weight, base_policy.actor.bias)
        lora_intermediate = torch.einsum("bi,eri->ber", hidden[0:1], base_policy.actor.expert_A)
        lora_out = torch.einsum("ber,eor->beo", lora_intermediate, base_policy.actor.expert_B)
        all_expert_outputs = base_out.unsqueeze(1) + lora_out * base_policy.actor.scaling

        first_action_dim = base_policy.atn_dim[0]
        expert_logits = all_expert_outputs[:, :, :first_action_dim]
        expert_probs = torch.nn.functional.softmax(expert_logits, dim=-1)

        for i in range(num_experts):
            probs = expert_probs[0, i, :10].numpy()
            argmax_action = expert_probs[0, i].argmax().item()
            print(f"Expert {i}: probs[:10]={probs}, argmax={argmax_action}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
