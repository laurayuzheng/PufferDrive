#!/usr/bin/env python
"""PufferDrive Matplotlib Visualization Script.

Generate images and videos of PufferDrive simulations using matplotlib.
Outputs are saved to ./viz_output/ by default.

Usage:
    # Basic state visualization
    python plt_visualize.py --output viz_output/state.png

    # Generate rollout GIF with random actions
    python plt_visualize.py --output viz_output/rollout.gif --num-steps 50

    # Use trained policy
    python plt_visualize.py --output viz_output/policy.gif --checkpoint experiments/puffer_drive_baseline.pt

    # Ground truth trajectories
    python plt_visualize.py --output viz_output/trajectories.png --mode trajectories

    # 3D rendering
    python plt_visualize.py --output viz_output/state_3d.png --render-3d

    # MoE expert comparison (mock data)
    python plt_visualize.py --output viz_output/multiverse.png --mode multiverse --num-experts 3

    # Run all visualizations at once
    python plt_visualize.py  # Uses default output viz_output/output.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch


def count_unique_agents(env):
    """Count unique agent positions (to detect duplicates from scene overflow)."""
    agent_states = env.get_global_agent_state()
    positions = np.column_stack([agent_states['x'], agent_states['y']])

    unique_indices = []
    for i, pos in enumerate(positions):
        is_unique = True
        for j in unique_indices:
            if np.linalg.norm(pos - positions[j]) < 0.1:
                is_unique = False
                break
        if is_unique:
            unique_indices.append(i)

    return unique_indices


def create_env(args):
    """Create PufferDrive environment."""
    from pufferlib.ocean.drive.drive import Drive

    env = Drive(
        num_agents=args.num_agents,
        map_dir=args.map_dir,
        num_maps=args.num_maps,
        episode_length=args.episode_length,
    )
    env.reset()

    # Check for duplicate agents
    unique_indices = count_unique_agents(env)
    if len(unique_indices) < env.num_agents:
        print(f"WARNING: Scene has only {len(unique_indices)} unique vehicles, but num_agents={env.num_agents}")
        print(f"         Duplicate agents will be spawned at same positions. Consider using --num-agents {len(unique_indices)}")

    return env, unique_indices


def load_policy(env, checkpoint_path, device="cpu"):
    """Load a trained policy from checkpoint.

    Returns:
        tuple: (policy, has_lstm, is_moe, num_experts)
    """
    from pufferlib.ocean.torch import Drive as DrivePolicy, DriveMoE
    from pufferlib.models import LSTMWrapper

    state_dict = torch.load(checkpoint_path, map_location=device)

    # Check if checkpoint has LSTM weights (indicates recurrent policy)
    has_lstm = any("lstm" in k for k in state_dict.keys())

    # Check if checkpoint is MoE (has router weights)
    is_moe = any("router" in k for k in state_dict.keys())

    # Strip 'policy.' prefix to get policy-only weights for architecture inference
    policy_dict = {k.replace("policy.", ""): v for k, v in state_dict.items() if "policy." in k}
    if not policy_dict:
        policy_dict = state_dict

    # Infer input_size from encoder weights (encoder hidden dimension)
    if "ego_encoder.0.weight" in policy_dict:
        input_size = policy_dict["ego_encoder.0.weight"].shape[0]
    else:
        input_size = 64  # Default for baseline checkpoints

    # Infer hidden_size from shared_embedding or actor weights (actor/value hidden dimension)
    if "shared_embedding.1.weight" in policy_dict:
        hidden_size = policy_dict["shared_embedding.1.weight"].shape[0]
    elif "actor.weight" in policy_dict:
        hidden_size = policy_dict["actor.weight"].shape[1]
    else:
        hidden_size = 256  # Default for baseline checkpoints

    # Infer MoE parameters
    num_experts = 3  # default
    lora_rank = 8  # default
    if is_moe:
        if "actor.expert_A" in policy_dict:
            num_experts = policy_dict["actor.expert_A"].shape[0]
            lora_rank = policy_dict["actor.expert_A"].shape[1]

    print(f"Detected input_size={input_size}, hidden_size={hidden_size}, has_lstm={has_lstm}, is_moe={is_moe}")
    if is_moe:
        print(f"         num_experts={num_experts}, lora_rank={lora_rank}")

    # Create base policy
    if is_moe:
        base_policy = DriveMoE(
            env,
            input_size=input_size,
            hidden_size=hidden_size,
            num_experts=num_experts,
            lora_rank=lora_rank,
            freeze_base=False,  # Don't freeze for inference
        )
    else:
        base_policy = DrivePolicy(env, input_size=input_size, hidden_size=hidden_size)

    if has_lstm:
        # Wrap with LSTM and load full checkpoint
        policy = LSTMWrapper(env, base_policy, input_size=hidden_size, hidden_size=hidden_size)
        policy.load_state_dict(state_dict, strict=False)
    else:
        # Load just the policy weights
        base_policy.load_state_dict(policy_dict, strict=False)
        policy = base_policy

    policy.eval()
    return policy, has_lstm, is_moe, num_experts


def visualize_state(args):
    """Render a single simulator state."""
    from pufferlib.visualize import MatplotlibVisualizer, save_img_as_png

    env, unique_indices = create_env(args)
    vis = MatplotlibVisualizer(
        env,
        figsize=(args.figsize, args.figsize),
        dpi=args.dpi,
        render_3d=args.render_3d,
        agent_filter=unique_indices,  # Only render unique agents
    )

    pov_idx = 0 if (args.pov and args.render_3d) else None
    img = vis.plot_simulator_state(timestep=0, center_agent_idx=0, zoom_radius=args.zoom_radius, pov_agent_idx=pov_idx)
    save_img_as_png(img, args.output)
    print(f"Saved state image to {args.output}")

    env.close()


def visualize_trajectories(args):
    """Render ground truth human trajectories."""
    from pufferlib.visualize import MatplotlibVisualizer, save_img_as_png

    env, unique_indices = create_env(args)
    vis = MatplotlibVisualizer(
        env,
        figsize=(args.figsize, args.figsize),
        dpi=args.dpi,
        agent_filter=unique_indices,
    )

    img = vis.plot_ground_truth_trajectories(max_agents=len(unique_indices))
    save_img_as_png(img, args.output)
    print(f"Saved trajectories image to {args.output}")

    env.close()


def visualize_rollout(args):
    """Generate a rollout GIF/video."""
    from pufferlib.visualize import MatplotlibVisualizer
    from pufferlib.visualize.utils import save_frames_as_gif, save_frames_as_video

    env, unique_indices = create_env(args)
    vis = MatplotlibVisualizer(
        env,
        figsize=(args.figsize, args.figsize),
        dpi=args.dpi,
        render_3d=args.render_3d,
        agent_filter=unique_indices,
    )

    # Load policy if provided
    policy = None
    has_lstm = False
    is_moe = False
    num_experts = 0
    lstm_state = None
    if args.checkpoint:
        print(f"Loading policy from {args.checkpoint}")
        policy, has_lstm, is_moe, num_experts = load_policy(env, args.checkpoint)
        if has_lstm:
            # Initialize LSTM state
            lstm_state = {"lstm_h": None, "lstm_c": None}

    obs, *_ = env.reset()
    frames = []
    trajectory_positions = []

    print(f"Generating {args.num_steps} frames...")
    for step in range(args.num_steps):
        expert_assignments = None

        # Get action
        if policy is not None:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                if has_lstm:
                    # Use forward_eval for LSTM policy
                    actions_tuple, value = policy.forward_eval(obs_tensor, lstm_state)
                else:
                    actions_tuple, value = policy(obs_tensor)

                # Get expert assignments for MoE models
                if is_moe:
                    # Access the base policy through the LSTM wrapper
                    base_policy = policy.policy if has_lstm else policy
                    if hasattr(base_policy, "_expert_probs") and base_policy._expert_probs is not None:
                        expert_assignments = base_policy._expert_probs.argmax(dim=-1).numpy()

                # For discrete actions, actions_tuple is a tuple from torch.split
                # Concatenate and get argmax to get discrete action indices
                if isinstance(actions_tuple, tuple):
                    # Each element is logits for one action dimension
                    # For Drive, there's only one action dimension (91 discrete actions)
                    action_logits = actions_tuple[0]  # Shape: [num_agents, 91]
                    action = action_logits.argmax(dim=-1, keepdim=True).numpy()
                else:
                    action = actions_tuple.argmax(dim=-1, keepdim=True).numpy()
        else:
            # Random actions
            action = np.random.randint(0, 91, size=(env.num_agents, 1))

        # Collect positions for trajectory visualization
        agent_states = env.get_global_agent_state()
        positions = np.column_stack([agent_states["x"], agent_states["y"]])
        trajectory_positions.append(positions)

        # Create trajectory array
        traj_array = np.stack(trajectory_positions, axis=1) if len(trajectory_positions) > 1 else None

        # Render frame (center on agent 0 to apply zoom_radius)
        pov_idx = 0 if (args.pov and args.render_3d) else None
        img = vis.plot_simulator_state(
            timestep=step,
            center_agent_idx=0,
            zoom_radius=args.zoom_radius,
            plot_trajectories=args.show_trajectories and traj_array is not None,
            policy_assignments=expert_assignments,
            trajectory_positions=traj_array,
            pov_agent_idx=pov_idx,
        )
        frames.append(img)

        # Step environment
        obs, *_ = env.step(action)

        if step % 10 == 0:
            print(f"  Frame {step}/{args.num_steps}")

    # Save
    if args.output.endswith(".gif"):
        save_frames_as_gif(frames, args.output, fps=args.fps)
    else:
        try:
            save_frames_as_video(frames, args.output, fps=args.fps)
        except ValueError as e:
            if "backend" in str(e).lower():
                # Fallback to GIF
                gif_path = args.output.rsplit(".", 1)[0] + ".gif"
                print(f"ffmpeg not available, saving as GIF instead: {gif_path}")
                save_frames_as_gif(frames, gif_path, fps=args.fps)
            else:
                raise

    print(f"Saved rollout to {args.output}")
    env.close()


def run_rollout_with_forced_expert(env, policy, expert_idx, num_steps, has_lstm=False):
    """Run a rollout with a specific expert forced.

    Args:
        env: PufferDrive environment.
        policy: MoE policy (or LSTM-wrapped MoE policy).
        expert_idx: Index of the expert to force.
        num_steps: Number of steps to run.
        has_lstm: Whether policy uses LSTM wrapper.

    Returns:
        Dict with 'positions', 'headings' arrays (num_agents, num_steps, ...).
    """
    obs, *_ = env.reset()
    lstm_h = None
    lstm_c = None

    positions = []
    headings = []

    for step in range(num_steps):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            # Get the base MoE policy
            base_policy = policy.policy if has_lstm else policy
            batch_size = obs_tensor.shape[0]

            # Encode observations (this sets _expert_probs but we'll override)
            hidden = base_policy.encode_observations(obs_tensor)

            if has_lstm:
                # Pass through LSTM cell
                if lstm_h is not None:
                    lstm_state = (lstm_h, lstm_c)
                else:
                    lstm_state = None
                hidden, lstm_c = policy.cell(hidden, lstm_state)
                lstm_h = hidden

            # Create forced one-hot expert weights
            forced_probs = torch.zeros(batch_size, base_policy.num_experts)
            forced_probs[:, expert_idx] = 1.0

            # Decode with forced expert weights
            actions_tuple, value = base_policy.decode_actions(hidden, expert_probs=forced_probs)

            # Get action from logits
            if isinstance(actions_tuple, tuple):
                action_logits = actions_tuple[0]
                action = action_logits.argmax(dim=-1, keepdim=True).numpy()
            else:
                action = actions_tuple.argmax(dim=-1, keepdim=True).numpy()

        # Record positions before stepping
        agent_states = env.get_global_agent_state()
        positions.append(np.column_stack([agent_states["x"], agent_states["y"]]))
        headings.append(agent_states["heading"].copy())

        # Step environment
        obs, *_ = env.step(action)

    return {
        "positions": np.stack(positions, axis=1),  # (num_agents, num_steps, 2)
        "headings": np.stack(headings, axis=1),  # (num_agents, num_steps)
    }


def visualize_multiverse(args):
    """Generate MoE multiverse comparison visualization.

    Runs actual rollouts with each expert forced to 100% weight,
    showing how different driving styles lead to different trajectories.
    """
    from pufferlib.visualize import MultiverseVisualizer, save_img_as_png

    if not args.checkpoint:
        print("ERROR: --checkpoint is required for multiverse visualization")
        print("       Please provide an MoE checkpoint path")
        return

    env, unique_indices = create_env(args)
    num_unique = len(unique_indices)

    # Load MoE policy
    print(f"Loading MoE policy from {args.checkpoint}")
    policy, has_lstm, is_moe, num_experts = load_policy(env, args.checkpoint)

    if not is_moe:
        print("WARNING: Checkpoint is not an MoE model. Multiverse visualization requires MoE.")
        print("         Falling back to single trajectory visualization.")
        num_experts = 1

    # Override num_experts from checkpoint if detected
    if is_moe:
        args.num_experts = num_experts

    # Create visualizer
    vis = MultiverseVisualizer(
        env,
        policy=policy,
        num_experts=args.num_experts,
        figsize_per_cell=(args.figsize / 2, args.figsize / 2),
        dpi=args.dpi,
    )

    # Run rollouts with each expert forced
    trajectories_by_expert = {}
    num_steps = args.num_steps

    print(f"Running {args.num_experts} expert rollouts ({num_steps} steps each)...")
    for expert_idx in range(args.num_experts):
        print(f"  Expert {expert_idx}...")

        # Reset environment to same initial state for fair comparison
        env.reset()

        if is_moe:
            # Run rollout with forced expert
            traj_data = run_rollout_with_forced_expert(
                env, policy, expert_idx, num_steps, has_lstm
            )
        else:
            # For non-MoE, just run regular rollout
            obs, *_ = env.reset()
            positions = []
            headings = []
            lstm_state = {"lstm_h": None, "lstm_c": None} if has_lstm else None

            for step in range(num_steps):
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32)
                    if has_lstm:
                        actions_tuple, value = policy.forward_eval(obs_tensor, lstm_state)
                    else:
                        actions_tuple, value = policy(obs_tensor)
                    if isinstance(actions_tuple, tuple):
                        action = actions_tuple[0].argmax(dim=-1, keepdim=True).numpy()
                    else:
                        action = actions_tuple.argmax(dim=-1, keepdim=True).numpy()

                agent_states = env.get_global_agent_state()
                positions.append(np.column_stack([agent_states["x"], agent_states["y"]]))
                headings.append(agent_states["heading"].copy())
                obs, *_ = env.step(action)

            traj_data = {
                "positions": np.stack(positions, axis=1),
                "headings": np.stack(headings, axis=1),
            }

        # Filter to unique agents
        trajectories_by_expert[expert_idx] = {
            "positions": traj_data["positions"][unique_indices],
            "headings": traj_data["headings"][unique_indices],
        }

    # Get center position from first agent's initial position
    agent_states = env.get_global_agent_state()
    center_x = agent_states["x"][unique_indices[0]]
    center_y = agent_states["y"][unique_indices[0]]

    # Check if output is a GIF/video (animated) or static image
    is_animated = args.output.endswith((".gif", ".mp4"))

    if is_animated:
        # Generate animated multiverse GIF
        from pufferlib.visualize.utils import save_frames_as_gif, save_frames_as_video

        print(f"Generating multiverse animation ({num_steps} frames)...")
        frames = []

        for step in range(num_steps):
            # Create trajectories up to this step for each expert
            step_trajectories = {}
            for expert_idx, traj_data in trajectories_by_expert.items():
                step_trajectories[expert_idx] = {
                    "positions": traj_data["positions"][:, : step + 1, :],
                    "headings": traj_data["headings"][:, : step + 1],
                }

            img = vis.plot_multiverse_grid(
                trajectories_by_expert=step_trajectories,
                center_position=(center_x, center_y),
                zoom_radius=args.zoom_radius,
                title=f"Multiverse Comparison (t={step})",
                show_current_position=True,
            )
            frames.append(img)

            if step % 10 == 0:
                print(f"  Frame {step}/{num_steps}")

        # Save animation
        if args.output.endswith(".gif"):
            save_frames_as_gif(frames, args.output, fps=args.fps)
        else:
            save_frames_as_video(frames, args.output, fps=args.fps)

        print(f"Saved multiverse animation to {args.output}")
    else:
        # Static image showing final trajectories
        img = vis.plot_multiverse_grid(
            trajectories_by_expert=trajectories_by_expert,
            center_position=(center_x, center_y),
            zoom_radius=args.zoom_radius,
            title=f"Multiverse: {args.num_experts} Expert Comparison",
        )
        save_img_as_png(img, args.output)
        print(f"Saved multiverse image to {args.output}")

    env.close()


def visualize_expert_distribution(args):
    """Visualize expert routing probabilities over time."""
    from pufferlib.visualize import MultiverseVisualizer, save_img_as_png

    # Create mock expert probabilities
    num_timesteps = args.num_steps
    num_agents = min(4, args.num_agents)  # Show up to 4 agents
    num_experts = args.num_experts

    # Generate mock probability curves (would come from actual router outputs)
    probs = np.zeros((num_timesteps, num_agents, num_experts))
    for agent_idx in range(num_agents):
        for expert_idx in range(num_experts):
            # Create smooth probability curves
            phase = agent_idx * 0.5 + expert_idx * 1.0
            probs[:, agent_idx, expert_idx] = np.sin(np.linspace(0, 2 * np.pi + phase, num_timesteps)) + 1.5

    # Normalize to sum to 1
    probs = probs / probs.sum(axis=2, keepdims=True)

    # Create minimal mock env for visualizer
    class MockEnv:
        def get_road_edge_polylines(self):
            return {"x": np.array([]), "y": np.array([]), "lengths": np.array([])}

    vis = MultiverseVisualizer(MockEnv(), policy=None, num_experts=num_experts)

    img = vis.plot_expert_distribution_over_time(
        probs,
        agent_indices=list(range(num_agents)),
    )
    from pufferlib.visualize.utils import save_img_as_png
    save_img_as_png(img, args.output)
    print(f"Saved expert distribution plot to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="PufferDrive Matplotlib Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Output
    parser.add_argument("--output", "-o", type=str, default="viz_output/output.png",
                        help="Output file path (.png, .gif, or .mp4)")

    # Mode
    parser.add_argument("--mode", "-m", type=str, default="auto",
                        choices=["auto", "state", "trajectories", "rollout", "multiverse", "expert-dist"],
                        help="Visualization mode (auto-detects from output extension)")

    # Environment
    parser.add_argument("--num-agents", type=int, default=16,
                        help="Number of agents")
    parser.add_argument("--map-dir", type=str, default="resources/drive/binaries/validation",
                        help="Path to map binaries")
    parser.add_argument("--num-maps", type=int, default=1,
                        help="Number of maps to load")
    parser.add_argument("--episode-length", type=int, default=91,
                        help="Episode length")

    # Visualization
    parser.add_argument("--figsize", type=float, default=10,
                        help="Figure size in inches")
    parser.add_argument("--dpi", type=int, default=100,
                        help="Figure DPI")
    parser.add_argument("--zoom-radius", type=float, default=80.0,
                        help="Viewport zoom radius")
    parser.add_argument("--render-3d", action="store_true",
                        help="Use 3D rendering")
    parser.add_argument("--pov", action="store_true",
                        help="Use first-person POV from ego agent (3D only)")

    # Rollout
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of steps for rollout/animation")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for GIF/video")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to policy checkpoint")
    parser.add_argument("--show-trajectories", action="store_true",
                        help="Show trajectory trails in rollout")

    # MoE
    parser.add_argument("--num-experts", type=int, default=3,
                        help="Number of MoE experts")

    args = parser.parse_args()

    # Auto-detect mode from output
    if args.mode == "auto":
        if args.output.endswith((".gif", ".mp4")):
            args.mode = "rollout"
        else:
            args.mode = "state"

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Run appropriate visualization
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")

    if args.mode == "state":
        visualize_state(args)
    elif args.mode == "trajectories":
        visualize_trajectories(args)
    elif args.mode == "rollout":
        visualize_rollout(args)
    elif args.mode == "multiverse":
        visualize_multiverse(args)
    elif args.mode == "expert-dist":
        visualize_expert_distribution(args)
    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
