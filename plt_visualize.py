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

    # Load a specific scenario/map by index
    python plt_visualize.py --output viz_output/scenario5.gif --mode multiverse --scenario 5 --checkpoint experiments/puffer_drive_moe.pt

    # Use INI config file (ensures env/policy params match training)
    python plt_visualize.py --mode multiverse --config puffer_drive_moe --checkpoint experiments/puffer_drive_moe.pt --output viz_output/moe.gif

    # Run all visualizations at once
    python plt_visualize.py  # Uses default output viz_output/output.png
"""

import argparse
import ast
import configparser
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch


def find_config_path(config_name):
    """Find config file by name or path.

    Args:
        config_name: Either a full path to .ini file, or an env_name like 'puffer_drive_moe'

    Returns:
        str: Full path to the config file

    Raises:
        FileNotFoundError: If config cannot be found
    """
    import glob

    # If it's already a full path, return it
    if config_name.endswith(".ini") and os.path.exists(config_name):
        return config_name

    # Find pufferlib directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Check if we're in project root or inside pufferlib
    if os.path.exists(os.path.join(script_dir, "pufferlib")):
        pufferlib_dir = os.path.join(script_dir, "pufferlib")
    elif "pufferlib" in script_dir:
        pufferlib_dir = script_dir
        while os.path.basename(pufferlib_dir) != "pufferlib" and pufferlib_dir != "/":
            pufferlib_dir = os.path.dirname(pufferlib_dir)
    else:
        pufferlib_dir = script_dir

    # Search for config files matching the env_name
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

    raise FileNotFoundError(
        f"Config '{config_name}' not found. Provide either:\n"
        f"  - Full path: pufferlib/config/ocean/puffer_drive_moe.ini\n"
        f"  - Env name: puffer_drive_moe"
    )


def load_ini_config(config_name):
    """Load configuration from an INI file.

    Args:
        config_name: Either a full path to .ini file, or an env_name like 'puffer_drive_moe'

    Returns:
        dict: Nested dict with sections as keys (env, policy, train, etc.)
    """
    config_path = find_config_path(config_name)
    print(f"Using config: {config_path}")

    # Find pufferlib directory for default config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(script_dir, "pufferlib")):
        pufferlib_dir = os.path.join(script_dir, "pufferlib")
    else:
        pufferlib_dir = os.path.dirname(config_path)
        while "config" in pufferlib_dir:
            pufferlib_dir = os.path.dirname(pufferlib_dir)

    default_config = os.path.join(pufferlib_dir, "config", "default.ini")

    p = configparser.ConfigParser()
    # Load default first, then overlay with specific config
    configs_to_load = [default_config] if os.path.exists(default_config) else []
    configs_to_load.append(config_path)
    p.read(configs_to_load)

    def parse_value(value):
        try:
            return ast.literal_eval(value)
        except:
            return value

    # Convert to nested dict
    config = {}
    for section in p.sections():
        config[section] = {}
        for key in p[section]:
            config[section][key] = parse_value(p[section][key])

    return config


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


def create_env(args, config=None):
    """Create PufferDrive environment.

    If --scenario is specified, creates a temp directory with a symlink to the specific map.
    If config is provided, uses config values as defaults (CLI args override).
    """
    import tempfile
    from pathlib import Path
    from pufferlib.ocean.drive.drive import Drive

    # Get env params from config (for params that must match training)
    env_config = config.get("env", {}) if config else {}

    # Visualization-specific params (hardcoded or from CLI)
    num_agents = 10  # Scenarios are small, no need for more
    map_dir = args.map_dir
    num_maps = args.num_maps

    # These params come from config if available (must match training)
    episode_length = env_config.get("episode_length", args.episode_length)
    action_type = env_config.get("action_type", "discrete")
    dynamics_model = env_config.get("dynamics_model", "classic")
    dt = env_config.get("dt", 0.1)

    temp_dir = None

    # If scenario is specified, create a temp directory with just that map
    if hasattr(args, 'scenario') and args.scenario is not None:
        scenario_idx = args.scenario
        map_file = Path(map_dir) / f"map_{scenario_idx:03d}.bin"
        if not map_file.exists():
            raise FileNotFoundError(f"Scenario {scenario_idx} not found: {map_file}")

        # Create temp directory with symlink to the specific map
        temp_dir = tempfile.mkdtemp(prefix="pufferdrive_viz_")
        temp_map = Path(temp_dir) / "map_000.bin"
        temp_map.symlink_to(map_file.resolve())
        map_dir = temp_dir
        num_maps = 1
        print(f"Loading scenario {scenario_idx} from {map_file}")

    print(f"Creating env: num_agents={num_agents}, map_dir={map_dir}, episode_length={episode_length}")
    print(f"              action_type={action_type}, dynamics_model={dynamics_model}, dt={dt}")

    env = Drive(
        num_agents=num_agents,
        map_dir=map_dir,
        num_maps=num_maps,
        episode_length=episode_length,
        action_type=action_type,
        dynamics_model=dynamics_model,
        dt=dt,
        goal_behavior=2,  # Stop at goal instead of respawn (for visualization)
    )
    env.reset()

    # Store temp_dir reference for cleanup
    env._temp_viz_dir = temp_dir
    # Store config for later use
    env._viz_config = config

    # Check for duplicate agents
    unique_indices = count_unique_agents(env)
    if len(unique_indices) < env.num_agents:
        print(f"WARNING: Scene has only {len(unique_indices)} unique vehicles, but num_agents={env.num_agents}")
        print(f"         Duplicate agents will be spawned at same positions.")

    return env, unique_indices


def cleanup_env(env):
    """Clean up environment and any temp directories."""
    import shutil
    env.close()
    if hasattr(env, '_temp_viz_dir') and env._temp_viz_dir is not None:
        shutil.rmtree(env._temp_viz_dir, ignore_errors=True)


def load_policy(env, checkpoint_path, device="cpu", lora_alpha=None, config=None):
    """Load a trained policy from checkpoint.

    Args:
        env: PufferDrive environment.
        checkpoint_path: Path to checkpoint file.
        device: Device to load weights to.
        lora_alpha: LoRA scaling factor. If None, auto-detect from checkpoint
                   (new format) or use config or default of 4.0.
        config: Optional INI config dict. If provided, uses [policy] and [rnn] sections.

    Returns:
        tuple: (policy, has_lstm, is_moe, num_experts)
    """
    from pufferlib.ocean.torch import Drive as DrivePolicy, DriveMoE
    from pufferlib.ocean.torch_mtr import DriveMTRMoE
    from pufferlib.models import LSTMWrapper

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config sections if provided
    ini_policy_config = config.get("policy", {}) if config else {}
    ini_rnn_config = config.get("rnn", {}) if config else {}
    ini_base_config = config.get("base", {}) if config else {}

    # Check if new checkpoint format (contains policy_config)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # New format: extract config and state_dict
        state_dict = checkpoint["state_dict"]
        ckpt_policy_config = checkpoint.get("policy_config", {})
        rnn_name = checkpoint.get("rnn_name", None)
        has_lstm = rnn_name is not None
    else:
        # Old format: state_dict is the checkpoint itself
        state_dict = checkpoint
        ckpt_policy_config = {}
        # Check if checkpoint has LSTM weights (indicates recurrent policy)
        has_lstm = any("lstm" in k for k in state_dict.keys())

    # Priority: CLI arg > INI config > checkpoint config > default
    # For has_lstm: check INI rnn_name first
    if ini_base_config.get("rnn_name") is not None:
        has_lstm = True
        print(f"Using LSTM from INI config (rnn_name={ini_base_config.get('rnn_name')})")

    # Check if checkpoint is MoE (has router weights)
    is_moe = any("router" in k for k in state_dict.keys())
    # Also check INI policy_name
    if ini_base_config.get("policy_name") == "DriveMoE":
        is_moe = True

    # Check if checkpoint is MTR-MoE (has mtr.encoder weights)
    is_mtr_moe = any("mtr.encoder" in k or "mtr_arch.encoder" in k for k in state_dict.keys())
    # Also check INI policy_name
    if ini_base_config.get("policy_name") == "DriveMTRMoE":
        is_mtr_moe = True
    if is_mtr_moe:
        is_moe = True  # MTR-MoE is also MoE

    # Strip 'policy.' prefix to get policy-only weights for architecture inference
    policy_dict = {k.replace("policy.", ""): v for k, v in state_dict.items() if "policy." in k}
    if not policy_dict:
        policy_dict = state_dict

    # Infer input_size: INI > checkpoint weights > checkpoint config > default
    if "input_size" in ini_policy_config:
        input_size = ini_policy_config["input_size"]
    elif "ego_encoder.0.weight" in policy_dict:
        input_size = policy_dict["ego_encoder.0.weight"].shape[0]
    else:
        input_size = ckpt_policy_config.get("input_size", 64)

    # Infer hidden_size: INI > checkpoint weights > checkpoint config > default
    if "hidden_size" in ini_policy_config:
        hidden_size = ini_policy_config["hidden_size"]
    elif "shared_embedding.1.weight" in policy_dict:
        hidden_size = policy_dict["shared_embedding.1.weight"].shape[0]
    elif "actor.weight" in policy_dict:
        hidden_size = policy_dict["actor.weight"].shape[1]
    else:
        hidden_size = ckpt_policy_config.get("hidden_size", 256)

    # Infer MoE parameters: INI > checkpoint weights > checkpoint config > default
    num_experts = ini_policy_config.get("num_experts", ckpt_policy_config.get("num_experts", 3))
    lora_rank = ini_policy_config.get("lora_rank", ckpt_policy_config.get("lora_rank", 8))
    if is_moe and "actor.expert_A" in policy_dict:
        # Override from weights if available
        num_experts = policy_dict["actor.expert_A"].shape[0]
        lora_rank = policy_dict["actor.expert_A"].shape[1]

    # lora_alpha priority: CLI arg > INI > checkpoint > default
    if lora_alpha is None:
        lora_alpha = ini_policy_config.get("lora_alpha", ckpt_policy_config.get("lora_alpha", 4.0))
    print(f"Using lora_alpha={lora_alpha}")

    # Detect router type: INI > checkpoint keys
    use_social_forces_routing = ini_policy_config.get("use_social_forces_routing", False)
    social_forces_only = ini_policy_config.get("social_forces_only", False)
    # Also check checkpoint keys
    if any(k.startswith("router.social_force_norm") for k in policy_dict.keys()):
        use_social_forces_routing = True
        social_forces_only = not any(k.startswith("router.fusion") for k in policy_dict.keys())

    print(f"Policy config: input_size={input_size}, hidden_size={hidden_size}, has_lstm={has_lstm}, is_moe={is_moe}, is_mtr_moe={is_mtr_moe}")
    if is_moe:
        print(f"               num_experts={num_experts}, lora_rank={lora_rank}, lora_alpha={lora_alpha}")
        if use_social_forces_routing and not is_mtr_moe:
            print(f"               use_social_forces_routing=True, social_forces_only={social_forces_only}")

    # Create base policy
    if is_mtr_moe:
        # MTR-MoE policy - get additional parameters from config
        d_model = ini_policy_config.get("d_model", ckpt_policy_config.get("d_model", 256))
        nhead = ini_policy_config.get("nhead", ckpt_policy_config.get("nhead", 8))
        num_encoder_layers = ini_policy_config.get("num_encoder_layers", ckpt_policy_config.get("num_encoder_layers", 6))
        num_decoder_layers = ini_policy_config.get("num_decoder_layers", ckpt_policy_config.get("num_decoder_layers", 6))
        num_queries = ini_policy_config.get("num_queries", ckpt_policy_config.get("num_queries", 6))
        dropout = ini_policy_config.get("dropout", ckpt_policy_config.get("dropout", 0.1))
        router_hidden_dim = ini_policy_config.get("router_hidden_dim", ckpt_policy_config.get("router_hidden_dim", 128))
        print(f"               d_model={d_model}, nhead={nhead}, num_encoder_layers={num_encoder_layers}")
        print(f"               num_decoder_layers={num_decoder_layers}, num_queries={num_queries}")
        base_policy = DriveMTRMoE(
            env,
            d_model=d_model,
            hidden_size=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_queries=num_queries,
            dropout=dropout,
            num_experts=num_experts,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            router_hidden_dim=router_hidden_dim,
            freeze_base=False,  # Don't freeze for inference
        )
    elif is_moe:
        base_policy = DriveMoE(
            env,
            input_size=input_size,
            hidden_size=hidden_size,
            num_experts=num_experts,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_base=False,  # Don't freeze for inference
            use_social_forces_routing=use_social_forces_routing,
            social_forces_only=social_forces_only,
        )
    else:
        base_policy = DrivePolicy(env, input_size=input_size, hidden_size=hidden_size)

    if has_lstm:
        policy = LSTMWrapper(env, base_policy, input_size=hidden_size, hidden_size=hidden_size)
    else:
        policy = base_policy

    # Remove training-only keys before strict loading
    keys_to_remove = [k for k in state_dict.keys() if "expert_prior" in k]
    for k in keys_to_remove:
        del state_dict[k]
    policy.load_state_dict(state_dict, strict=True)

    policy.eval()
    return policy, has_lstm, is_moe, num_experts


def visualize_state(args):
    """Render a single simulator state."""
    from pufferlib.visualize import MatplotlibVisualizer, save_img_as_png

    env, unique_indices = create_env(args, config=args._config)
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

    cleanup_env(env)


def visualize_trajectories(args):
    """Render ground truth human trajectories."""
    from pufferlib.visualize import MatplotlibVisualizer, save_img_as_png

    env, unique_indices = create_env(args, config=args._config)
    vis = MatplotlibVisualizer(
        env,
        figsize=(args.figsize, args.figsize),
        dpi=args.dpi,
        agent_filter=unique_indices,
    )

    img = vis.plot_ground_truth_trajectories(max_agents=len(unique_indices))
    save_img_as_png(img, args.output)
    print(f"Saved trajectories image to {args.output}")

    cleanup_env(env)


def visualize_rollout(args):
    """Generate a rollout GIF/video."""
    from pufferlib.visualize import MatplotlibVisualizer
    from pufferlib.visualize.utils import save_frames_as_gif, save_frames_as_video

    env, unique_indices = create_env(args, config=args._config)
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
        policy, has_lstm, is_moe, num_experts = load_policy(env, args.checkpoint, lora_alpha=args.lora_alpha, config=args._config)
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

                # For discrete actions, use sampling (like evaluation) instead of argmax
                import pufferlib.pytorch
                action, _, _ = pufferlib.pytorch.sample_logits(actions_tuple)
                action = action.numpy()
                if action.ndim == 1:
                    action = action[:, np.newaxis]
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
    cleanup_env(env)


def run_rollout_with_forced_expert(env, policy, expert_idx, num_steps, has_lstm=False, forced_agent_idx=None, debug=False, initial_obs=None):
    """Run a rollout with a specific expert forced for one or all agents.

    Args:
        env: PufferDrive environment.
        policy: MoE policy (or LSTM-wrapped MoE policy).
        expert_idx: Index of the expert to force.
        num_steps: Number of steps to run.
        has_lstm: Whether policy uses LSTM wrapper.
        forced_agent_idx: If provided, only force this agent to use the expert.
                         Other agents use their inferred router weights.
                         If None, all agents are forced to use the expert.
        debug: If True, print debug information about expert forcing.
        initial_obs: If provided, use this as initial observation instead of resetting.
                    The caller is responsible for resetting the environment.

    Returns:
        Dict with 'positions', 'headings', 'actions', 'goal_reached_step' arrays.
    """
    if initial_obs is not None:
        obs = initial_obs
    else:
        obs, *_ = env.reset()
    lstm_h = None
    lstm_c = None

    num_agents = env.num_agents
    positions = []
    headings = []
    actions_list = []
    goal_reached_step = np.full(num_agents, -1, dtype=np.int32)  # -1 means not reached
    frozen_positions = None  # Store positions when goal is reached
    frozen_headings = None

    for step in range(num_steps):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            # Get the base MoE policy
            base_policy = policy.policy if has_lstm else policy
            batch_size = obs_tensor.shape[0]

            # Encode observations (this computes and stores _expert_probs from router)
            hidden = base_policy.encode_observations(obs_tensor)

            if has_lstm:
                # Pass through LSTM cell
                if lstm_h is not None:
                    lstm_state = (lstm_h, lstm_c)
                else:
                    lstm_state = None
                hidden, lstm_c = policy.cell(hidden, lstm_state)
                lstm_h = hidden

            # Get the inferred expert probs from router
            inferred_probs = base_policy._expert_probs.clone()

            if forced_agent_idx is not None:
                # Only override the specified agent's expert probs
                mixed_probs = inferred_probs.clone()
                mixed_probs[forced_agent_idx] = 0.0
                mixed_probs[forced_agent_idx, expert_idx] = 1.0
            else:
                # Force all agents to use the specified expert
                mixed_probs = torch.zeros(batch_size, base_policy.num_experts)
                mixed_probs[:, expert_idx] = 1.0

            # Debug: print expert probs being used
            if debug and step == 0:
                print(f"    [Debug] Expert {expert_idx}, step {step}")
                print(f"    [Debug] Inferred probs (agent {forced_agent_idx}): {inferred_probs[forced_agent_idx].numpy()}")
                print(f"    [Debug] Forced probs (agent {forced_agent_idx}): {mixed_probs[forced_agent_idx].numpy()}")

            # Decode with mixed expert weights
            actions_tuple, value = base_policy.decode_actions(hidden, expert_probs=mixed_probs)

            # Get action from logits using sampling (like evaluation) instead of argmax
            import pufferlib.pytorch
            action, _, _ = pufferlib.pytorch.sample_logits(actions_tuple)
            action = action.numpy()
            if action.ndim == 1:
                action = action[:, np.newaxis]

        # Record positions before stepping
        agent_states = env.get_global_agent_state()
        current_positions = np.column_stack([agent_states["x"], agent_states["y"]])
        current_headings = agent_states["heading"].copy()

        # For agents that have reached goal, freeze their position
        if frozen_positions is not None:
            for i in range(num_agents):
                if goal_reached_step[i] >= 0:
                    current_positions[i] = frozen_positions[i]
                    current_headings[i] = frozen_headings[i]

        positions.append(current_positions)
        headings.append(current_headings)
        actions_list.append(action.flatten().copy())

        # Step environment
        obs, rewards, terminals, truncations, info = env.step(action)

        # Check for goal reached (terminal state)
        for i in range(num_agents):
            if goal_reached_step[i] < 0 and terminals[i]:
                goal_reached_step[i] = step
                if frozen_positions is None:
                    frozen_positions = current_positions.copy()
                    frozen_headings = current_headings.copy()
                else:
                    frozen_positions[i] = current_positions[i]
                    frozen_headings[i] = current_headings[i]

    return {
        "positions": np.stack(positions, axis=1),  # (num_agents, num_steps, 2)
        "headings": np.stack(headings, axis=1),  # (num_agents, num_steps)
        "actions": np.stack(actions_list, axis=1),  # (num_agents, num_steps)
        "goal_reached_step": goal_reached_step,  # (num_agents,) -1 if not reached
    }


def visualize_multiverse(args):
    """Generate MoE multiverse comparison visualization.

    Runs actual rollouts with each expert forced for a single agent,
    showing how different driving styles lead to different trajectories.
    Other agents follow their inferred router path.
    """
    from pufferlib.visualize import MultiverseVisualizer, save_img_as_png

    if not args.checkpoint:
        print("ERROR: --checkpoint is required for multiverse visualization")
        print("       Please provide an MoE checkpoint path")
        return

    env, unique_indices = create_env(args, config=args._config)
    num_unique = len(unique_indices)

    # Load MoE policy
    print(f"Loading MoE policy from {args.checkpoint}")
    policy, has_lstm, is_moe, num_experts = load_policy(env, args.checkpoint, lora_alpha=args.lora_alpha, config=args._config)

    if not is_moe:
        print("WARNING: Checkpoint is not an MoE model. Multiverse visualization requires MoE.")
        print("         Falling back to single trajectory visualization.")
        num_experts = 1

    # Override num_experts from checkpoint if detected
    if is_moe:
        args.num_experts = num_experts

    # Determine which agent to force (default to first unique agent)
    forced_agent_idx = getattr(args, 'forced_agent', 0)
    if forced_agent_idx is None:
        forced_agent_idx = 0
    # Clamp to valid range of unique agents
    if forced_agent_idx >= len(unique_indices):
        print(f"WARNING: --forced-agent {forced_agent_idx} >= num unique agents ({len(unique_indices)}), clamping to {len(unique_indices) - 1}")
        forced_agent_idx = len(unique_indices) - 1
    # Map to actual agent index in the full env array
    actual_forced_idx = unique_indices[forced_agent_idx]

    print(f"Forcing expert on agent {forced_agent_idx} (env idx: {actual_forced_idx})")
    print(f"Other agents will follow their inferred router path")

    # Create visualizer
    vis = MultiverseVisualizer(
        env,
        policy=policy,
        num_experts=args.num_experts,
        figsize_per_cell=(args.figsize / 2, args.figsize / 2),
        dpi=args.dpi,
    )

    # Run rollouts with each expert forced on the selected agent only
    trajectories_by_expert = {}
    num_steps = args.num_steps

    # Set random seed for reproducible initial state
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get initial observation from first reset - we'll restore this for each expert
    initial_obs, *_ = env.reset(seed=seed)

    # Store initial agent states to verify consistency
    initial_agent_states = env.get_global_agent_state()
    initial_positions = np.column_stack([initial_agent_states["x"], initial_agent_states["y"]])
    print(f"Initial position of forced agent {forced_agent_idx}: {initial_positions[actual_forced_idx]}")

    print(f"Running {args.num_experts} expert rollouts ({num_steps} steps each)...")
    all_actions_by_expert = {}  # For diagnostic comparison
    for expert_idx in range(args.num_experts):
        print(f"  Expert {expert_idx} (forced on agent {forced_agent_idx})...")

        # Reset environment to SAME initial state for fair comparison
        np.random.seed(seed)
        torch.manual_seed(seed)
        initial_obs, *_ = env.reset(seed=seed)

        # Verify we got the same initial state
        check_states = env.get_global_agent_state()
        check_pos = np.column_stack([check_states["x"], check_states["y"]])
        if not np.allclose(check_pos, initial_positions):
            print(f"  WARNING: Initial positions differ! Env reset is not deterministic.")

        if is_moe:
            # Run rollout with forced expert on single agent only
            traj_data = run_rollout_with_forced_expert(
                env, policy, expert_idx, num_steps, has_lstm,
                forced_agent_idx=actual_forced_idx,  # Only force this agent
                debug=True,  # Enable debug output
                initial_obs=initial_obs,  # Use the reset observation
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
        # Include actions if available
        if "actions" in traj_data:
            trajectories_by_expert[expert_idx]["actions"] = traj_data["actions"][unique_indices]
        # Include goal_reached_step if available
        if "goal_reached_step" in traj_data:
            trajectories_by_expert[expert_idx]["goal_reached_step"] = traj_data["goal_reached_step"][unique_indices]

    # Get center position from forced agent's INITIAL position (not final)
    center_x = initial_positions[actual_forced_idx, 0]
    center_y = initial_positions[actual_forced_idx, 1]

    # Diagnostic: Compare actions across experts for the forced agent
    print(f"\n=== Expert Action Comparison (agent {forced_agent_idx}) ===")
    for expert_idx, traj_data in trajectories_by_expert.items():
        if "actions" in traj_data:
            actions = traj_data["actions"][forced_agent_idx]
            positions = traj_data["positions"][forced_agent_idx]
            # Show first 10 actions and final position
            action_str = ", ".join([str(a) for a in actions[:10]])
            final_pos = positions[-1] if len(positions) > 0 else [0, 0]
            print(f"  Expert {expert_idx}: actions=[{action_str}...], final_pos=({final_pos[0]:.1f}, {final_pos[1]:.1f})")

    # Check if all experts produce identical actions
    if len(trajectories_by_expert) > 1:
        first_actions = trajectories_by_expert[0]["actions"][forced_agent_idx] if "actions" in trajectories_by_expert[0] else None
        all_same = True
        for expert_idx in range(1, len(trajectories_by_expert)):
            if "actions" in trajectories_by_expert[expert_idx]:
                other_actions = trajectories_by_expert[expert_idx]["actions"][forced_agent_idx]
                if not np.array_equal(first_actions, other_actions):
                    all_same = False
                    break
        if all_same:
            print("  WARNING: All experts produced IDENTICAL actions! Expert collapse detected.")
        else:
            print("  OK: Experts produced different actions.")
    print()

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
                if "actions" in traj_data:
                    step_trajectories[expert_idx]["actions"] = traj_data["actions"][:, : step + 1]
                if "goal_reached_step" in traj_data:
                    step_trajectories[expert_idx]["goal_reached_step"] = traj_data["goal_reached_step"]

            # POV mode for multiverse
            pov_idx = forced_agent_idx if args.pov else None

            img = vis.plot_multiverse_grid(
                trajectories_by_expert=step_trajectories,
                center_position=(center_x, center_y),
                zoom_radius=args.zoom_radius,
                title=None,  # No suptitle to avoid covering content
                show_current_position=True,
                forced_agent_idx=forced_agent_idx,  # Highlight the forced agent
                pov_agent_idx=pov_idx,  # POV mode centers on forced agent
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
        pov_idx = forced_agent_idx if args.pov else None

        img = vis.plot_multiverse_grid(
            trajectories_by_expert=trajectories_by_expert,
            center_position=(center_x, center_y),
            zoom_radius=args.zoom_radius,
            title=None,  # No suptitle to avoid covering content
            forced_agent_idx=forced_agent_idx,  # Highlight the forced agent
            pov_agent_idx=pov_idx,  # POV mode centers on forced agent
        )
        save_img_as_png(img, args.output)
        print(f"Saved multiverse image to {args.output}")

    cleanup_env(env)


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


def visualize_multiverse_batch(args, start_scenario, end_scenario):
    """Generate multiverse visualizations for multiple scenarios.

    Args:
        args: Parsed arguments
        start_scenario: First scenario index (inclusive)
        end_scenario: Last scenario index (inclusive)
    """
    from pathlib import Path

    # Determine output directory and extension
    output_dir = Path("viz_output/multiverse")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output extension from args.output or default to .gif
    if args.output.endswith(".mp4"):
        ext = ".mp4"
    elif args.output.endswith(".png"):
        ext = ".png"
    else:
        ext = ".gif"

    print(f"\n{'='*60}")
    print(f"BATCH MULTIVERSE VISUALIZATION")
    print(f"Scenarios: {start_scenario} to {end_scenario}")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {ext}")
    print(f"{'='*60}\n")

    # Track results
    successful = []
    failed = []
    skipped = []

    for scenario_idx in range(start_scenario, end_scenario + 1):
        output_path = output_dir / f"scenario_{scenario_idx:03d}{ext}"

        # Check if map file exists
        map_file = Path(args.map_dir) / f"map_{scenario_idx:03d}.bin"
        if not map_file.exists():
            print(f"[{scenario_idx:03d}] SKIP - Map file not found: {map_file}")
            skipped.append(scenario_idx)
            continue

        print(f"\n[{scenario_idx:03d}] Processing scenario {scenario_idx}...")

        # Create a copy of args with the specific scenario and output
        import copy
        scenario_args = copy.copy(args)
        scenario_args.scenario = scenario_idx
        scenario_args.output = str(output_path)

        try:
            visualize_multiverse(scenario_args)
            successful.append(scenario_idx)
            print(f"[{scenario_idx:03d}] SUCCESS - Saved to {output_path}")
        except Exception as e:
            failed.append((scenario_idx, str(e)))
            print(f"[{scenario_idx:03d}] FAILED - {e}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped (no map file): {len(skipped)}")

    if failed:
        print(f"\nFailed scenarios:")
        for scenario_idx, error in failed:
            print(f"  [{scenario_idx:03d}] {error[:80]}...")

    print(f"\nOutput directory: {output_dir}")


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
    parser.add_argument("--forced-agent", type=int, default=0,
                        help="Agent index to force expert on (others follow inferred path)")
    parser.add_argument("--lora-alpha", type=float, default=None,
                        help="LoRA alpha scaling factor. Auto-detected from new checkpoints, defaults to 4.0 for old ones.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--scenario", type=int, default=None,
                        help="Specific scenario/map index to load (0-based). Overrides --num-maps to 1.")
    parser.add_argument("--batch-scenarios", type=str, default=None,
                        help="Generate visualizations for multiple scenarios. Format: 'START-END' (e.g., '0-100'). "
                             "Output files are saved to viz_output/multiverse/scenario_XXX.gif")
    parser.add_argument("--config", type=str, default=None,
                        help="Config name or path (e.g., 'puffer_drive_moe' or full path to .ini). "
                             "Uses config values for env and policy parameters.")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        try:
            args._config = load_ini_config(args.config)
        except FileNotFoundError as e:
            parser.error(str(e))
    else:
        args._config = None

    # Handle batch scenarios mode
    if args.batch_scenarios:
        # Parse range format "START-END"
        try:
            parts = args.batch_scenarios.split("-")
            if len(parts) == 2:
                start_scenario = int(parts[0])
                end_scenario = int(parts[1])
            else:
                parser.error(f"Invalid --batch-scenarios format. Use 'START-END' (e.g., '0-100')")
        except ValueError:
            parser.error(f"Invalid --batch-scenarios format. Use 'START-END' (e.g., '0-100')")

        if start_scenario > end_scenario:
            parser.error(f"Start scenario ({start_scenario}) must be <= end scenario ({end_scenario})")

        # Force multiverse mode for batch
        args.mode = "multiverse"
        print(f"Using config: {args.config}")
        visualize_multiverse_batch(args, start_scenario, end_scenario)
        return

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