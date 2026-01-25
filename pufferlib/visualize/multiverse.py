"""Multiverse visualization for MoE expert comparison.

Visualizes "what-if" scenarios by running the same situation
with different expert assignments (one-hot router weights).
"""

import matplotlib

matplotlib.use("Agg")

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch

from pufferlib.visualize.color import AGENT_COLOR_BY_POLICY
from pufferlib.visualize.utils import (
    img_from_fig,
    save_img_as_png,
    plot_numpy_bounding_boxes,
    plot_trajectory,
    plot_road_polylines,
    compute_viewport,
    save_frames_as_gif,
    save_frames_as_video,
)


class MultiverseVisualizer:
    """Visualizer for comparing MoE expert outcomes.

    Creates side-by-side grid visualizations showing what happens
    when different experts are forced to control the agents.
    """

    def __init__(
        self,
        env,
        policy,
        num_experts: int = 3,
        figsize_per_cell: Tuple[float, float] = (5, 5),
        dpi: int = 100,
    ):
        """Initialize the multiverse visualizer.

        Args:
            env: PufferDrive environment instance.
            policy: MoE policy with router that can be overridden.
            num_experts: Number of experts in the MoE.
            figsize_per_cell: Figure size for each grid cell.
            dpi: Figure resolution.
        """
        self.env = env
        self.policy = policy
        self.num_experts = num_experts
        self.figsize_per_cell = figsize_per_cell
        self.dpi = dpi

        # Cache road data
        self._road_polylines = None

    def _get_road_polylines(self) -> Dict[str, np.ndarray]:
        """Get road polylines with caching."""
        if self._road_polylines is None:
            self._road_polylines = self.env.get_road_edge_polylines()
        return self._road_polylines

    def _create_one_hot_weights(self, expert_idx: int) -> torch.Tensor:
        """Create one-hot expert weights.

        Args:
            expert_idx: Index of the active expert.

        Returns:
            (1, num_experts) tensor with 1.0 at expert_idx, 0.0 elsewhere.
        """
        weights = torch.zeros(1, self.num_experts)
        weights[0, expert_idx] = 1.0
        return weights

    def _run_rollout_with_expert(
        self,
        expert_idx: int,
        initial_obs: np.ndarray,
        initial_state: Optional[Any] = None,
        max_steps: int = 50,
        device: str = "cuda",
    ) -> Dict[str, np.ndarray]:
        """Run a rollout with a specific expert forced.

        Args:
            expert_idx: Index of the expert to use.
            initial_obs: Initial observation.
            initial_state: Optional LSTM state.
            max_steps: Maximum steps to run.
            device: Device for policy inference.

        Returns:
            Dict with 'positions', 'headings', 'collisions', 'offorad' arrays.
        """
        # Create one-hot weights for this expert
        expert_weights = self._create_one_hot_weights(expert_idx).to(device)

        # Store trajectory data
        positions = []
        headings = []
        collisions = []
        offroad = []

        obs = torch.tensor(initial_obs, dtype=torch.float32, device=device)
        state = initial_state

        for step in range(max_steps):
            # Get action from policy with forced expert weights
            with torch.no_grad():
                if hasattr(self.policy, "forward_with_expert_weights"):
                    # Custom method for forcing expert weights
                    action, _, _, state = self.policy.forward_with_expert_weights(obs.unsqueeze(0), expert_weights, state)
                else:
                    # Standard forward - expert weights override happens internally
                    # This requires the policy to have a way to accept forced weights
                    action, _, _, state = self.policy(obs.unsqueeze(0), state)

            action = action.squeeze(0).cpu().numpy()

            # Step environment
            obs_np, reward, done, truncated, info = self.env.step(action)

            # Record state
            agent_states = self.env.get_global_agent_state()
            positions.append(np.column_stack([agent_states["x"], agent_states["y"]]))
            headings.append(agent_states["heading"])

            # Track collisions/offroad from info if available
            if "collision" in info:
                collisions.append(info["collision"])
            if "offroad" in info:
                offroad.append(info["offroad"])

            obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

            if done.all():
                break

        return {
            "positions": np.stack(positions, axis=1) if positions else np.array([]),  # (num_agents, num_steps, 2)
            "headings": np.stack(headings, axis=1) if headings else np.array([]),
            "collisions": np.array(collisions) if collisions else np.array([]),
            "offroad": np.array(offroad) if offroad else np.array([]),
        }

    def plot_multiverse_grid(
        self,
        trajectories_by_expert: Dict[int, Dict[str, np.ndarray]],
        agent_indices: Optional[List[int]] = None,
        center_position: Optional[Tuple[float, float]] = None,
        zoom_radius: float = 50.0,
        title: Optional[str] = None,
        show_legend: bool = True,
        show_current_position: bool = False,
    ) -> np.ndarray:
        """Create a grid visualization comparing expert outcomes.

        Args:
            trajectories_by_expert: Dict mapping expert_idx to trajectory data.
            agent_indices: Optional list of agent indices to highlight.
            center_position: Optional (x, y) to center the view.
            zoom_radius: Radius for viewport.
            title: Overall title for the figure.
            show_legend: If True, show legend.
            show_current_position: If True, draw vehicle boxes at current (last) position.

        Returns:
            [H, W, 3] uint8 numpy array.
        """
        num_experts = len(trajectories_by_expert)

        # Determine grid layout
        if num_experts <= 3:
            nrows, ncols = 1, num_experts
        elif num_experts <= 6:
            nrows, ncols = 2, (num_experts + 1) // 2
        else:
            nrows = int(np.ceil(np.sqrt(num_experts)))
            ncols = int(np.ceil(num_experts / nrows))

        figsize = (self.figsize_per_cell[0] * ncols, self.figsize_per_cell[1] * nrows)
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        gs = GridSpec(nrows, ncols, figure=fig, wspace=0.05, hspace=0.15)

        road_polylines = self._get_road_polylines()

        # Compute viewport
        if center_position is not None:
            x_min, x_max = center_position[0] - zoom_radius, center_position[0] + zoom_radius
            y_min, y_max = center_position[1] - zoom_radius, center_position[1] + zoom_radius
        else:
            x_min, x_max, y_min, y_max = compute_viewport(road_polylines["x"], road_polylines["y"], padding=20.0)

        for idx, (expert_idx, traj_data) in enumerate(sorted(trajectories_by_expert.items())):
            row, col = idx // ncols, idx % ncols
            ax = fig.add_subplot(gs[row, col])
            ax.set_aspect("equal")

            # Plot road
            plot_road_polylines(
                ax,
                road_polylines["x"],
                road_polylines["y"],
                road_polylines["lengths"],
                color="#444444",
                linewidth=0.8,
                alpha=0.6,
            )

            # Plot trajectories
            positions = traj_data["positions"]
            if len(positions) > 0:
                num_agents = positions.shape[0]
                color = AGENT_COLOR_BY_POLICY[expert_idx % len(AGENT_COLOR_BY_POLICY)]

                for agent_idx in range(num_agents):
                    if agent_indices is not None and agent_idx not in agent_indices:
                        continue

                    x = positions[agent_idx, :, 0]
                    y = positions[agent_idx, :, 1]

                    # Skip invalid trajectories
                    if np.all(x == 0) or np.all(np.isnan(x)):
                        continue

                    ax.plot(x, y, "-", color=color, alpha=0.7, linewidth=1.5)

                    # Mark start
                    ax.plot(x[0], y[0], "o", color=color, markersize=4)

                    # Draw vehicle at current position if requested
                    if show_current_position and len(x) > 0:
                        # Get heading at last position
                        headings = traj_data.get("headings", None)
                        if headings is not None and len(headings.shape) > 1:
                            heading = headings[agent_idx, -1]
                        else:
                            # Estimate heading from trajectory
                            if len(x) > 1:
                                heading = np.arctan2(y[-1] - y[-2], x[-1] - x[-2])
                            else:
                                heading = 0

                        # Draw vehicle bounding box (default 4.5m x 2m)
                        length, width = 4.5, 2.0
                        plot_numpy_bounding_boxes(
                            ax,
                            np.array([[x[-1], y[-1], length, width, heading]]),
                            color=color,
                            alpha=0.9,
                            draw_heading=True,
                        )
                    else:
                        # Just mark end point
                        ax.plot(x[-1], y[-1], "s", color=color, markersize=4)

            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_title(f"Expert {expert_idx}", fontsize=10)
            ax.axis("off")

        if title:
            fig.suptitle(title, fontsize=14)

        img = img_from_fig(fig)
        return img

    def plot_multiverse_comparison(
        self,
        initial_obs: np.ndarray,
        initial_state: Optional[Any] = None,
        max_steps: int = 50,
        device: str = "cuda",
        center_agent_idx: Optional[int] = 0,
        zoom_radius: float = 50.0,
        title: Optional[str] = None,
    ) -> np.ndarray:
        """Run rollouts with each expert and create comparison visualization.

        This is a convenience method that:
        1. Runs rollouts with each expert forced
        2. Creates grid visualization of outcomes

        Args:
            initial_obs: Initial observation to start from.
            initial_state: Optional LSTM state.
            max_steps: Maximum steps per rollout.
            device: Device for policy inference.
            center_agent_idx: Agent index to center view on.
            zoom_radius: Viewport radius.
            title: Optional title.

        Returns:
            [H, W, 3] uint8 numpy array.
        """
        trajectories_by_expert = {}

        for expert_idx in range(self.num_experts):
            # Reset environment to same initial state
            self.env.reset()

            # Run rollout with this expert
            traj_data = self._run_rollout_with_expert(
                expert_idx=expert_idx,
                initial_obs=initial_obs,
                initial_state=initial_state,
                max_steps=max_steps,
                device=device,
            )
            trajectories_by_expert[expert_idx] = traj_data

        # Determine center position
        center_position = None
        if center_agent_idx is not None and 0 in trajectories_by_expert:
            positions = trajectories_by_expert[0]["positions"]
            if len(positions) > 0 and center_agent_idx < len(positions):
                # Use initial position of the agent
                center_position = (positions[center_agent_idx, 0, 0], positions[center_agent_idx, 0, 1])

        return self.plot_multiverse_grid(
            trajectories_by_expert=trajectories_by_expert,
            center_position=center_position,
            zoom_radius=zoom_radius,
            title=title or "Multiverse: Expert Comparison",
        )

    def plot_expert_distribution_over_time(
        self,
        expert_probs_history: np.ndarray,
        agent_indices: Optional[List[int]] = None,
        figsize: Tuple[float, float] = (12, 4),
    ) -> np.ndarray:
        """Visualize how expert probabilities change over time.

        Args:
            expert_probs_history: (num_timesteps, num_agents, num_experts) array.
            agent_indices: Optional list of agent indices to show.
            figsize: Figure size.

        Returns:
            [H, W, 3] uint8 numpy array.
        """
        num_timesteps, num_agents, num_experts = expert_probs_history.shape

        if agent_indices is None:
            agent_indices = list(range(min(num_agents, 4)))

        fig, axes = plt.subplots(1, len(agent_indices), figsize=figsize, dpi=self.dpi)
        if len(agent_indices) == 1:
            axes = [axes]

        for ax, agent_idx in zip(axes, agent_indices):
            for expert_idx in range(num_experts):
                color = AGENT_COLOR_BY_POLICY[expert_idx % len(AGENT_COLOR_BY_POLICY)]
                probs = expert_probs_history[:, agent_idx, expert_idx]
                ax.plot(probs, color=color, label=f"Expert {expert_idx}", linewidth=2)

            ax.set_xlabel("Timestep")
            ax.set_ylabel("Probability")
            ax.set_title(f"Agent {agent_idx}")
            ax.set_ylim([0, 1])
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Expert Selection Probabilities Over Time", fontsize=12)
        plt.tight_layout()

        img = img_from_fig(fig)
        return img

    def plot_trajectory_with_expert_coloring(
        self,
        positions: np.ndarray,
        expert_assignments: np.ndarray,
        agent_idx: int = 0,
        show_road: bool = True,
        zoom_radius: float = 50.0,
    ) -> np.ndarray:
        """Plot a single trajectory colored by expert assignment at each timestep.

        Args:
            positions: (num_timesteps, 2) array of positions.
            expert_assignments: (num_timesteps,) array of expert indices.
            agent_idx: Agent index for title.
            show_road: If True, show road network.
            zoom_radius: Viewport radius.

        Returns:
            [H, W, 3] uint8 numpy array.
        """
        fig, ax = plt.subplots(figsize=self.figsize_per_cell, dpi=self.dpi)
        ax.set_aspect("equal")

        if show_road:
            road_polylines = self._get_road_polylines()
            plot_road_polylines(
                ax,
                road_polylines["x"],
                road_polylines["y"],
                road_polylines["lengths"],
                color="#444444",
                linewidth=0.8,
                alpha=0.6,
            )

        # Plot trajectory segments colored by expert
        num_timesteps = len(positions)
        for t in range(num_timesteps - 1):
            x = [positions[t, 0], positions[t + 1, 0]]
            y = [positions[t, 1], positions[t + 1, 1]]
            expert = expert_assignments[t]
            color = AGENT_COLOR_BY_POLICY[expert % len(AGENT_COLOR_BY_POLICY)]
            ax.plot(x, y, "-", color=color, linewidth=2, alpha=0.8)

        # Mark start and end
        ax.plot(positions[0, 0], positions[0, 1], "o", color="green", markersize=8, label="Start")
        ax.plot(positions[-1, 0], positions[-1, 1], "s", color="red", markersize=8, label="End")

        # Viewport
        center_x, center_y = positions.mean(axis=0)
        ax.set_xlim([center_x - zoom_radius, center_x + zoom_radius])
        ax.set_ylim([center_y - zoom_radius, center_y + zoom_radius])

        # Legend for experts
        for expert_idx in range(self.num_experts):
            color = AGENT_COLOR_BY_POLICY[expert_idx % len(AGENT_COLOR_BY_POLICY)]
            ax.plot([], [], "-", color=color, linewidth=2, label=f"Expert {expert_idx}")

        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Agent {agent_idx} Trajectory by Expert")
        ax.axis("off")

        img = img_from_fig(fig)
        return img

    def create_multiverse_video(
        self,
        initial_obs: np.ndarray,
        initial_state: Optional[Any] = None,
        max_steps: int = 50,
        device: str = "cuda",
        save_path: str = "multiverse.gif",
        fps: int = 5,
        zoom_radius: float = 50.0,
    ) -> List[np.ndarray]:
        """Create an animated video showing multiverse outcomes over time.

        Args:
            initial_obs: Initial observation.
            initial_state: Optional LSTM state.
            max_steps: Maximum steps.
            device: Device for inference.
            save_path: Output path for video/gif.
            fps: Frames per second.
            zoom_radius: Viewport radius.

        Returns:
            List of [H, W, 3] frames.
        """
        # First, collect all trajectories
        all_trajectories = {}
        for expert_idx in range(self.num_experts):
            self.env.reset()
            traj_data = self._run_rollout_with_expert(
                expert_idx=expert_idx,
                initial_obs=initial_obs,
                initial_state=initial_state,
                max_steps=max_steps,
                device=device,
            )
            all_trajectories[expert_idx] = traj_data

        # Create frames showing progressive trajectories
        frames = []
        actual_steps = min(max_steps, min(t["positions"].shape[1] for t in all_trajectories.values() if len(t["positions"]) > 0))

        for step in range(1, actual_steps + 1):
            # Create partial trajectories up to this step
            partial_trajectories = {}
            for expert_idx, traj_data in all_trajectories.items():
                partial_trajectories[expert_idx] = {
                    "positions": traj_data["positions"][:, :step, :] if len(traj_data["positions"]) > 0 else np.array([]),
                    "headings": traj_data["headings"][:, :step] if len(traj_data["headings"]) > 0 else np.array([]),
                }

            # Determine center from first expert's first agent
            center_position = None
            if 0 in partial_trajectories and len(partial_trajectories[0]["positions"]) > 0:
                positions = partial_trajectories[0]["positions"]
                center_position = (positions[0, 0, 0], positions[0, 0, 1])

            frame = self.plot_multiverse_grid(
                trajectories_by_expert=partial_trajectories,
                center_position=center_position,
                zoom_radius=zoom_radius,
                title=f"Multiverse Comparison - Step {step}",
            )
            frames.append(frame)

        # Save
        if save_path.endswith(".gif"):
            save_frames_as_gif(frames, save_path, fps=fps)
        else:
            save_frames_as_video(frames, save_path, fps=fps)

        return frames
