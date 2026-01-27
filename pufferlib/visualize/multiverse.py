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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
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
    get_corners_polygon,
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
        self._road_lines = None

    def _get_road_polylines(self) -> Dict[str, np.ndarray]:
        """Get road polylines with caching."""
        if self._road_polylines is None:
            self._road_polylines = self.env.get_road_edge_polylines()
            # Also fetch road lines if available
            if hasattr(self.env, 'get_road_line_polylines'):
                self._road_lines = self.env.get_road_line_polylines()
        return self._road_polylines

    def _get_road_lines(self) -> Optional[Dict[str, np.ndarray]]:
        """Get road lines (lane markings) with caching."""
        if self._road_polylines is None:
            self._get_road_polylines()  # This also fetches road lines
        return self._road_lines

    def _plot_road_lines_dashed(
        self,
        ax,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        lengths: np.ndarray,
        color: str = "#000000",
        linewidth: float = 0.8,
        alpha: float = 0.6,
    ):
        """Plot road lines as dashed lines (lane markings).

        Args:
            ax: Matplotlib axes.
            x_coords: All x coordinates concatenated.
            y_coords: All y coordinates concatenated.
            lengths: Length of each polyline.
            color: Line color.
            linewidth: Line width.
            alpha: Line transparency.
        """
        idx = 0
        for length in lengths:
            if length < 2:
                idx += length
                continue
            x = x_coords[idx : idx + length]
            y = y_coords[idx : idx + length]
            # Draw every other segment for dashed effect
            for i in range(0, len(x) - 1, 2):
                ax.plot(
                    x[i : i + 2],
                    y[i : i + 2],
                    "-",
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                )
            idx += length

    def _plot_roadgraph_3d(
        self,
        ax,
        polylines: Dict[str, np.ndarray],
        color: str = "#444444",
        linewidth: float = 0.8,
        alpha: float = 0.6,
        dashed: bool = False,
    ):
        """Plot road polylines in 3D (on the ground plane z=0).

        Args:
            ax: 3D matplotlib axes.
            polylines: Dict with 'x', 'y', 'lengths' arrays.
            color: Line color.
            linewidth: Line width.
            alpha: Line transparency.
            dashed: If True, draw dashed lines.
        """
        x_coords = polylines["x"]
        y_coords = polylines["y"]
        lengths = polylines["lengths"]

        idx = 0
        for length in lengths:
            if length < 2:
                idx += length
                continue
            x = x_coords[idx : idx + length]
            y = y_coords[idx : idx + length]
            z = np.zeros_like(x)

            if dashed:
                # Draw every other segment
                for i in range(0, len(x) - 1, 2):
                    ax.plot(
                        x[i : i + 2],
                        y[i : i + 2],
                        z[i : i + 2],
                        "-",
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                    )
            else:
                ax.plot(x, y, z, "-", color=color, linewidth=linewidth, alpha=alpha)
            idx += length

    def _plot_vehicle_3d(
        self,
        ax,
        x: float,
        y: float,
        heading: float,
        color: str,
        alpha: float = 0.9,
        length: float = 4.5,
        width: float = 2.0,
        height: float = 1.5,
    ):
        """Plot a vehicle as a 3D box.

        Args:
            ax: 3D matplotlib axes.
            x, y: Vehicle center position.
            heading: Vehicle heading in radians.
            color: Vehicle color.
            alpha: Transparency.
            length, width, height: Vehicle dimensions.
        """
        # Get 2D corners
        corners_2d = get_corners_polygon(x, y, length, width, heading)

        # Create 3D vertices (bottom and top)
        bottom = np.array([[c[0], c[1], 0] for c in corners_2d])
        top = np.array([[c[0], c[1], height] for c in corners_2d])

        # Create faces
        faces = []
        # Bottom face
        faces.append(bottom)
        # Top face
        faces.append(top)
        # Side faces
        for i in range(4):
            j = (i + 1) % 4
            face = np.array([bottom[i], bottom[j], top[j], top[i]])
            faces.append(face)

        # Plot as Poly3DCollection
        collection = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor="black", linewidth=0.5)
        ax.add_collection3d(collection)

        # Draw heading indicator (arrow on top)
        front_center = (top[0] + top[1]) / 2
        back_center = (top[2] + top[3]) / 2
        center_top = (front_center + back_center) / 2
        ax.plot(
            [center_top[0], front_center[0]],
            [center_top[1], front_center[1]],
            [center_top[2] + 0.1, front_center[2] + 0.1],
            "-",
            color="black",
            linewidth=2,
        )

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
        forced_agent_idx: Optional[int] = None,
        pov_agent_idx: Optional[int] = None,
        show_kinematics: bool = True,
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
            forced_agent_idx: If provided, only this agent is colored with the expert color.
                             Other agents are drawn in gray (they follow inferred path).
            pov_agent_idx: If provided, use POV mode centered on this agent with tight zoom
                          and view rotated so agent heading points up.
            show_kinematics: If True, show accel/steer info in panel titles.

        Returns:
            [H, W, 3] uint8 numpy array.
        """
        num_experts = len(trajectories_by_expert)

        # POV mode: tight zoom on specific agent
        if pov_agent_idx is not None:
            zoom_radius = min(zoom_radius, 15.0)  # Very tight zoom for POV
            # Use forced_agent_idx as POV agent if not specified separately
            if forced_agent_idx is None:
                forced_agent_idx = pov_agent_idx

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
        # Use larger hspace to accommodate multi-line titles (kinematics info)
        gs = GridSpec(nrows, ncols, figure=fig, wspace=0.05, hspace=0.25)

        road_polylines = self._get_road_polylines()

        # Compute viewport bounds (used for non-POV mode or as fallback)
        if center_position is not None:
            x_min, x_max = center_position[0] - zoom_radius, center_position[0] + zoom_radius
            y_min, y_max = center_position[1] - zoom_radius, center_position[1] + zoom_radius
        else:
            x_min, x_max, y_min, y_max = compute_viewport(road_polylines["x"], road_polylines["y"], padding=20.0)

        # Get road lines for lane markings
        road_lines = self._get_road_lines()

        for idx, (expert_idx, traj_data) in enumerate(sorted(trajectories_by_expert.items())):
            row, col = idx // ncols, idx % ncols

            # Use 3D subplot for POV mode
            if pov_agent_idx is not None:
                ax = fig.add_subplot(gs[row, col], projection="3d")

                # Get POV agent's heading for this expert's trajectory
                positions = traj_data["positions"]
                headings = traj_data.get("headings", None)
                if headings is not None and pov_agent_idx < len(headings) and len(headings.shape) > 1:
                    pov_heading = headings[pov_agent_idx, -1]
                else:
                    pov_heading = 0

                # Rotate view so vehicle heading points up
                azim = np.degrees(pov_heading) - 180
                ax.view_init(elev=25, azim=azim)

                # Plot road in 3D
                self._plot_roadgraph_3d(ax, road_polylines, color="#444444", linewidth=0.8, alpha=0.6)

                # Plot road lines in 3D (dashed)
                if road_lines is not None and len(road_lines.get("x", [])) > 0:
                    self._plot_roadgraph_3d(ax, road_lines, color="#000000", linewidth=0.5, alpha=0.5, dashed=True)
            else:
                ax = fig.add_subplot(gs[row, col])
                ax.set_aspect("equal")

                # Plot road edges (borders)
                plot_road_polylines(
                    ax,
                    road_polylines["x"],
                    road_polylines["y"],
                    road_polylines["lengths"],
                    color="#444444",
                    linewidth=0.8,
                    alpha=0.6,
                )

                # Plot road lines (lane markings) - dashed black
                if road_lines is not None and len(road_lines.get("x", [])) > 0:
                    self._plot_road_lines_dashed(
                        ax,
                        road_lines["x"],
                        road_lines["y"],
                        road_lines["lengths"],
                        color="#000000",
                        linewidth=0.5,
                        alpha=0.5,
                    )

            # Plot trajectories
            positions = traj_data["positions"]
            if len(positions) > 0:
                num_agents = positions.shape[0]
                expert_color = AGENT_COLOR_BY_POLICY[expert_idx % len(AGENT_COLOR_BY_POLICY)]
                # Neutral gray color for non-forced agents (inferred path)
                inferred_color = "#888888"

                for agent_idx in range(num_agents):
                    if agent_indices is not None and agent_idx not in agent_indices:
                        continue

                    x = positions[agent_idx, :, 0]
                    y = positions[agent_idx, :, 1]

                    # Skip invalid trajectories
                    if np.all(x == 0) or np.all(np.isnan(x)):
                        continue

                    # Determine color based on whether this is the forced agent
                    if forced_agent_idx is not None:
                        if agent_idx == forced_agent_idx:
                            # Forced agent gets expert color, thicker line
                            color = expert_color
                            linewidth = 2.5
                            alpha = 0.95
                            markersize = 6
                        else:
                            # Other agents follow inferred path, shown in gray
                            color = inferred_color
                            linewidth = 1.0
                            alpha = 0.4
                            markersize = 3
                    else:
                        # No forced agent specified - all agents use expert color
                        color = expert_color
                        linewidth = 1.5
                        alpha = 0.7
                        markersize = 4

                    # Plot trajectory (3D or 2D)
                    if pov_agent_idx is not None:
                        # 3D mode - plot on ground plane
                        z = np.zeros_like(x)
                        ax.plot(x, y, z, "-", color=color, alpha=alpha, linewidth=linewidth)
                        ax.scatter([x[0]], [y[0]], [0], color=color, s=markersize * 10, alpha=alpha, marker="o")
                    else:
                        ax.plot(x, y, "-", color=color, alpha=alpha, linewidth=linewidth)
                        ax.plot(x[0], y[0], "o", color=color, markersize=markersize, alpha=alpha)

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
                        veh_length, veh_width = 4.5, 2.0
                        if pov_agent_idx is not None:
                            # 3D vehicle
                            self._plot_vehicle_3d(ax, x[-1], y[-1], heading, color, alpha=alpha)
                        else:
                            plot_numpy_bounding_boxes(
                                ax,
                                np.array([[x[-1], y[-1], veh_length, veh_width, heading]]),
                                color=color,
                                alpha=alpha,
                                draw_heading=True,
                            )
                    else:
                        # Just mark end point
                        if pov_agent_idx is not None:
                            ax.scatter([x[-1]], [y[-1]], [0], color=color, s=markersize * 10, alpha=alpha, marker="s")
                        else:
                            ax.plot(x[-1], y[-1], "s", color=color, markersize=markersize, alpha=alpha)

            # Set viewport - POV mode centers on POV agent's current position per panel
            if pov_agent_idx is not None:
                # 3D POV mode - center on POV agent's current position
                if len(positions) > 0 and pov_agent_idx < len(positions):
                    pov_x = positions[pov_agent_idx, -1, 0]
                    pov_y = positions[pov_agent_idx, -1, 1]
                else:
                    # Fallback to center_position
                    pov_x, pov_y = (x_min + x_max) / 2, (y_min + y_max) / 2

                ax.set_xlim([pov_x - zoom_radius, pov_x + zoom_radius])
                ax.set_ylim([pov_y - zoom_radius, pov_y + zoom_radius])

                # Set z limits for 3D view
                z_max = max(15, zoom_radius * 0.5)
                ax.set_zlim([0, z_max])

                # Set box aspect ratio for isometric-like view
                ax.set_box_aspect([1, 1, 0.4])

                # Completely hide axes for clean look
                ax.set_axis_off()
            else:
                ax.set_xlim([x_min, x_max])
                ax.set_ylim([y_min, y_max])

            # Build panel title with optional kinematic info
            panel_title = f"Expert {expert_idx}"

            # Check if forced agent reached goal
            agent_idx_for_info = forced_agent_idx if forced_agent_idx is not None else 0
            goal_reached = False
            if "goal_reached_step" in traj_data:
                goal_step = traj_data["goal_reached_step"]
                if agent_idx_for_info < len(goal_step):
                    current_step = positions.shape[1] - 1  # Last timestep in current trajectory
                    if goal_step[agent_idx_for_info] >= 0 and goal_step[agent_idx_for_info] <= current_step:
                        goal_reached = True
                        panel_title += "\nGoal Reached!"

            if show_kinematics and "actions" in traj_data and not goal_reached:
                actions = traj_data["actions"]
                if agent_idx_for_info < len(actions) and len(actions[agent_idx_for_info]) > 0:
                    # Get last action and decode to accel/steer
                    # 91 actions = 7 accel * 13 steer
                    last_action = actions[agent_idx_for_info, -1]
                    accel_idx = last_action // 13
                    steer_idx = last_action % 13
                    # Map to human-readable values
                    # Accel: 0-6 maps to roughly -4 to +4 m/s^2
                    # Steer: 0-12 maps to roughly -0.3 to +0.3 rad
                    accel_val = (accel_idx - 3) * 1.33  # Approximate
                    steer_val = (steer_idx - 6) * 0.05  # Approximate
                    panel_title += f"\na={accel_val:+.1f} m/s², δ={steer_val:+.2f} rad"

            # Place title text inside the plot area (top-left corner) to avoid clipping
            # Use axes coordinates (0-1) with transform
            # For 3D axes, use text2D instead of text
            if pov_agent_idx is not None:
                # 3D axes - use text2D for 2D overlay
                ax.text2D(
                    0.02, 0.98, panel_title,
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
                )
            else:
                # 2D axes - use regular text
                ax.text(
                    0.02, 0.98, panel_title,
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
                )
            if pov_agent_idx is None:
                ax.axis("off")

        if title:
            fig.suptitle(title, fontsize=14, y=0.98)

        # Adjust layout - use subplots_adjust instead of tight_layout for better control
        # when using GridSpec with equal aspect axes
        fig.subplots_adjust(top=0.85 if title else 0.90, bottom=0.05, left=0.02, right=0.98)

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
