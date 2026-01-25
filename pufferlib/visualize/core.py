"""Core visualization class for PufferLib environments.

Provides matplotlib-based visualization of driving simulation states.
Adapted from GPUDrive visualization utilities.
"""

import matplotlib

matplotlib.use("Agg")

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from pufferlib.visualize.color import (
    AGENT_COLOR_BY_STATE,
    AGENT_COLOR_BY_POLICY,
    ROAD_GRAPH_COLORS,
)
from pufferlib.visualize.utils import (
    img_from_fig,
    save_img_as_png,
    plot_numpy_bounding_boxes,
    plot_trajectory,
    plot_road_polylines,
    plot_goals,
    compute_viewport,
    get_corners_polygon,
)


class MatplotlibVisualizer:
    """Matplotlib-based visualizer for PufferDrive environments.

    Provides methods to visualize:
    - Road network (polylines, edges)
    - Agent bounding boxes with heading
    - Trajectories with temporal coloring
    - Observation features
    - 2D and 3D rendering modes
    """

    def __init__(
        self,
        env,
        goal_radius: float = 2.0,
        figsize: Tuple[float, float] = (10, 10),
        dpi: int = 100,
        render_3d: bool = False,
        vehicle_height: float = 1.5,
        agent_filter: Optional[List[int]] = None,
    ):
        """Initialize the visualizer.

        Args:
            env: PufferDrive environment instance with data access methods.
            goal_radius: Radius for goal visualization.
            figsize: Default figure size.
            dpi: Figure resolution.
            render_3d: If True, use 3D rendering.
            vehicle_height: Height of vehicles for 3D rendering.
            agent_filter: Optional list of agent indices to render (for filtering duplicates).
        """
        self.env = env
        self.goal_radius = goal_radius
        self.figsize = figsize
        self.dpi = dpi
        self.render_3d = render_3d
        self.vehicle_height = vehicle_height
        self.agent_filter = agent_filter

        # Cache static data
        self._road_polylines = None
        self._road_lines = None
        self._road_lanes = None
        self._cached_scenario_id = None

    def _get_road_polylines(self, scenario_id: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Get road polylines, with caching.

        Args:
            scenario_id: Optional scenario ID to filter by.

        Returns:
            Dict with 'x', 'y', 'lengths', 'scenario_id' arrays.
        """
        if self._road_polylines is None:
            self._road_polylines = self.env.get_road_edge_polylines()
            # Also fetch road lines and lanes if available
            if hasattr(self.env, 'get_road_line_polylines'):
                self._road_lines = self.env.get_road_line_polylines()
            if hasattr(self.env, 'get_road_lane_polylines'):
                self._road_lanes = self.env.get_road_lane_polylines()

        if scenario_id is not None:
            # Filter by scenario
            mask = self._road_polylines["scenario_id"] == scenario_id
            filtered = {
                "lengths": self._road_polylines["lengths"][mask],
                "scenario_id": self._road_polylines["scenario_id"][mask],
            }

            # Need to also filter x, y based on lengths
            idx = 0
            x_filtered, y_filtered = [], []
            for i, length in enumerate(self._road_polylines["lengths"]):
                if mask[i]:
                    x_filtered.extend(self._road_polylines["x"][idx : idx + length])
                    y_filtered.extend(self._road_polylines["y"][idx : idx + length])
                idx += length

            filtered["x"] = np.array(x_filtered, dtype=np.float32)
            filtered["y"] = np.array(y_filtered, dtype=np.float32)
            return filtered

        return self._road_polylines

    def plot_simulator_state(
        self,
        env_idx: int = 0,
        timestep: Optional[int] = None,
        center_agent_idx: Optional[int] = None,
        zoom_radius: float = 100.0,
        plot_trajectories: bool = False,
        trajectory_positions: Optional[np.ndarray] = None,
        agent_states: Optional[Dict[str, np.ndarray]] = None,
        collision_mask: Optional[np.ndarray] = None,
        offroad_mask: Optional[np.ndarray] = None,
        goal_reached_mask: Optional[np.ndarray] = None,
        policy_assignments: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        pov_agent_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Plot the current simulator state.

        Args:
            env_idx: Environment index (for multi-env setups).
            timestep: Current timestep for title display.
            center_agent_idx: If provided, center view on this agent.
            zoom_radius: Radius for zoomed view around center agent.
            plot_trajectories: If True, plot agent trajectories.
            trajectory_positions: Optional (num_agents, num_timesteps, 2) array of positions.
            agent_states: Optional pre-fetched agent states dict.
            collision_mask: Boolean mask for collided agents.
            offroad_mask: Boolean mask for off-road agents.
            goal_reached_mask: Boolean mask for agents that reached goal.
            policy_assignments: Integer array of policy/expert assignments per agent.
            title: Optional title for the plot.
            pov_agent_idx: If provided (3D only), use first-person POV from this agent.

        Returns:
            [H, W, 3] uint8 numpy array of the rendered image.
        """
        # Get data
        if agent_states is None:
            agent_states = self.env.get_global_agent_state()

        road_polylines = self._get_road_polylines()

        # Auto-reduce zoom for POV mode (simulate being in the car)
        if pov_agent_idx is not None:
            zoom_radius = min(zoom_radius, 15.0)  # Very tight zoom for POV
            # Auto-center on POV agent
            center_agent_idx = pov_agent_idx

        # Create figure
        if self.render_3d:
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            ax = fig.add_subplot(111, projection="3d")

            if pov_agent_idx is not None:
                # POV mode: camera follows vehicle, heading always points up
                agent_heading = agent_states["heading"][pov_agent_idx]

                # Rotate view so vehicle heading points up
                azim = np.degrees(agent_heading) - 180

                # Low elevation for forward-looking view
                ax.view_init(elev=25, azim=azim)
            else:
                # Isometric-style view: 30 degrees elevation, 225 degrees azimuth
                ax.view_init(elev=30, azim=225)
        else:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.set_aspect("equal")

        # Plot road
        self._plot_roadgraph(ax, road_polylines)

        # Plot agents
        self._plot_agent_bounding_boxes(
            ax,
            agent_states,
            collision_mask=collision_mask,
            offroad_mask=offroad_mask,
            goal_reached_mask=goal_reached_mask,
            policy_assignments=policy_assignments,
            pov_agent_idx=pov_agent_idx,
        )

        # Plot trajectories if provided
        if plot_trajectories and trajectory_positions is not None:
            self._plot_trajectories(ax, trajectory_positions, policy_assignments)

        # Set viewport
        if center_agent_idx is not None:
            center_x = agent_states["x"][center_agent_idx]
            center_y = agent_states["y"][center_agent_idx]
            x_min, x_max, y_min, y_max = compute_viewport(
                np.array([center_x]),
                np.array([center_y]),
                center=(center_x, center_y),
                radius=zoom_radius,
            )
        else:
            x_min, x_max, y_min, y_max = compute_viewport(
                road_polylines["x"],
                road_polylines["y"],
                padding=20.0,
            )

        if self.render_3d:
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

            # Scale z-axis to make vehicles visible
            scene_size = max(x_max - x_min, y_max - y_min)
            # Store scene size for vehicle height scaling
            self._3d_scene_size = scene_size
            # Z range should be proportional to make vehicles visible
            # Typical vehicle is ~5m long, height ~2.5m (50%), so z_max = 10-15m for close-up
            z_max = max(15, scene_size * 0.1)
            ax.set_zlim([0, z_max])

            # Set box aspect ratio for isometric-like view
            # Exaggerate z for better 3D visibility
            ax.set_box_aspect([1, 1, 0.4])

            # Remove axis labels and ticks for cleaner look
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # Hide all 3D axis elements for clean look
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("none")
            ax.yaxis.pane.set_edgecolor("none")
            ax.zaxis.pane.set_edgecolor("none")
            ax.xaxis.line.set_color("none")
            ax.yaxis.line.set_color("none")
            ax.zaxis.line.set_color("none")
            ax.grid(False)
        else:
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.axis("off")

        # Title
        if title:
            ax.set_title(title)
        elif timestep is not None:
            ax.set_title(f"Timestep: {timestep}")

        # Convert to image
        img = img_from_fig(fig)
        return img

    def _plot_roadgraph(
        self,
        ax,
        road_polylines: Dict[str, np.ndarray],
        color: str = "#444444",
        linewidth: float = 1.0,
        alpha: float = 0.8,
    ) -> None:
        """Plot road network including edges, lines, and lanes.

        Args:
            ax: Matplotlib axes.
            road_polylines: Dict with 'x', 'y', 'lengths' arrays (road edges).
            color: Line color for edges.
            linewidth: Line width.
            alpha: Transparency.
        """
        if self.render_3d:
            self._plot_roadgraph_3d(ax, road_polylines, color, linewidth, alpha)
            # Plot road lines (lane markings) in black, dashed
            if self._road_lines is not None and len(self._road_lines.get("x", [])) > 0:
                self._plot_roadgraph_3d(ax, self._road_lines, "#000000", linewidth * 0.6, alpha * 0.8, dashed=True)
        else:
            plot_road_polylines(
                ax,
                road_polylines["x"],
                road_polylines["y"],
                road_polylines["lengths"],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )
            # Plot road lines (lane markings) in black, dashed
            if self._road_lines is not None and len(self._road_lines.get("x", [])) > 0:
                self._plot_road_lines_dashed(
                    ax,
                    self._road_lines["x"],
                    self._road_lines["y"],
                    self._road_lines["lengths"],
                    color="#000000",
                    linewidth=linewidth * 0.6,
                    alpha=alpha * 0.8,
                )

    def _plot_roadgraph_3d(
        self,
        ax,
        road_polylines: Dict[str, np.ndarray],
        color: str = "#444444",
        linewidth: float = 1.0,
        alpha: float = 0.8,
        dashed: bool = False,
    ) -> None:
        """Plot road network in 3D.

        Args:
            ax: 3D Matplotlib axes.
            road_polylines: Dict with 'x', 'y', 'lengths' arrays.
            color: Line color.
            linewidth: Line width.
            alpha: Transparency.
            dashed: If True, draw dashed lines.
        """
        segments = []
        idx = 0

        for length in road_polylines["lengths"]:
            if length < 2:
                idx += length
                continue

            x = road_polylines["x"][idx : idx + length]
            y = road_polylines["y"][idx : idx + length]
            z = np.zeros_like(x)  # Road at z=0

            if dashed:
                # Draw every other segment for dashed effect
                for i in range(0, len(x) - 1, 2):
                    segments.append([(x[i], y[i], z[i]), (x[i + 1], y[i + 1], z[i + 1])])
            else:
                for i in range(len(x) - 1):
                    segments.append([(x[i], y[i], z[i]), (x[i + 1], y[i + 1], z[i + 1])])

            idx += length

        if segments:
            lc = Line3DCollection(segments, colors=color, linewidths=linewidth, alpha=alpha)
            ax.add_collection3d(lc)

    def _plot_road_lines_dashed(
        self,
        ax,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        lengths: np.ndarray,
        color: str = "#000000",
        linewidth: float = 1.0,
        alpha: float = 0.8,
    ) -> None:
        """Plot road lines as dashed lines in 2D.

        Args:
            ax: Matplotlib axes.
            x_coords, y_coords: Flattened point coordinates.
            lengths: Number of points per polyline.
            color: Line color.
            linewidth: Line width.
            alpha: Transparency.
        """
        segments = []
        idx = 0

        for length in lengths:
            if length < 2:
                idx += length
                continue

            x = x_coords[idx : idx + length]
            y = y_coords[idx : idx + length]

            # Draw every other segment for dashed effect
            for i in range(0, len(x) - 1, 2):
                segments.append([(x[i], y[i]), (x[i + 1], y[i + 1])])

            idx += length

        if segments:
            lc = LineCollection(segments, colors=color, linewidths=linewidth, alpha=alpha)
            ax.add_collection(lc)

    def _plot_agent_bounding_boxes(
        self,
        ax,
        agent_states: Dict[str, np.ndarray],
        collision_mask: Optional[np.ndarray] = None,
        offroad_mask: Optional[np.ndarray] = None,
        goal_reached_mask: Optional[np.ndarray] = None,
        policy_assignments: Optional[np.ndarray] = None,
        pov_agent_idx: Optional[int] = None,
    ) -> None:
        """Plot agent bounding boxes with state-based coloring.

        Args:
            ax: Matplotlib axes.
            agent_states: Dict with 'x', 'y', 'heading', 'length', 'width' arrays.
            collision_mask: Boolean mask for collided agents.
            offroad_mask: Boolean mask for off-road agents.
            goal_reached_mask: Boolean mask for agents that reached goal.
            policy_assignments: Integer array of policy/expert assignments.
            pov_agent_idx: If provided, highlight this agent with a special color.
        """
        # Apply agent filter to remove duplicates
        # Also need to map pov_agent_idx to the filtered index
        filtered_pov_idx = None
        if self.agent_filter is not None:
            idx = self.agent_filter
            agent_states = {k: v[idx] for k, v in agent_states.items()}
            if collision_mask is not None:
                collision_mask = collision_mask[idx]
            if offroad_mask is not None:
                offroad_mask = offroad_mask[idx]
            if goal_reached_mask is not None:
                goal_reached_mask = goal_reached_mask[idx]
            if policy_assignments is not None:
                policy_assignments = policy_assignments[idx]
            # Map pov_agent_idx to filtered index
            if pov_agent_idx is not None:
                try:
                    filtered_pov_idx = list(idx).index(pov_agent_idx)
                except ValueError:
                    filtered_pov_idx = None  # POV agent was filtered out
        else:
            filtered_pov_idx = pov_agent_idx

        num_agents = len(agent_states["x"])

        # Build bboxes array: (N, 5) with [x, y, length, width, yaw]
        bboxes = np.column_stack(
            [
                agent_states["x"],
                agent_states["y"],
                agent_states["length"],
                agent_states["width"],
                agent_states["heading"],
            ]
        )

        # Special color for POV agent (bright gold/yellow)
        POV_AGENT_COLOR = "#FFD700"  # Gold

        # Determine colors based on state or policy
        if policy_assignments is not None:
            # Color by policy assignment
            for policy_idx in range(int(policy_assignments.max()) + 1):
                mask = policy_assignments == policy_idx
                if not mask.any():
                    continue

                color = AGENT_COLOR_BY_POLICY[policy_idx % len(AGENT_COLOR_BY_POLICY)]

                if self.render_3d:
                    self._plot_bboxes_3d(ax, bboxes[mask], color)
                else:
                    plot_numpy_bounding_boxes(
                        ax,
                        bboxes[mask],
                        color=color,
                        label=f"Expert {policy_idx}",
                        draw_heading=True,
                    )

            # Draw POV agent on top with special color
            if filtered_pov_idx is not None and filtered_pov_idx < num_agents:
                if self.render_3d:
                    self._plot_bboxes_3d(ax, bboxes[filtered_pov_idx : filtered_pov_idx + 1], POV_AGENT_COLOR)
                else:
                    plot_numpy_bounding_boxes(
                        ax,
                        bboxes[filtered_pov_idx : filtered_pov_idx + 1],
                        color=POV_AGENT_COLOR,
                        label="POV Agent",
                        draw_heading=True,
                    )
        else:
            # Color by state
            colors = np.array([AGENT_COLOR_BY_STATE["ok"]] * num_agents)

            if collision_mask is not None:
                colors[collision_mask] = AGENT_COLOR_BY_STATE["collided"]
            if offroad_mask is not None:
                colors[offroad_mask] = AGENT_COLOR_BY_STATE["off_road"]
            if goal_reached_mask is not None:
                colors[goal_reached_mask] = AGENT_COLOR_BY_STATE["goal_reached"]

            # Group by color and plot
            unique_colors = np.unique(colors)
            for color in unique_colors:
                mask = colors == color
                state_name = [k for k, v in AGENT_COLOR_BY_STATE.items() if v == color]
                label = state_name[0] if state_name else None

                if self.render_3d:
                    self._plot_bboxes_3d(ax, bboxes[mask], color)
                else:
                    plot_numpy_bounding_boxes(ax, bboxes[mask], color=color, label=label, draw_heading=True)

            # Draw POV agent on top with special color
            if filtered_pov_idx is not None and filtered_pov_idx < num_agents:
                if self.render_3d:
                    self._plot_bboxes_3d(ax, bboxes[filtered_pov_idx : filtered_pov_idx + 1], POV_AGENT_COLOR)
                else:
                    plot_numpy_bounding_boxes(
                        ax,
                        bboxes[filtered_pov_idx : filtered_pov_idx + 1],
                        color=POV_AGENT_COLOR,
                        label="POV Agent",
                        draw_heading=True,
                    )

    def _plot_bboxes_3d(
        self,
        ax,
        bboxes: np.ndarray,
        color: str,
        alpha: float = 0.8,
        height_scale: float = 1.0,
    ) -> None:
        """Plot bounding boxes in 3D as cuboids.

        Args:
            ax: 3D Matplotlib axes.
            bboxes: (N, 5) array with [x, y, length, width, yaw].
            color: Color for the boxes.
            alpha: Transparency.
            height_scale: Scale factor for vehicle height.
        """
        for bbox in bboxes:
            x, y, length, width, yaw = bbox
            corners_2d = get_corners_polygon(x, y, length, width, yaw)

            # Create bottom and top faces
            # Height = 40% of vehicle length for realistic proportions
            z_bottom = 0
            z_top = max(length * 0.4, 1.5) * height_scale

            # Bottom face vertices
            bottom = [(c[0], c[1], z_bottom) for c in corners_2d]
            # Top face vertices
            top = [(c[0], c[1], z_top) for c in corners_2d]

            # Create faces
            faces = [
                bottom,  # Bottom
                top,  # Top
                [bottom[0], bottom[1], top[1], top[0]],  # Front
                [bottom[1], bottom[2], top[2], top[1]],  # Right
                [bottom[2], bottom[3], top[3], top[2]],  # Back
                [bottom[3], bottom[0], top[0], top[3]],  # Left
            ]

            # Use darker edge color for better visibility
            collection = Poly3DCollection(
                faces,
                alpha=alpha,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
            )
            ax.add_collection3d(collection)

    def _plot_trajectories(
        self,
        ax,
        trajectory_positions: np.ndarray,
        policy_assignments: Optional[np.ndarray] = None,
        alpha: float = 0.6,
    ) -> None:
        """Plot agent trajectories with temporal coloring.

        Args:
            ax: Matplotlib axes.
            trajectory_positions: (num_agents, num_timesteps, 2) array.
            policy_assignments: Optional policy assignments for coloring.
            alpha: Transparency.
        """
        # Apply agent filter to remove duplicates
        if self.agent_filter is not None:
            trajectory_positions = trajectory_positions[self.agent_filter]
            if policy_assignments is not None:
                policy_assignments = policy_assignments[self.agent_filter]

        num_agents = trajectory_positions.shape[0]

        for i in range(num_agents):
            x = trajectory_positions[i, :, 0]
            y = trajectory_positions[i, :, 1]

            # Skip if no valid positions
            if np.all(np.isnan(x)) or np.all(x == 0):
                continue

            if policy_assignments is not None:
                color = AGENT_COLOR_BY_POLICY[policy_assignments[i] % len(AGENT_COLOR_BY_POLICY)]
                ax.plot(x, y, "-", color=color, alpha=alpha, linewidth=1.5)
            else:
                plot_trajectory(ax, x, y, cmap="viridis", alpha=alpha)

    def plot_agent_observation(
        self,
        agent_idx: int,
        obs: np.ndarray,
        ego_features: int = 7,
        partner_features: int = 7,
        road_features: int = 4,
        max_partners: int = 63,
        max_road_segments: int = 100,
        obs_radius: float = 50.0,
    ) -> np.ndarray:
        """Visualize the observation from a single agent's perspective.

        Args:
            agent_idx: Index of the agent to visualize.
            obs: Raw observation array for this agent.
            ego_features: Number of ego features.
            partner_features: Number of features per partner.
            road_features: Number of features per road segment.
            max_partners: Maximum number of partner observations.
            max_road_segments: Maximum road segment observations.
            obs_radius: Radius for observation circle.

        Returns:
            [H, W, 3] uint8 numpy array.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_aspect("equal")

        # Parse observation
        ego_obs = obs[:ego_features]
        partner_start = ego_features
        partner_end = partner_start + partner_features * max_partners
        partner_obs = obs[partner_start:partner_end].reshape(max_partners, partner_features)

        road_start = partner_end
        road_obs = obs[road_start : road_start + road_features * max_road_segments].reshape(max_road_segments, road_features)

        # Ego is at origin in its own frame
        ego_x, ego_y = 0, 0

        # Draw observation radius
        circle = Circle((ego_x, ego_y), obs_radius, fill=False, edgecolor="#0066FF", linestyle="--", alpha=0.3)
        ax.add_patch(circle)

        # Draw ego
        ax.plot(ego_x, ego_y, "o", color="#0066FF", markersize=12, zorder=10, label="Ego")

        # Draw partners (in ego frame)
        for i in range(max_partners):
            rel_x = partner_obs[i, 0] / 0.02  # Undo scaling
            rel_y = partner_obs[i, 1] / 0.02

            if abs(rel_x) > 200 or abs(rel_y) > 200:
                continue
            if rel_x == 0 and rel_y == 0:
                continue

            ax.plot(rel_x, rel_y, "s", color="#FF884D", markersize=6, alpha=0.7)
            ax.plot([ego_x, rel_x], [ego_y, rel_y], "-", color="#FF884D", alpha=0.2, linewidth=1)

        # Draw road segments (in ego frame)
        for i in range(max_road_segments):
            # Road features: [rel_x, rel_y, heading_x, heading_y] or similar
            if road_features >= 2:
                rx = road_obs[i, 0] / 0.02 if road_obs[i, 0] != 0 else 0
                ry = road_obs[i, 1] / 0.02 if road_obs[i, 1] != 0 else 0

                if abs(rx) > 200 or abs(ry) > 200:
                    continue
                if rx == 0 and ry == 0:
                    continue

                ax.plot(rx, ry, ".", color="#888888", markersize=3, alpha=0.5)

        ax.set_xlim([-obs_radius * 1.1, obs_radius * 1.1])
        ax.set_ylim([-obs_radius * 1.1, obs_radius * 1.1])
        ax.set_title(f"Agent {agent_idx} Observation")
        ax.legend(loc="upper right")
        ax.axis("off")

        img = img_from_fig(fig)
        return img

    def plot_ground_truth_trajectories(
        self,
        scenario_id: Optional[int] = None,
        max_agents: int = 32,
        show_road: bool = True,
    ) -> np.ndarray:
        """Plot ground truth (human) trajectories.

        Args:
            scenario_id: Optional scenario ID to filter.
            max_agents: Maximum agents to plot.
            show_road: If True, include road network.

        Returns:
            [H, W, 3] uint8 numpy array.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_aspect("equal")

        # Get trajectories
        trajectories = self.env.get_ground_truth_trajectories()

        # Get road if requested
        if show_road:
            road_polylines = self._get_road_polylines(scenario_id)
            plot_road_polylines(
                ax,
                road_polylines["x"],
                road_polylines["y"],
                road_polylines["lengths"],
                color="#444444",
                linewidth=1.0,
            )

        # Plot each agent's trajectory
        num_agents = min(len(trajectories["x"]), max_agents)
        colors = plt.cm.tab20(np.linspace(0, 1, num_agents))

        for i in range(num_agents):
            x = trajectories["x"][i].flatten()
            y = trajectories["y"][i].flatten()
            valid = trajectories["valid"][i].flatten()

            if valid.sum() < 2:
                continue

            plot_trajectory(ax, x, y, valid=valid, cmap="viridis", alpha=0.7, linewidth=2)

        # Viewport
        all_x = trajectories["x"][:num_agents].flatten()
        all_y = trajectories["y"][:num_agents].flatten()
        all_valid = trajectories["valid"][:num_agents].flatten()

        valid_x = all_x[all_valid.astype(bool)]
        valid_y = all_y[all_valid.astype(bool)]

        if len(valid_x) > 0:
            x_min, x_max, y_min, y_max = compute_viewport(valid_x, valid_y, padding=20.0)
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

        ax.set_title("Ground Truth Trajectories")
        ax.axis("off")

        img = img_from_fig(fig)
        return img

    def render_episode(
        self,
        max_steps: int = 91,
        save_path: Optional[str] = None,
        fps: int = 10,
        center_agent_idx: Optional[int] = None,
        zoom_radius: float = 100.0,
    ) -> List[np.ndarray]:
        """Render an entire episode as a sequence of frames.

        Args:
            max_steps: Maximum steps to render.
            save_path: If provided, save as video/gif.
            fps: Frames per second for saved video.
            center_agent_idx: If provided, follow this agent.
            zoom_radius: Zoom radius when following agent.

        Returns:
            List of [H, W, 3] uint8 numpy arrays.
        """
        frames = []
        trajectory_positions = []

        for step in range(max_steps):
            agent_states = self.env.get_global_agent_state()

            # Collect positions for trajectory
            positions = np.column_stack([agent_states["x"], agent_states["y"]])
            trajectory_positions.append(positions)

            # Create trajectory array from history
            if len(trajectory_positions) > 1:
                traj_array = np.stack(trajectory_positions, axis=1)  # (num_agents, num_steps, 2)
            else:
                traj_array = None

            # Render frame
            frame = self.plot_simulator_state(
                timestep=step,
                center_agent_idx=center_agent_idx,
                zoom_radius=zoom_radius,
                plot_trajectories=len(trajectory_positions) > 1,
                trajectory_positions=traj_array,
                agent_states=agent_states,
            )
            frames.append(frame)

            # Step environment
            # Note: This assumes env.step() is called externally
            # This method is for visualization only

        # Save if requested
        if save_path is not None:
            from pufferlib.visualize.utils import save_frames_as_gif, save_frames_as_video

            if save_path.endswith(".gif"):
                save_frames_as_gif(frames, save_path, fps=fps)
            else:
                save_frames_as_video(frames, save_path, fps=fps)

        return frames
