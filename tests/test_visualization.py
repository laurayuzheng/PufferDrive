"""Test visualization utilities for PufferDrive.

Tests both image and video generation using the baseline model.
"""

import os
import sys
import tempfile

# Set matplotlib backend before importing pyplot
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVisualizationImports:
    """Test that all visualization modules can be imported."""

    def test_import_color(self):
        from pufferlib.visualize.color import (
            ROAD_GRAPH_COLORS,
            AGENT_COLOR_BY_STATE,
            AGENT_COLOR_BY_POLICY,
        )

        assert len(ROAD_GRAPH_COLORS) > 0
        assert "ok" in AGENT_COLOR_BY_STATE
        assert "collided" in AGENT_COLOR_BY_STATE
        assert len(AGENT_COLOR_BY_POLICY) >= 3

    def test_import_utils(self):
        from pufferlib.visualize.utils import (
            img_from_fig,
            save_img_as_png,
            plot_numpy_bounding_boxes,
            plot_trajectory,
            plot_road_polylines,
            compute_viewport,
        )

    def test_import_core(self):
        from pufferlib.visualize.core import MatplotlibVisualizer

    def test_import_multiverse(self):
        from pufferlib.visualize.multiverse import MultiverseVisualizer

    def test_import_package(self):
        from pufferlib.visualize import (
            MatplotlibVisualizer,
            MultiverseVisualizer,
            AGENT_COLOR_BY_POLICY,
            img_from_fig,
        )


class TestUtilityFunctions:
    """Test low-level utility functions."""

    def test_img_from_fig(self):
        import matplotlib.pyplot as plt
        from pufferlib.visualize.utils import img_from_fig

        fig, ax = plt.subplots(figsize=(4, 4), dpi=50)
        ax.plot([0, 1], [0, 1])

        img = img_from_fig(fig)

        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3  # RGB
        assert img.dtype == np.uint8

    def test_save_img_as_png(self):
        from pufferlib.visualize.utils import save_img_as_png

        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.png")
            save_img_as_png(img, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_get_corners_polygon(self):
        from pufferlib.visualize.utils import get_corners_polygon

        corners = get_corners_polygon(x=0, y=0, length=4, width=2, yaw=0)

        assert corners.shape == (4, 2)
        # At yaw=0, front should be at positive x
        assert corners[0, 0] > 0  # top-left x
        assert corners[1, 0] > 0  # top-right x

    def test_plot_numpy_bounding_boxes(self):
        import matplotlib.pyplot as plt
        from pufferlib.visualize.utils import plot_numpy_bounding_boxes, img_from_fig

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])

        # Create some test bboxes: [x, y, length, width, yaw]
        bboxes = np.array(
            [
                [0, 0, 4, 2, 0],
                [5, 5, 4, 2, np.pi / 4],
                [-5, -5, 4, 2, np.pi / 2],
            ],
            dtype=np.float32,
        )

        plot_numpy_bounding_boxes(ax, bboxes, color="blue")

        img = img_from_fig(fig)
        assert img.shape[2] == 3

    def test_plot_trajectory(self):
        import matplotlib.pyplot as plt
        from pufferlib.visualize.utils import plot_trajectory, img_from_fig

        fig, ax = plt.subplots(figsize=(6, 6))

        x = np.linspace(0, 10, 50)
        y = np.sin(x)

        plot_trajectory(ax, x, y, cmap="viridis")

        ax.set_xlim([-1, 11])
        ax.set_ylim([-2, 2])

        img = img_from_fig(fig)
        assert img.shape[2] == 3

    def test_compute_viewport(self):
        from pufferlib.visualize.utils import compute_viewport

        x = np.array([0, 10, 5])
        y = np.array([0, 10, 5])

        x_min, x_max, y_min, y_max = compute_viewport(x, y, padding=5)

        assert x_min < 0
        assert x_max > 10
        assert y_min < 0
        assert y_max > 10

    def test_compute_viewport_centered(self):
        from pufferlib.visualize.utils import compute_viewport

        x_min, x_max, y_min, y_max = compute_viewport(np.array([0]), np.array([0]), center=(5, 5), radius=10)

        assert x_min == -5
        assert x_max == 15
        assert y_min == -5
        assert y_max == 15


class TestMatplotlibVisualizer:
    """Test the main MatplotlibVisualizer class with actual environment."""

    @pytest.fixture
    def env_and_visualizer(self):
        """Create environment and visualizer for testing."""
        from pufferlib.ocean.drive.drive import Drive

        # Create a small environment for testing
        env = Drive(
            num_agents=16,
            map_dir="resources/drive/binaries/validation",
            num_maps=1,
            episode_length=91,
        )
        env.reset()

        from pufferlib.visualize import MatplotlibVisualizer

        vis = MatplotlibVisualizer(env, goal_radius=2.0, figsize=(8, 8), dpi=80)

        yield env, vis

        env.close()

    def test_plot_simulator_state(self, env_and_visualizer):
        env, vis = env_and_visualizer

        img = vis.plot_simulator_state(timestep=0, zoom_radius=100.0)

        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8
        # Should be reasonable size
        assert img.shape[0] > 100
        assert img.shape[1] > 100

    def test_plot_simulator_state_with_masks(self, env_and_visualizer):
        env, vis = env_and_visualizer

        agent_states = env.get_global_agent_state()
        num_agents = len(agent_states["x"])

        # Create some test masks
        collision_mask = np.zeros(num_agents, dtype=bool)
        collision_mask[0] = True

        offroad_mask = np.zeros(num_agents, dtype=bool)
        offroad_mask[1] = True

        img = vis.plot_simulator_state(
            timestep=0,
            agent_states=agent_states,
            collision_mask=collision_mask,
            offroad_mask=offroad_mask,
        )

        assert img.shape[2] == 3

    def test_plot_simulator_state_with_policy_assignments(self, env_and_visualizer):
        env, vis = env_and_visualizer

        agent_states = env.get_global_agent_state()
        num_agents = len(agent_states["x"])

        # Assign agents to different policies
        policy_assignments = np.array([i % 3 for i in range(num_agents)])

        img = vis.plot_simulator_state(
            timestep=0,
            agent_states=agent_states,
            policy_assignments=policy_assignments,
        )

        assert img.shape[2] == 3

    def test_plot_ground_truth_trajectories(self, env_and_visualizer):
        env, vis = env_and_visualizer

        img = vis.plot_ground_truth_trajectories(max_agents=8)

        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_plot_3d_mode(self, env_and_visualizer):
        env, _ = env_and_visualizer

        from pufferlib.visualize import MatplotlibVisualizer

        vis_3d = MatplotlibVisualizer(env, render_3d=True, figsize=(8, 8), dpi=80)

        img = vis_3d.plot_simulator_state(timestep=0)

        assert img.shape[2] == 3

    def test_save_image(self, env_and_visualizer):
        from pufferlib.visualize.utils import save_img_as_png

        env, vis = env_and_visualizer

        img = vis.plot_simulator_state(timestep=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_state.png")
            save_img_as_png(img, path)

            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000  # Should be a real image


class TestVideoGeneration:
    """Test video/GIF generation."""

    @pytest.fixture
    def env(self):
        from pufferlib.ocean.drive.drive import Drive

        env = Drive(
            num_agents=16,
            map_dir="resources/drive/binaries/validation",
            num_maps=1,
            episode_length=91,
        )
        env.reset()
        yield env
        env.close()

    def test_save_frames_as_gif(self, env):
        from pufferlib.visualize import MatplotlibVisualizer
        from pufferlib.visualize.utils import save_frames_as_gif

        vis = MatplotlibVisualizer(env, figsize=(6, 6), dpi=50)

        # Generate a few frames
        frames = []
        for step in range(5):
            img = vis.plot_simulator_state(timestep=step)
            frames.append(img)

            # Take a random action to change state (shape must match action space)
            action = np.random.randint(0, 91, size=(env.num_agents, 1))
            env.step(action)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.gif")
            save_frames_as_gif(frames, path, fps=2)

            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000

    def test_save_frames_as_video(self, env):
        """Test video saving (requires ffmpeg)."""
        from pufferlib.visualize import MatplotlibVisualizer
        from pufferlib.visualize.utils import save_frames_as_video

        vis = MatplotlibVisualizer(env, figsize=(6, 6), dpi=50)

        # Generate a few frames
        frames = []
        for step in range(5):
            img = vis.plot_simulator_state(timestep=step)
            frames.append(img)

            action = np.random.randint(0, 91, size=(env.num_agents, 1))
            env.step(action)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.mp4")
            try:
                save_frames_as_video(frames, path, fps=2)
                assert os.path.exists(path)
                assert os.path.getsize(path) > 1000
            except ValueError as e:
                if "backend" in str(e).lower() or "ffmpeg" in str(e).lower():
                    pytest.skip("ffmpeg backend not available for video encoding")
                raise


class TestMultiverseVisualizer:
    """Test multiverse visualization for MoE comparison."""

    def test_plot_multiverse_grid_mock_data(self):
        """Test multiverse grid with mock trajectory data (no policy needed)."""
        from pufferlib.ocean.drive.drive import Drive
        from pufferlib.visualize.multiverse import MultiverseVisualizer

        env = Drive(
            num_agents=8,
            map_dir="resources/drive/binaries/validation",
            num_maps=1,
            episode_length=91,
        )
        env.reset()

        # Create visualizer without policy (we'll use mock data)
        vis = MultiverseVisualizer(env, policy=None, num_experts=3, figsize_per_cell=(4, 4), dpi=50)

        # Create mock trajectory data for each expert
        num_agents = 8
        num_steps = 20

        # Get initial positions
        agent_states = env.get_global_agent_state()
        init_x, init_y = agent_states["x"], agent_states["y"]

        trajectories_by_expert = {}
        for expert_idx in range(3):
            # Create slightly different trajectories for each expert
            positions = np.zeros((num_agents, num_steps, 2))
            for i in range(num_agents):
                # Each expert moves in slightly different direction
                angle = expert_idx * np.pi / 4
                for t in range(num_steps):
                    positions[i, t, 0] = init_x[i] + t * np.cos(angle) * 0.5
                    positions[i, t, 1] = init_y[i] + t * np.sin(angle) * 0.5

            trajectories_by_expert[expert_idx] = {
                "positions": positions,
                "headings": np.zeros((num_agents, num_steps)),
            }

        img = vis.plot_multiverse_grid(
            trajectories_by_expert=trajectories_by_expert,
            zoom_radius=50.0,
            title="Mock Multiverse Test",
        )

        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3

        env.close()

    def test_plot_expert_distribution_over_time(self):
        """Test expert probability visualization."""
        from pufferlib.visualize.multiverse import MultiverseVisualizer

        # Create mock expert probabilities
        num_timesteps = 50
        num_agents = 4
        num_experts = 3

        # Random expert probabilities that sum to 1
        probs = np.random.rand(num_timesteps, num_agents, num_experts)
        probs = probs / probs.sum(axis=2, keepdims=True)

        # Create minimal visualizer (env/policy not needed for this)
        class MockEnv:
            def get_road_edge_polylines(self):
                return {"x": np.array([]), "y": np.array([]), "lengths": np.array([])}

        vis = MultiverseVisualizer(MockEnv(), policy=None, num_experts=3)

        img = vis.plot_expert_distribution_over_time(probs, agent_indices=[0, 1])

        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3

    def test_plot_trajectory_with_expert_coloring(self):
        """Test trajectory colored by expert assignment."""
        from pufferlib.visualize.multiverse import MultiverseVisualizer

        class MockEnv:
            def get_road_edge_polylines(self):
                return {
                    "x": np.array([0, 10, 20], dtype=np.float32),
                    "y": np.array([0, 0, 0], dtype=np.float32),
                    "lengths": np.array([3], dtype=np.int32),
                }

        vis = MultiverseVisualizer(MockEnv(), policy=None, num_experts=3)

        # Create mock trajectory
        num_steps = 30
        positions = np.zeros((num_steps, 2))
        for t in range(num_steps):
            positions[t, 0] = t * 0.5
            positions[t, 1] = np.sin(t * 0.2) * 5

        # Expert assignments change over time
        expert_assignments = np.array([t // 10 for t in range(num_steps)])

        img = vis.plot_trajectory_with_expert_coloring(
            positions=positions,
            expert_assignments=expert_assignments,
            agent_idx=0,
            show_road=True,
            zoom_radius=20.0,
        )

        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3


class TestIntegrationWithPolicy:
    """Integration tests with actual policy (requires model weights)."""

    @pytest.fixture
    def env_policy_vis(self):
        """Load environment and policy for integration testing."""
        import torch

        baseline_path = "experiments/puffer_drive_baseline.pt"

        if not os.path.exists(baseline_path):
            pytest.skip(f"Baseline weights not found at {baseline_path}")

        from pufferlib.ocean.drive.drive import Drive
        from pufferlib.ocean.torch import Drive as DrivePolicy
        from pufferlib.visualize import MatplotlibVisualizer

        env = Drive(
            num_agents=16,
            map_dir="resources/drive/binaries/validation",
            num_maps=1,
            episode_length=91,
        )

        # Load policy - use hidden_size from checkpoint
        state_dict = torch.load(baseline_path, map_location="cpu")
        # Strip 'policy.' prefix if present (from LSTM wrapper)
        filtered_dict = {k.replace("policy.", ""): v for k, v in state_dict.items() if k.startswith("policy.")}
        if not filtered_dict:
            filtered_dict = state_dict

        # Infer hidden_size from checkpoint
        if "shared_embedding.1.weight" in filtered_dict:
            hidden_size = filtered_dict["shared_embedding.1.weight"].shape[0]
        else:
            hidden_size = 128  # default

        # Create policy with matching architecture
        policy = DrivePolicy(env, hidden_size=hidden_size)
        try:
            policy.load_state_dict(filtered_dict, strict=False)
        except RuntimeError as e:
            pytest.skip(f"Could not load checkpoint (architecture mismatch): {e}")
        policy.eval()

        vis = MatplotlibVisualizer(env, figsize=(8, 8), dpi=80)

        yield env, policy, vis

        env.close()

    def test_rollout_visualization(self, env_policy_vis):
        """Test visualization during policy rollout."""
        import torch

        env, policy, vis = env_policy_vis

        obs, *_ = env.reset()
        frames = []

        for step in range(10):
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                action, _, _, _ = policy(obs_tensor)
                action = action.numpy()

            # Visualize current state
            img = vis.plot_simulator_state(timestep=step, zoom_radius=80.0)
            frames.append(img)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)

        assert len(frames) == 10
        assert all(f.shape[2] == 3 for f in frames)

        # Save as GIF
        with tempfile.TemporaryDirectory() as tmpdir:
            from pufferlib.visualize.utils import save_frames_as_gif

            path = os.path.join(tmpdir, "rollout.gif")
            save_frames_as_gif(frames, path, fps=5)
            assert os.path.exists(path)
            print(f"Saved rollout GIF to {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
