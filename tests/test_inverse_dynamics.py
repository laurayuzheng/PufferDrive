"""
Unit tests for the inverse dynamics module.

Tests the computation of expert actions from trajectory state transitions.
"""

import numpy as np
import pytest
import torch

from pufferlib.ocean.inverse_dynamics import (
    ACCELERATION_VALUES,
    STEERING_VALUES,
    NUM_ACCEL,
    NUM_STEER,
    NUM_DISCRETE_ACTIONS,
    compute_expert_actions_classic,
    compute_expert_actions_from_trajectory,
    InverseDynamicsModule,
    imitation_loss,
)


class TestConstants:
    """Test that constants match expected values from drive.h."""

    def test_acceleration_values(self):
        expected = np.array([-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0])
        np.testing.assert_array_almost_equal(ACCELERATION_VALUES, expected, decimal=3)

    def test_steering_values(self):
        expected = np.array(
            [-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0]
        )
        np.testing.assert_array_almost_equal(STEERING_VALUES, expected, decimal=3)

    def test_num_actions(self):
        assert NUM_ACCEL == 7
        assert NUM_STEER == 13
        assert NUM_DISCRETE_ACTIONS == 91


class TestComputeExpertActionsClassic:
    """Test the core inverse dynamics computation."""

    def test_straight_driving_constant_speed(self):
        """Straight driving at constant speed should give zero acceleration and steering."""
        batch_size = 10
        dt = 0.1
        speed = 5.0  # m/s

        # Current state: at origin, facing +x, moving at constant speed
        x_t = np.zeros(batch_size)
        y_t = np.zeros(batch_size)
        heading_t = np.zeros(batch_size)
        vx_t = np.full(batch_size, speed)
        vy_t = np.zeros(batch_size)

        # Next state: moved forward by speed * dt
        x_tp1 = x_t + speed * dt
        y_tp1 = y_t
        heading_tp1 = heading_t

        actions = compute_expert_actions_classic(
            x_t, y_t, heading_t, vx_t, vy_t, x_tp1, y_tp1, heading_tp1, dt
        )

        # Actions are now (batch, 2) with [accel_idx, steer_idx]
        accel_idx = actions[:, 0]
        steer_idx = actions[:, 1]

        # Zero acceleration is index 3, zero steering is index 6
        assert np.all(accel_idx == 3), f"Expected accel_idx=3, got {accel_idx}"
        assert np.all(steer_idx == 6), f"Expected steer_idx=6, got {steer_idx}"

    def test_acceleration(self):
        """Accelerating should give positive acceleration index."""
        batch_size = 5
        dt = 0.1
        speed = 5.0

        x_t = np.zeros(batch_size)
        y_t = np.zeros(batch_size)
        heading_t = np.zeros(batch_size)
        vx_t = np.full(batch_size, speed)
        vy_t = np.zeros(batch_size)

        # Next state: moved more than constant speed would suggest (accelerating)
        accel = 2.0  # m/s^2
        new_speed = speed + accel * dt
        x_tp1 = x_t + new_speed * dt
        y_tp1 = y_t
        heading_tp1 = heading_t

        actions = compute_expert_actions_classic(
            x_t, y_t, heading_t, vx_t, vy_t, x_tp1, y_tp1, heading_tp1, dt
        )

        accel_idx = actions[:, 0]
        # Acceleration index should be > 3 (positive acceleration)
        assert np.all(accel_idx > 3), f"Expected accel_idx > 3, got {accel_idx}"

    def test_deceleration(self):
        """Decelerating should give negative acceleration index."""
        batch_size = 5
        dt = 0.1
        speed = 5.0

        x_t = np.zeros(batch_size)
        y_t = np.zeros(batch_size)
        heading_t = np.zeros(batch_size)
        vx_t = np.full(batch_size, speed)
        vy_t = np.zeros(batch_size)

        # Next state: moved less than constant speed would suggest (decelerating)
        accel = -2.0  # m/s^2
        new_speed = speed + accel * dt
        x_tp1 = x_t + new_speed * dt
        y_tp1 = y_t
        heading_tp1 = heading_t

        actions = compute_expert_actions_classic(
            x_t, y_t, heading_t, vx_t, vy_t, x_tp1, y_tp1, heading_tp1, dt
        )

        accel_idx = actions[:, 0]
        # Acceleration index should be < 3 (negative acceleration)
        assert np.all(accel_idx < 3), f"Expected accel_idx < 3, got {accel_idx}"

    def test_left_turn(self):
        """Turning left should give positive steering index."""
        batch_size = 5
        dt = 0.1
        speed = 5.0

        x_t = np.zeros(batch_size)
        y_t = np.zeros(batch_size)
        heading_t = np.zeros(batch_size)
        vx_t = np.full(batch_size, speed)
        vy_t = np.zeros(batch_size)

        # Next state: turned left (positive heading change)
        heading_change = 0.1  # radians
        x_tp1 = x_t + speed * dt * np.cos(heading_change / 2)
        y_tp1 = y_t + speed * dt * np.sin(heading_change / 2)
        heading_tp1 = heading_t + heading_change

        actions = compute_expert_actions_classic(
            x_t, y_t, heading_t, vx_t, vy_t, x_tp1, y_tp1, heading_tp1, dt
        )

        steer_idx = actions[:, 1]
        # Steering index should be > 6 (positive steering = left turn)
        assert np.all(steer_idx > 6), f"Expected steer_idx > 6, got {steer_idx}"

    def test_right_turn(self):
        """Turning right should give negative steering index."""
        batch_size = 5
        dt = 0.1
        speed = 5.0

        x_t = np.zeros(batch_size)
        y_t = np.zeros(batch_size)
        heading_t = np.zeros(batch_size)
        vx_t = np.full(batch_size, speed)
        vy_t = np.zeros(batch_size)

        # Next state: turned right (negative heading change)
        heading_change = -0.1  # radians
        x_tp1 = x_t + speed * dt * np.cos(heading_change / 2)
        y_tp1 = y_t + speed * dt * np.sin(heading_change / 2)
        heading_tp1 = heading_t + heading_change

        actions = compute_expert_actions_classic(
            x_t, y_t, heading_t, vx_t, vy_t, x_tp1, y_tp1, heading_tp1, dt
        )

        steer_idx = actions[:, 1]
        # Steering index should be < 6 (negative steering = right turn)
        assert np.all(steer_idx < 6), f"Expected steer_idx < 6, got {steer_idx}"

    def test_action_bounds(self):
        """All computed actions should be within valid range."""
        batch_size = 100
        dt = 0.1

        # Random states
        np.random.seed(42)
        x_t = np.random.randn(batch_size) * 10
        y_t = np.random.randn(batch_size) * 10
        heading_t = np.random.uniform(-np.pi, np.pi, batch_size)
        speed = np.random.uniform(1, 10, batch_size)
        vx_t = speed * np.cos(heading_t)
        vy_t = speed * np.sin(heading_t)

        # Random next states (within reasonable bounds)
        x_tp1 = x_t + vx_t * dt + np.random.randn(batch_size) * 0.5
        y_tp1 = y_t + vy_t * dt + np.random.randn(batch_size) * 0.5
        heading_tp1 = heading_t + np.random.randn(batch_size) * 0.2

        actions = compute_expert_actions_classic(
            x_t, y_t, heading_t, vx_t, vy_t, x_tp1, y_tp1, heading_tp1, dt
        )

        assert np.all(actions >= 0), f"Action below 0: {actions.min()}"
        assert np.all(actions < NUM_DISCRETE_ACTIONS), f"Action >= 91: {actions.max()}"

    def test_batch_independence(self):
        """Each sample in batch should be computed independently."""
        dt = 0.1
        speed = 5.0

        # Two different scenarios
        x_t = np.array([0.0, 0.0])
        y_t = np.array([0.0, 0.0])
        heading_t = np.array([0.0, 0.0])
        vx_t = np.array([speed, speed])
        vy_t = np.array([0.0, 0.0])

        # First: straight, Second: turning
        x_tp1 = np.array([speed * dt, speed * dt * 0.99])
        y_tp1 = np.array([0.0, speed * dt * 0.1])
        heading_tp1 = np.array([0.0, 0.1])

        actions = compute_expert_actions_classic(
            x_t, y_t, heading_t, vx_t, vy_t, x_tp1, y_tp1, heading_tp1, dt
        )

        # Actions should be different (compare tuples)
        assert not np.array_equal(actions[0], actions[1]), "Batch samples should have different actions"

    def test_output_dtype(self):
        """Output should be int64."""
        batch_size = 5
        dt = 0.1

        x_t = np.zeros(batch_size)
        y_t = np.zeros(batch_size)
        heading_t = np.zeros(batch_size)
        vx_t = np.ones(batch_size)
        vy_t = np.zeros(batch_size)
        x_tp1 = x_t + dt
        y_tp1 = y_t
        heading_tp1 = heading_t

        actions = compute_expert_actions_classic(
            x_t, y_t, heading_t, vx_t, vy_t, x_tp1, y_tp1, heading_tp1, dt
        )

        assert actions.dtype == np.int64


class TestComputeExpertActionsFromTrajectory:
    """Test trajectory-based expert action computation."""

    def test_simple_trajectory(self):
        """Test with a simple straight-line trajectory."""
        num_agents = 3
        num_timesteps = 10
        dt = 0.1
        speed = 5.0

        # Straight trajectory: constant speed in +x direction
        traj_x = np.zeros((num_agents, num_timesteps))
        traj_y = np.zeros((num_agents, num_timesteps))
        traj_heading = np.zeros((num_agents, num_timesteps))
        traj_valid = np.ones((num_agents, num_timesteps), dtype=np.int32)

        for t in range(num_timesteps):
            traj_x[:, t] = speed * dt * t

        # Get actions at timestep 5
        expert_actions, valid_mask = compute_expert_actions_from_trajectory(
            traj_x, traj_y, traj_heading, traj_valid, timestep=5, dt=dt
        )

        assert expert_actions.shape == (num_agents, 2)
        assert valid_mask.shape == (num_agents,)
        assert np.all(valid_mask)  # All should be valid

        # Should be zero acceleration, zero steering
        accel_idx = expert_actions[:, 0]
        steer_idx = expert_actions[:, 1]
        assert np.all(accel_idx == 3), f"Expected accel_idx=3, got {accel_idx}"
        assert np.all(steer_idx == 6), f"Expected steer_idx=6, got {steer_idx}"

    def test_invalid_timesteps(self):
        """Test that invalid timesteps are handled correctly."""
        num_agents = 3
        num_timesteps = 10
        dt = 0.1

        traj_x = np.zeros((num_agents, num_timesteps))
        traj_y = np.zeros((num_agents, num_timesteps))
        traj_heading = np.zeros((num_agents, num_timesteps))
        traj_valid = np.ones((num_agents, num_timesteps), dtype=np.int32)

        # Make some timesteps invalid
        traj_valid[0, 5] = 0  # Agent 0, timestep 5 invalid
        traj_valid[1, 6] = 0  # Agent 1, timestep 6 invalid (next timestep)

        # At timestep 5:
        # - Agent 0: current timestep (5) invalid -> invalid
        # - Agent 1: next timestep (6) invalid -> invalid (need both t and t+1 valid)
        # - Agent 2: both valid -> valid
        expert_actions, valid_mask = compute_expert_actions_from_trajectory(
            traj_x, traj_y, traj_heading, traj_valid, timestep=5, dt=dt
        )
        assert not valid_mask[0], "Agent 0 should be invalid (current timestep invalid)"
        assert not valid_mask[1], "Agent 1 should be invalid (next timestep invalid)"
        assert valid_mask[2], "Agent 2 should be valid"

        # At timestep 4, all should be valid (agent 1's invalid is at timestep 6)
        expert_actions, valid_mask = compute_expert_actions_from_trajectory(
            traj_x, traj_y, traj_heading, traj_valid, timestep=4, dt=dt
        )
        assert not valid_mask[0], "Agent 0 should be invalid (timestep 5 invalid)"
        assert valid_mask[1], "Agent 1 should be valid at timestep 4"
        assert valid_mask[2], "Agent 2 should be valid"

    def test_last_timestep(self):
        """Test that last timestep returns zeros (can't compute action)."""
        num_agents = 3
        num_timesteps = 10
        dt = 0.1

        traj_x = np.zeros((num_agents, num_timesteps))
        traj_y = np.zeros((num_agents, num_timesteps))
        traj_heading = np.zeros((num_agents, num_timesteps))
        traj_valid = np.ones((num_agents, num_timesteps), dtype=np.int32)

        # Request action at last timestep
        expert_actions, valid_mask = compute_expert_actions_from_trajectory(
            traj_x, traj_y, traj_heading, traj_valid, timestep=num_timesteps - 1, dt=dt
        )

        assert np.all(expert_actions == 0)
        assert np.all(~valid_mask)  # All should be invalid

    def test_beyond_trajectory(self):
        """Test that timestep beyond trajectory length returns zeros."""
        num_agents = 3
        num_timesteps = 10
        dt = 0.1

        traj_x = np.zeros((num_agents, num_timesteps))
        traj_y = np.zeros((num_agents, num_timesteps))
        traj_heading = np.zeros((num_agents, num_timesteps))
        traj_valid = np.ones((num_agents, num_timesteps), dtype=np.int32)

        # Request action at timestep beyond trajectory
        expert_actions, valid_mask = compute_expert_actions_from_trajectory(
            traj_x, traj_y, traj_heading, traj_valid, timestep=20, dt=dt
        )

        assert np.all(expert_actions == 0)
        assert np.all(~valid_mask)


class TestInverseDynamicsModule:
    """Test the InverseDynamicsModule wrapper class."""

    def test_initialization(self):
        """Test module initialization."""
        module = InverseDynamicsModule(dt=0.1, vehicle_length=4.5)
        assert module.dt == 0.1
        assert module.vehicle_length == 4.5
        assert module.trajectories is None

    def test_set_trajectories(self):
        """Test setting trajectory data."""
        module = InverseDynamicsModule()

        trajectories = {
            "x": np.zeros((5, 10)),
            "y": np.zeros((5, 10)),
            "heading": np.zeros((5, 10)),
            "valid": np.ones((5, 10), dtype=np.int32),
        }

        module.set_trajectories(trajectories, init_steps=0)
        assert module.trajectories is not None

    def test_set_trajectories_3d(self):
        """Test setting trajectory data with extra dimension."""
        module = InverseDynamicsModule()

        # Trajectories with extra dimension (from drive.py)
        trajectories = {
            "x": np.zeros((5, 1, 10)),
            "y": np.zeros((5, 1, 10)),
            "heading": np.zeros((5, 1, 10)),
            "valid": np.ones((5, 1, 10), dtype=np.int32),
        }

        module.set_trajectories(trajectories, init_steps=0)

        # Should squeeze the extra dimension
        assert module.trajectories["x"].shape == (5, 10)

    def test_get_expert_actions(self):
        """Test getting expert actions from module."""
        module = InverseDynamicsModule(dt=0.1)

        # Create simple straight trajectory
        num_agents = 3
        num_timesteps = 10
        speed = 5.0
        dt = 0.1

        trajectories = {
            "x": np.zeros((num_agents, num_timesteps)),
            "y": np.zeros((num_agents, num_timesteps)),
            "heading": np.zeros((num_agents, num_timesteps)),
            "valid": np.ones((num_agents, num_timesteps), dtype=np.int32),
        }

        for t in range(num_timesteps):
            trajectories["x"][:, t] = speed * dt * t

        module.set_trajectories(trajectories)

        expert_actions, valid_mask = module.get_expert_actions(timestep=5)

        assert expert_actions.shape == (num_agents, 2)
        assert valid_mask.shape == (num_agents,)

    def test_get_expert_actions_without_trajectories(self):
        """Test that getting actions without setting trajectories raises error."""
        module = InverseDynamicsModule()

        with pytest.raises(RuntimeError, match="Trajectories not set"):
            module.get_expert_actions(timestep=0)


class TestImitationLoss:
    """Test the imitation loss computation for multi-head discrete actions."""

    def test_basic_loss(self):
        """Test basic cross-entropy loss computation with multi-head logits."""
        batch_size = 10

        # Multi-head logits: (batch, 7) for accel, (batch, 13) for steering
        accel_logits = torch.randn(batch_size, NUM_ACCEL)
        steer_logits = torch.randn(batch_size, NUM_STEER)
        policy_logits = [accel_logits, steer_logits]

        # Expert actions: (batch, 2) with [accel_idx, steer_idx]
        expert_actions = torch.zeros(batch_size, 2, dtype=torch.long)

        # All valid
        valid_mask = torch.ones(batch_size, dtype=torch.bool)

        loss = imitation_loss(policy_logits, expert_actions, valid_mask)

        assert loss.shape == ()  # Scalar
        assert loss.item() > 0  # Should be positive

    def test_no_valid_samples(self):
        """Test that loss is zero when no valid samples."""
        batch_size = 10

        accel_logits = torch.randn(batch_size, NUM_ACCEL)
        steer_logits = torch.randn(batch_size, NUM_STEER)
        policy_logits = [accel_logits, steer_logits]
        expert_actions = torch.zeros(batch_size, 2, dtype=torch.long)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool)  # All invalid

        loss = imitation_loss(policy_logits, expert_actions, valid_mask)

        assert loss.item() == 0.0

    def test_partial_valid(self):
        """Test loss computation with partially valid samples."""
        batch_size = 10

        accel_logits = torch.randn(batch_size, NUM_ACCEL)
        steer_logits = torch.randn(batch_size, NUM_STEER)
        policy_logits = [accel_logits, steer_logits]
        expert_actions = torch.zeros(batch_size, 2, dtype=torch.long)

        # Only first 5 valid
        valid_mask = torch.zeros(batch_size, dtype=torch.bool)
        valid_mask[:5] = True

        loss = imitation_loss(policy_logits, expert_actions, valid_mask)

        assert loss.item() > 0

    def test_perfect_prediction(self):
        """Test that loss is low when predictions match experts."""
        batch_size = 10

        # Expert actions: random valid indices
        expert_accel = torch.randint(0, NUM_ACCEL, (batch_size,))
        expert_steer = torch.randint(0, NUM_STEER, (batch_size,))
        expert_actions = torch.stack([expert_accel, expert_steer], dim=1)

        # Create logits that strongly predict expert actions
        accel_logits = torch.full((batch_size, NUM_ACCEL), -10.0)
        steer_logits = torch.full((batch_size, NUM_STEER), -10.0)
        for i in range(batch_size):
            accel_logits[i, expert_accel[i]] = 10.0
            steer_logits[i, expert_steer[i]] = 10.0

        policy_logits = [accel_logits, steer_logits]
        valid_mask = torch.ones(batch_size, dtype=torch.bool)

        loss = imitation_loss(policy_logits, expert_actions, valid_mask)

        # Loss should be very small (close to 0)
        assert loss.item() < 0.01

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        batch_size = 10

        accel_logits = torch.randn(batch_size, NUM_ACCEL, requires_grad=True)
        steer_logits = torch.randn(batch_size, NUM_STEER, requires_grad=True)
        policy_logits = [accel_logits, steer_logits]
        expert_actions = torch.stack([
            torch.randint(0, NUM_ACCEL, (batch_size,)),
            torch.randint(0, NUM_STEER, (batch_size,))
        ], dim=1)
        valid_mask = torch.ones(batch_size, dtype=torch.bool)

        loss = imitation_loss(policy_logits, expert_actions, valid_mask)
        loss.backward()

        assert accel_logits.grad is not None
        assert steer_logits.grad is not None
        assert accel_logits.grad.shape == accel_logits.shape
        assert steer_logits.grad.shape == steer_logits.shape


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_stationary_vehicle(self):
        """Test inverse dynamics for stationary vehicle."""
        batch_size = 5
        dt = 0.1

        # Stationary: zero velocity, no movement
        x_t = np.zeros(batch_size)
        y_t = np.zeros(batch_size)
        heading_t = np.zeros(batch_size)
        vx_t = np.zeros(batch_size)
        vy_t = np.zeros(batch_size)
        x_tp1 = np.zeros(batch_size)
        y_tp1 = np.zeros(batch_size)
        heading_tp1 = np.zeros(batch_size)

        # Should not crash
        actions = compute_expert_actions_classic(
            x_t, y_t, heading_t, vx_t, vy_t, x_tp1, y_tp1, heading_tp1, dt
        )

        assert actions.shape == (batch_size, 2)
        assert np.all(np.isfinite(actions))

    def test_very_slow_speed(self):
        """Test inverse dynamics at very low speed."""
        batch_size = 5
        dt = 0.1
        speed = 0.01  # Very slow

        x_t = np.zeros(batch_size)
        y_t = np.zeros(batch_size)
        heading_t = np.zeros(batch_size)
        vx_t = np.full(batch_size, speed)
        vy_t = np.zeros(batch_size)
        x_tp1 = x_t + speed * dt
        y_tp1 = y_t
        heading_tp1 = heading_t

        actions = compute_expert_actions_classic(
            x_t, y_t, heading_t, vx_t, vy_t, x_tp1, y_tp1, heading_tp1, dt
        )

        assert np.all(np.isfinite(actions))
        assert np.all(actions >= 0)
        assert np.all(actions < NUM_DISCRETE_ACTIONS)

    def test_reverse_driving(self):
        """Test inverse dynamics for reverse driving."""
        batch_size = 5
        dt = 0.1
        speed = -3.0  # Negative = reverse

        x_t = np.zeros(batch_size)
        y_t = np.zeros(batch_size)
        heading_t = np.zeros(batch_size)  # Facing +x
        vx_t = np.full(batch_size, speed)  # Moving in -x direction
        vy_t = np.zeros(batch_size)
        x_tp1 = x_t + speed * dt  # Negative displacement
        y_tp1 = y_t
        heading_tp1 = heading_t

        actions = compute_expert_actions_classic(
            x_t, y_t, heading_t, vx_t, vy_t, x_tp1, y_tp1, heading_tp1, dt
        )

        assert np.all(np.isfinite(actions))
        assert np.all(actions >= 0)
        assert np.all(actions < NUM_DISCRETE_ACTIONS)

    def test_heading_wraparound(self):
        """Test that heading wraparound is handled correctly."""
        batch_size = 5
        dt = 0.1
        speed = 5.0

        x_t = np.zeros(batch_size)
        y_t = np.zeros(batch_size)
        heading_t = np.full(batch_size, np.pi - 0.05)  # Near +pi
        vx_t = speed * np.cos(heading_t)
        vy_t = speed * np.sin(heading_t)

        # Turn past pi (wraps to near -pi)
        heading_tp1 = np.full(batch_size, -np.pi + 0.05)
        x_tp1 = x_t + vx_t * dt
        y_tp1 = y_t + vy_t * dt

        actions = compute_expert_actions_classic(
            x_t, y_t, heading_t, vx_t, vy_t, x_tp1, y_tp1, heading_tp1, dt
        )

        assert np.all(np.isfinite(actions))
        # Should detect a small turn, not a large one
        steer_idx = actions[:, 1]
        # Steering should be near center (small turn)
        assert np.all(np.abs(steer_idx - 6) <= 4), f"Unexpected large steering: {steer_idx}"


class TestMultiprocessingExpertActions:
    """Test that Multiprocessing backend provides expert actions correctly."""

    @pytest.fixture
    def vecenv_kwargs(self):
        """Common kwargs for creating test environments."""
        return {
            "num_agents": 32,
            "num_maps": 10,
            "map_dir": "resources/drive/binaries/training",
            "episode_length": 91,
            "resample_frequency": 910,
        }

    def test_multiprocessing_has_get_expert_actions(self):
        """Verify Multiprocessing class has get_expert_actions method."""
        import pufferlib.vector as vector

        assert hasattr(vector.Multiprocessing, "get_expert_actions")

    def test_serial_has_get_expert_actions(self):
        """Verify Serial class has get_expert_actions method."""
        import pufferlib.vector as vector

        assert hasattr(vector.Serial, "get_expert_actions")

    def test_serial_expert_actions_valid(self, vecenv_kwargs):
        """Test Serial backend returns valid expert actions."""
        import pufferlib
        import pufferlib.vector as vector
        from pufferlib.ocean import env_creator

        make_env = env_creator("puffer_drive")
        vecenv = pufferlib.vector.make(
            make_env,
            env_kwargs=vecenv_kwargs,
            backend=vector.Serial,
            num_envs=1,
        )

        try:
            vecenv.async_reset(seed=42)
            vecenv.recv()

            expert_actions, expert_valid = vecenv.get_expert_actions()

            # Check shapes: (num_agents, 2) for [accel_idx, steer_idx]
            assert expert_actions.shape == (32, 2)
            assert expert_valid.shape == (32,)
            assert expert_actions.dtype == np.int64
            assert expert_valid.dtype == bool

            # Serial backend should have valid expert actions
            assert expert_valid.sum() > 0, "Serial backend should return valid expert actions"

            # Actions should be in valid range: accel_idx in [0, 6], steer_idx in [0, 12]
            valid_actions = expert_actions[expert_valid]
            assert np.all(valid_actions[:, 0] >= 0) and np.all(valid_actions[:, 0] <= 6)
            assert np.all(valid_actions[:, 1] >= 0) and np.all(valid_actions[:, 1] <= 12)
        finally:
            vecenv.close()

    def test_multiprocessing_expert_actions_valid(self, vecenv_kwargs):
        """Test Multiprocessing backend returns valid expert actions."""
        import pufferlib
        import pufferlib.vector as vector
        from pufferlib.ocean import env_creator

        make_env = env_creator("puffer_drive")
        vecenv = pufferlib.vector.make(
            make_env,
            env_kwargs=vecenv_kwargs,
            backend=vector.Multiprocessing,
            num_envs=2,
            num_workers=2,
            batch_size=2,
        )

        try:
            vecenv.async_reset(seed=42)
            vecenv.recv()

            expert_actions, expert_valid = vecenv.get_expert_actions()

            # Check shapes (2 envs * 32 agents = 64, 2 for [accel_idx, steer_idx])
            assert expert_actions.shape == (64, 2)
            assert expert_valid.shape == (64,)
            assert expert_actions.dtype == np.int64
            assert expert_valid.dtype == bool

            # Multiprocessing backend should have valid expert actions
            assert expert_valid.sum() > 0, "Multiprocessing backend should return valid expert actions"

            # Actions should be in valid range: accel_idx in [0, 6], steer_idx in [0, 12]
            valid_actions = expert_actions[expert_valid]
            assert np.all(valid_actions[:, 0] >= 0) and np.all(valid_actions[:, 0] <= 6)
            assert np.all(valid_actions[:, 1] >= 0) and np.all(valid_actions[:, 1] <= 12)
        finally:
            vecenv.close()

    def test_multiprocessing_expert_actions_update_on_step(self, vecenv_kwargs):
        """Test that expert actions update correctly after steps."""
        import pufferlib
        import pufferlib.vector as vector
        from pufferlib.ocean import env_creator

        make_env = env_creator("puffer_drive")
        vecenv = pufferlib.vector.make(
            make_env,
            env_kwargs=vecenv_kwargs,
            backend=vector.Multiprocessing,
            num_envs=2,
            num_workers=2,
            batch_size=2,
        )

        try:
            vecenv.async_reset(seed=42)
            vecenv.recv()

            # Get initial expert actions
            expert_actions_0, expert_valid_0 = vecenv.get_expert_actions()
            valid_sum_0 = expert_valid_0.sum()

            # Step
            actions = vecenv.action_space.sample()
            vecenv.send(actions)
            vecenv.recv()

            # Get expert actions after step
            expert_actions_1, expert_valid_1 = vecenv.get_expert_actions()
            valid_sum_1 = expert_valid_1.sum()
            
            print()
            print("expert actions 0", expert_actions_0)
            print("expert actions 1", expert_actions_1)

            # Both should have valid expert actions
            assert valid_sum_0 > 0, "Should have valid actions after reset"
            assert valid_sum_1 > 0, "Should have valid actions after step"
        finally:
            vecenv.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
