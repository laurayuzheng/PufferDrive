"""
Inverse dynamics for computing expert actions from trajectory state transitions.

Given consecutive states from human trajectories, computes the discrete action
that would most closely produce the observed state transition.
"""

import numpy as np
import torch
from torch import Tensor


# Classic dynamics discrete action space (must match drive.h)
ACCELERATION_VALUES = np.array([-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0], dtype=np.float32)
STEERING_VALUES = np.array(
    [-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0],
    dtype=np.float32,
)

NUM_ACCEL = len(ACCELERATION_VALUES)  # 7
NUM_STEER = len(STEERING_VALUES)  # 13
NUM_DISCRETE_ACTIONS = NUM_ACCEL * NUM_STEER  # 91


def compute_expert_actions_classic(
    x_t: np.ndarray,
    y_t: np.ndarray,
    heading_t: np.ndarray,
    vx_t: np.ndarray,
    vy_t: np.ndarray,
    x_tp1: np.ndarray,
    y_tp1: np.ndarray,
    heading_tp1: np.ndarray,
    dt: float = 0.1,
    vehicle_length: float = 4.5,
) -> np.ndarray:
    """
    Compute discrete expert actions from state transitions using classic dynamics.

    Uses inverse bicycle model to recover (acceleration, steering) from state transition,
    then finds the closest discrete action indices.

    Args:
        x_t, y_t, heading_t: Current position and heading (batch,)
        vx_t, vy_t: Current velocity components (batch,)
        x_tp1, y_tp1, heading_tp1: Next position and heading (batch,)
        dt: Time step (default 0.1s)
        vehicle_length: Vehicle wheelbase for bicycle model

    Returns:
        expert_actions: Array of shape (batch, 2) with [accel_idx, steer_idx] per sample
                       accel_idx in [0, 6], steer_idx in [0, 12]
    """
    batch_size = x_t.shape[0]

    # Compute current signed speed
    speed_magnitude = np.sqrt(vx_t**2 + vy_t**2)
    heading_x = np.cos(heading_t)
    heading_y = np.sin(heading_t)
    v_dot_heading = vx_t * heading_x + vy_t * heading_y
    signed_speed = np.copysign(speed_magnitude, v_dot_heading)

    # Estimate next speed from position change (approximate)
    dx = x_tp1 - x_t
    dy = y_tp1 - y_t
    displacement = np.sqrt(dx**2 + dy**2)

    # Direction of movement relative to current heading
    movement_angle = np.arctan2(dy, dx)
    angle_diff = movement_angle - heading_t
    # Normalize to [-pi, pi]
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    movement_sign = np.sign(np.cos(angle_diff))
    movement_sign = np.where(displacement < 0.01, np.sign(signed_speed), movement_sign)

    next_signed_speed = movement_sign * displacement / dt

    # Compute acceleration from speed change
    acceleration = (next_signed_speed - signed_speed) / dt

    # Compute yaw rate from heading change
    dheading = heading_tp1 - heading_t
    # Normalize to [-pi, pi]
    dheading = np.arctan2(np.sin(dheading), np.cos(dheading))
    yaw_rate = dheading / dt

    # Inverse bicycle model: solve for steering from yaw rate
    # yaw_rate = (v * cos(beta) * tan(steering)) / L
    # where beta = tanh(0.5 * tan(steering))
    # For small steering, beta ≈ 0, so:
    # steering ≈ arctan(yaw_rate * L / v)

    # Use average speed for stability
    avg_speed = 0.5 * (signed_speed + next_signed_speed)
    avg_speed = np.where(np.abs(avg_speed) < 0.1, 0.1 * np.sign(avg_speed + 1e-8), avg_speed)

    # Approximate steering (small angle approximation)
    steering_approx = np.arctan(yaw_rate * vehicle_length / avg_speed)

    # Clamp to valid range
    steering_approx = np.clip(steering_approx, STEERING_VALUES[0], STEERING_VALUES[-1])
    acceleration = np.clip(acceleration, ACCELERATION_VALUES[0], ACCELERATION_VALUES[-1])

    # Find closest discrete action indices
    accel_idx = np.argmin(np.abs(acceleration[:, None] - ACCELERATION_VALUES[None, :]), axis=1)
    steer_idx = np.argmin(np.abs(steering_approx[:, None] - STEERING_VALUES[None, :]), axis=1)

    # Return as (batch, 2) array with [accel_idx, steer_idx] per sample
    expert_actions = np.stack([accel_idx, steer_idx], axis=1).astype(np.int64)

    return expert_actions


def compute_expert_actions_from_trajectory(
    traj_x: np.ndarray,
    traj_y: np.ndarray,
    traj_heading: np.ndarray,
    traj_valid: np.ndarray,
    timestep: int,
    dt: float = 0.1,
    vehicle_length: float = 4.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute expert actions for all agents at current timestep from trajectory data.

    Args:
        traj_x: Trajectory x positions (num_agents, num_timesteps)
        traj_y: Trajectory y positions (num_agents, num_timesteps)
        traj_heading: Trajectory headings (num_agents, num_timesteps)
        traj_valid: Trajectory validity flags (num_agents, num_timesteps)
        timestep: Current timestep
        dt: Time step duration
        vehicle_length: Vehicle wheelbase

    Returns:
        expert_actions: Array of shape (num_agents, 2) with [accel_idx, steer_idx]
        valid_mask: Boolean mask for valid expert actions (num_agents,)
    """
    num_agents = traj_x.shape[0]
    max_timestep = traj_x.shape[1] - 1

    if timestep >= max_timestep:
        # Can't compute action for last timestep
        return np.zeros((num_agents, 2), dtype=np.int64), np.zeros(num_agents, dtype=bool)

    # Extract current and next states
    x_t = traj_x[:, timestep]
    y_t = traj_y[:, timestep]
    heading_t = traj_heading[:, timestep]

    x_tp1 = traj_x[:, timestep + 1]
    y_tp1 = traj_y[:, timestep + 1]
    heading_tp1 = traj_heading[:, timestep + 1]

    # Approximate velocity from position difference (for t-1 to t, or use t to t+1)
    if timestep > 0:
        dx_prev = x_t - traj_x[:, timestep - 1]
        dy_prev = y_t - traj_y[:, timestep - 1]
        vx_t = dx_prev / dt
        vy_t = dy_prev / dt
    else:
        # First timestep: estimate from forward difference
        vx_t = (x_tp1 - x_t) / dt
        vy_t = (y_tp1 - y_t) / dt

    # Compute expert actions
    expert_actions = compute_expert_actions_classic(
        x_t, y_t, heading_t, vx_t, vy_t, x_tp1, y_tp1, heading_tp1, dt, vehicle_length
    )

    # Valid mask: both current and next timestep must be valid
    valid_t = traj_valid[:, timestep].astype(bool)
    valid_tp1 = traj_valid[:, timestep + 1].astype(bool)
    valid_mask = valid_t & valid_tp1

    return expert_actions, valid_mask


class InverseDynamicsModule:
    """
    Module for computing expert actions during rollouts.

    Caches trajectory data and provides expert action computation at each timestep.
    """

    def __init__(self, dt: float = 0.1, vehicle_length: float = 4.5):
        self.dt = dt
        self.vehicle_length = vehicle_length
        self.trajectories = None
        self.init_steps = 0

    def set_trajectories(self, trajectories: dict, init_steps: int = 0):
        """
        Set trajectory data for the current batch of scenarios.

        Args:
            trajectories: Dict with 'x', 'y', 'heading', 'valid' arrays
                         Shape: (num_agents, num_timesteps) or (num_agents, 1, num_timesteps)
            init_steps: Number of initial steps to skip
        """
        self.trajectories = trajectories
        self.init_steps = init_steps

        # Handle extra dimension if present (from drive.py)
        if len(trajectories["x"].shape) == 3:
            self.trajectories = {k: v[:, 0, :] for k, v in trajectories.items()}

    def get_expert_actions(self, timestep: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get expert actions for current timestep.

        Args:
            timestep: Current environment timestep (relative to init_steps)

        Returns:
            expert_actions: Array of shape (num_agents, 2) with [accel_idx, steer_idx]
            valid_mask: Boolean mask for valid expert actions (num_agents,)
        """
        if self.trajectories is None:
            raise RuntimeError("Trajectories not set. Call set_trajectories first.")

        # Adjust timestep to account for init_steps
        traj_timestep = timestep  # Already relative to init_steps in stored trajectories

        return compute_expert_actions_from_trajectory(
            self.trajectories["x"],
            self.trajectories["y"],
            self.trajectories["heading"],
            self.trajectories["valid"],
            traj_timestep,
            self.dt,
            self.vehicle_length,
        )


def imitation_loss(
    policy_logits: list[Tensor] | tuple[Tensor, ...],
    expert_actions: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """
    Compute cross-entropy imitation loss for multi-head discrete actions.

    Args:
        policy_logits: List/tuple of policy output logits [(batch, 7), (batch, 13)]
                      for acceleration and steering heads
        expert_actions: Ground truth action indices (batch, 2) with [accel_idx, steer_idx]
        valid_mask: Boolean mask for valid samples (batch,)

    Returns:
        Cross-entropy loss (scalar), averaged over valid samples and heads
    """
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=policy_logits[0].device)

    # Compute cross-entropy for each action head
    total_loss = torch.tensor(0.0, device=policy_logits[0].device)
    for head_idx, logits in enumerate(policy_logits):
        valid_logits = logits[valid_mask]
        valid_actions = expert_actions[valid_mask, head_idx]
        head_loss = torch.nn.functional.cross_entropy(valid_logits, valid_actions)
        total_loss = total_loss + head_loss

    # Average over heads
    return total_loss / len(policy_logits)
