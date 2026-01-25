"""Color scheme definitions for PufferLib visualization.

Adapted from GPUDrive visualization utilities.
"""

import numpy as np

# Road graph entity types (matching drive.h)
ROAD_EDGE = 4
ROAD_LINE = 5
ROAD_LANE = 6
CROSSWALK = 7
SPEED_BUMP = 8
STOP_SIGN = 9
DRIVEWAY = 10

# Color definitions
dark_grey = "#000000"
light_grey = np.array([120, 120, 120]) / 255.0

ROAD_GRAPH_COLORS = {
    ROAD_EDGE: dark_grey,
    ROAD_LINE: light_grey,
    ROAD_LANE: np.array([230, 230, 230]) / 255.0,
    CROSSWALK: np.array([200, 200, 200]) / 255.0,
    SPEED_BUMP: np.array([0.85, 0.65, 0.13]),  # Golden
    STOP_SIGN: np.array([255, 0, 0]) / 255.0,  # Red
    DRIVEWAY: np.array([255, 100, 100]) / 255.0,  # Light red
}

ROAD_GRAPH_TYPE_NAMES = {
    ROAD_EDGE: "Road edge",
    ROAD_LINE: "Road line",
    ROAD_LANE: "Lane center",
    CROSSWALK: "Crosswalk",
    SPEED_BUMP: "Speed bump",
    STOP_SIGN: "Stop sign",
    DRIVEWAY: "Driveway",
}

# Agent colors by state
AGENT_COLOR_BY_STATE = {
    "ok": "#4B77BE",  # Blue - controlled and doing fine
    "collided": "#FF0000",  # Red - collided
    "off_road": "#FFA500",  # Orange - off-road
    "goal_reached": "#00FF00",  # Green - reached goal
    "log_replay": "#C7C7C7",  # Grey - expert/replay agents
    "ego": "#0066FF",  # Bright blue - ego agent
}

# Colors for multiple policies (MoE experts)
AGENT_COLOR_BY_POLICY = [
    "#2ECC71",  # Green - Expert 0
    "#3498DB",  # Blue - Expert 1
    "#9B59B6",  # Purple - Expert 2
    "#E74C3C",  # Red - Expert 3
    "#F39C12",  # Orange - Expert 4
]

# Observation visualization colors
REL_OBS_OBJ_COLORS = {
    "ego": "#0066FF",
    "ego_goal": "#0099CC",
    "partner_agents": "#FF884D",
    "road_points": "#888888",
}

# Trajectory colors (temporal gradient)
TRAJECTORY_CMAP = "viridis"
TRAJECTORY_PAST_COLOR = "#00FF00"  # Green
TRAJECTORY_FUTURE_COLOR = "#FF0000"  # Red

# PufferLib brand colors (from drive.h)
PUFF_RED = np.array([187, 0, 0]) / 255.0
PUFF_CYAN = np.array([0, 187, 187]) / 255.0
PUFF_WHITE = np.array([241, 241, 241]) / 255.0
PUFF_BACKGROUND = np.array([6, 24, 24]) / 255.0
