"""PufferLib Visualization Utilities.

Matplotlib-based visualization for PufferDrive environments.
Inspired by GPUDrive visualization utilities.
"""

from pufferlib.visualize.color import (
    ROAD_GRAPH_COLORS,
    ROAD_GRAPH_TYPE_NAMES,
    AGENT_COLOR_BY_STATE,
    AGENT_COLOR_BY_POLICY,
    REL_OBS_OBJ_COLORS,
)
from pufferlib.visualize.utils import (
    img_from_fig,
    save_img_as_png,
    plot_numpy_bounding_boxes,
    plot_trajectory,
    get_corners_polygon,
)
from pufferlib.visualize.core import MatplotlibVisualizer
from pufferlib.visualize.multiverse import MultiverseVisualizer

__all__ = [
    "MatplotlibVisualizer",
    "MultiverseVisualizer",
    "ROAD_GRAPH_COLORS",
    "ROAD_GRAPH_TYPE_NAMES",
    "AGENT_COLOR_BY_STATE",
    "AGENT_COLOR_BY_POLICY",
    "REL_OBS_OBJ_COLORS",
    "img_from_fig",
    "save_img_as_png",
    "plot_numpy_bounding_boxes",
    "plot_trajectory",
    "get_corners_polygon",
]
