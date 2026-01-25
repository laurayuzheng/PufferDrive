"""Utility functions for PufferLib visualization.

Low-level drawing primitives for matplotlib-based rendering.
Adapted from GPUDrive visualization utilities.
"""

import os
from typing import Optional, Tuple, List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.collections import LineCollection, PatchCollection

from pufferlib.visualize.color import ROAD_GRAPH_COLORS


def img_from_fig(fig: matplotlib.figure.Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy RGB array.

    Args:
        fig: Matplotlib figure object.

    Returns:
        [H, W, 3] uint8 numpy array.
    """
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    fig.canvas.draw()

    # Get RGBA buffer and convert to RGB
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()  # Drop alpha channel

    plt.close(fig)
    return img


def save_img_as_png(img: np.ndarray, filename: str = "/tmp/img.png"):
    """Save numpy image to disk as PNG.

    Args:
        img: [H, W, 3] uint8 numpy array.
        filename: Output file path.
    """
    outdir = os.path.dirname(filename)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    Image.fromarray(img).save(filename)


def get_corners_polygon(x: float, y: float, length: float, width: float, yaw: float) -> np.ndarray:
    """Calculate corners of a rotated rectangle.

    Args:
        x, y: Center position.
        length: Length of rectangle (along heading direction).
        width: Width of rectangle (perpendicular to heading).
        yaw: Rotation angle in radians.

    Returns:
        (4, 2) array of corner coordinates [tl, tr, br, bl].
    """
    c, s = np.cos(yaw), np.sin(yaw)

    # Unit vectors
    u = np.array([c, s])  # Along length
    ut = np.array([s, -c])  # Perpendicular

    pt = np.array([x, y])

    # Corners: top-left, top-right, bottom-right, bottom-left
    tl = pt + length / 2 * u - width / 2 * ut
    tr = pt + length / 2 * u + width / 2 * ut
    br = pt - length / 2 * u + width / 2 * ut
    bl = pt - length / 2 * u - width / 2 * ut

    return np.array([tl, tr, br, bl])


def plot_numpy_bounding_boxes(
    ax: matplotlib.axes.Axes,
    bboxes: np.ndarray,
    color: Union[str, np.ndarray, List],
    alpha: float = 1.0,
    line_width_scale: float = 1.5,
    as_center_pts: bool = False,
    label: Optional[str] = None,
    draw_heading: bool = True,
    zorder: int = 4,
) -> None:
    """Plot multiple bounding boxes with optional heading arrows.

    Args:
        ax: Matplotlib axes.
        bboxes: Shape (num_bbox, 5) with [x, y, length, width, yaw].
        color: RGB color or color string.
        alpha: Transparency (0=transparent, 1=opaque).
        line_width_scale: Scale factor for line width.
        as_center_pts: If True, draw as center points only.
        label: Legend label for this set of boxes.
        draw_heading: If True, draw heading arrow inside box.
        zorder: Drawing order (higher = on top).
    """
    if bboxes.ndim != 2 or bboxes.shape[1] != 5:
        raise ValueError(f"Expected bboxes shape (N, 5), got {bboxes.shape}")

    if len(bboxes) == 0:
        return

    if as_center_pts:
        ax.plot(
            bboxes[:, 0],
            bboxes[:, 1],
            "o",
            color=color,
            ms=4,
            alpha=alpha,
            label=label,
        )
        return

    # Vectorized corner computation
    c = np.cos(bboxes[:, 4])
    s = np.sin(bboxes[:, 4])
    pt = np.array([bboxes[:, 0], bboxes[:, 1]])  # (2, N)
    length, width = bboxes[:, 2], bboxes[:, 3]
    u = np.array([c, s])
    ut = np.array([s, -c])

    # Compute box corners
    tl = pt + length / 2 * u - width / 2 * ut
    tr = pt + length / 2 * u + width / 2 * ut
    br = pt - length / 2 * u + width / 2 * ut
    bl = pt - length / 2 * u - width / 2 * ut

    # Draw bounding boxes
    ax.plot(
        [tl[0, :], tr[0, :], br[0, :], bl[0, :], tl[0, :]],
        [tl[1, :], tr[1, :], br[1, :], bl[1, :], tl[1, :]],
        color=color,
        zorder=zorder,
        linewidth=1.7 * line_width_scale,
        alpha=alpha,
        label=label,
    )

    if draw_heading:
        # Heading arrow: center-left -> center-right -> center-front -> center-left
        cl = pt - width / 2 * ut
        cr = pt + width / 2 * ut
        cf = pt + length / 2 * u

        ax.plot(
            [cl[0, :], cr[0, :], cf[0, :], cl[0, :]],
            [cl[1, :], cr[1, :], cf[1, :], cl[1, :]],
            color=color,
            zorder=zorder + 2,
            alpha=alpha,
            linewidth=1.5 * line_width_scale,
        )


def plot_single_bounding_box(
    ax: matplotlib.axes.Axes,
    x: float,
    y: float,
    length: float,
    width: float,
    yaw: float,
    color: str = "blue",
    alpha: float = 1.0,
    linewidth: float = 2.0,
    fill: bool = False,
    fill_alpha: float = 0.3,
    label: Optional[str] = None,
) -> None:
    """Plot a single bounding box with heading.

    Args:
        ax: Matplotlib axes.
        x, y: Center position.
        length, width: Dimensions.
        yaw: Rotation angle in radians.
        color: Color string or RGB.
        alpha: Line transparency.
        linewidth: Line width.
        fill: If True, fill the rectangle.
        fill_alpha: Fill transparency.
        label: Legend label.
    """
    corners = get_corners_polygon(x, y, length, width, yaw)
    polygon = Polygon(corners, closed=True, fill=fill, edgecolor=color, facecolor=color if fill else "none", alpha=alpha if not fill else fill_alpha, linewidth=linewidth, label=label)
    ax.add_patch(polygon)

    # Draw heading arrow
    c, s = np.cos(yaw), np.sin(yaw)
    arrow_len = length * 0.4
    ax.arrow(
        x,
        y,
        arrow_len * c,
        arrow_len * s,
        head_width=width * 0.3,
        head_length=length * 0.15,
        fc=color,
        ec=color,
        alpha=alpha,
    )


def plot_trajectory(
    ax: matplotlib.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    valid: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    linewidth: float = 2.0,
    alpha: float = 0.8,
    label: Optional[str] = None,
    show_points: bool = False,
    point_size: float = 20,
) -> None:
    """Plot a trajectory with temporal color gradient.

    Args:
        ax: Matplotlib axes.
        x, y: Arrays of positions.
        valid: Optional boolean mask for valid points.
        cmap: Colormap name for temporal gradient.
        linewidth: Line width.
        alpha: Transparency.
        label: Legend label.
        show_points: If True, also plot points.
        point_size: Size of trajectory points.
    """
    if valid is not None:
        # Filter to valid points
        mask = valid.astype(bool)
        x = x[mask]
        y = y[mask]

    if len(x) < 2:
        return

    # Create line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color by time
    colors = np.linspace(0, 1, len(segments))

    lc = LineCollection(segments, cmap=cmap, alpha=alpha, linewidth=linewidth, label=label)
    lc.set_array(colors)
    ax.add_collection(lc)

    if show_points:
        ax.scatter(x, y, c=np.linspace(0, 1, len(x)), cmap=cmap, s=point_size, alpha=alpha, zorder=5)


def plot_road_polylines(
    ax: matplotlib.axes.Axes,
    polylines_x: np.ndarray,
    polylines_y: np.ndarray,
    lengths: np.ndarray,
    color: str = "#444444",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    label: Optional[str] = None,
) -> None:
    """Plot road edge polylines.

    Args:
        ax: Matplotlib axes.
        polylines_x, polylines_y: Flattened point coordinates.
        lengths: Number of points per polyline.
        color: Line color.
        linewidth: Line width.
        alpha: Transparency.
        label: Legend label.
    """
    segments = []
    idx = 0

    for length in lengths:
        if length < 2:
            idx += length
            continue

        x = polylines_x[idx : idx + length]
        y = polylines_y[idx : idx + length]

        # Create segments for this polyline
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        segments.extend(segs)

        idx += length

    if segments:
        lc = LineCollection(segments, colors=color, linewidth=linewidth, alpha=alpha, label=label)
        ax.add_collection(lc)


def plot_goals(
    ax: matplotlib.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    radius: float = 2.0,
    color: str = "#00CC00",
    alpha: float = 0.5,
    label: Optional[str] = None,
) -> None:
    """Plot goal positions as circles.

    Args:
        ax: Matplotlib axes.
        x, y: Goal positions.
        radius: Goal radius.
        color: Circle color.
        alpha: Transparency.
        label: Legend label.
    """
    for i, (gx, gy) in enumerate(zip(x, y)):
        circle = Circle((gx, gy), radius, fill=True, facecolor=color, edgecolor=color, alpha=alpha, label=label if i == 0 else None)
        ax.add_patch(circle)


def plot_observation_features(
    ax: matplotlib.axes.Axes,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
    partner_obs: np.ndarray,
    road_obs: np.ndarray,
    obs_radius: float = 50.0,
    ego_color: str = "#0066FF",
    partner_color: str = "#FF884D",
    road_color: str = "#888888",
) -> None:
    """Visualize observation features from ego-centric view.

    Args:
        ax: Matplotlib axes.
        ego_x, ego_y: Ego position.
        ego_heading: Ego heading in radians.
        partner_obs: Partner observations (N, 7) [rel_x, rel_y, width, length, heading_x, heading_y, speed].
        road_obs: Road observations (M, features).
        obs_radius: Observation radius for display.
        ego_color, partner_color, road_color: Colors for different elements.
    """
    # Draw observation radius circle
    circle = Circle((ego_x, ego_y), obs_radius, fill=False, edgecolor=ego_color, linestyle="--", alpha=0.3)
    ax.add_patch(circle)

    # Draw ego marker
    ax.plot(ego_x, ego_y, "o", color=ego_color, markersize=10, zorder=10, label="Ego")

    # Transform partner observations from ego frame to world frame
    c, s = np.cos(ego_heading), np.sin(ego_heading)

    for i in range(len(partner_obs)):
        rel_x, rel_y = partner_obs[i, 0], partner_obs[i, 1]

        # Skip if too far (indicates invalid/padding)
        if abs(rel_x) > 100 or abs(rel_y) > 100:
            continue

        # Rotate from ego frame to world frame
        world_x = ego_x + rel_x * c - rel_y * s
        world_y = ego_y + rel_x * s + rel_y * c

        ax.plot(world_x, world_y, "s", color=partner_color, markersize=6, alpha=0.7)

        # Draw line from ego to partner
        ax.plot([ego_x, world_x], [ego_y, world_y], "-", color=partner_color, alpha=0.3, linewidth=1)


def create_figure(
    figsize: Tuple[float, float] = (10, 10),
    dpi: int = 100,
    dark_background: bool = False,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Create a figure with standard settings.

    Args:
        figsize: Figure size in inches.
        dpi: Resolution.
        dark_background: If True, use dark background style.

    Returns:
        (figure, axes) tuple.
    """
    if dark_background:
        plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")

    return fig, ax


def compute_viewport(
    x: np.ndarray,
    y: np.ndarray,
    padding: float = 10.0,
    center: Optional[Tuple[float, float]] = None,
    radius: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """Compute viewport bounds for visualization.

    Args:
        x, y: Point coordinates to include.
        padding: Extra padding around bounds.
        center: If provided, center viewport here.
        radius: If provided with center, use fixed radius.

    Returns:
        (x_min, x_max, y_min, y_max) bounds.
    """
    if center is not None and radius is not None:
        cx, cy = center
        return cx - radius, cx + radius, cy - radius, cy + radius

    if len(x) == 0 or len(y) == 0:
        return -100, 100, -100, 100

    x_min, x_max = np.min(x) - padding, np.max(x) + padding
    y_min, y_max = np.min(y) - padding, np.max(y) + padding

    # Make square
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    return x_center - max_range / 2, x_center + max_range / 2, y_center - max_range / 2, y_center + max_range / 2


def save_frames_as_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 10,
) -> None:
    """Save list of frames as video using imageio.

    Args:
        frames: List of [H, W, 3] uint8 arrays.
        output_path: Output video path.
        fps: Frames per second.
    """
    import imageio

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)


def save_frames_as_gif(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 10,
    loop: int = 0,
) -> None:
    """Save list of frames as animated GIF.

    Args:
        frames: List of [H, W, 3] uint8 arrays.
        output_path: Output GIF path.
        fps: Frames per second.
        loop: Number of loops (0=infinite).
    """
    duration = int(1000 / fps)  # milliseconds per frame
    images = [Image.fromarray(frame) for frame in frames]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=loop)
