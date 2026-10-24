"""
ClimaQuest AI — Visualization Module: Mission Control Storm & DNA Renderer
Generates dark scientific mission-control frames with storm boundaries,
motion vector fields, historical track paths, and embedded Storm DNA radar charts.
"""

from typing import List, Dict, Any, Optional
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from ..vision.tracker import StormTrack
from ..vision.storm_dna import StormDNA


class MissionControlRenderer:
    """
    Renders high-contrast dark scientific mission control frames for ClimaQuest AI.
    Palette: Dark Navy (#080d1a), Cyan (#00f5d4), Amber (#ffbe0b), Magenta (#f72585).
    """

    COLORMAP_RADAR = [
        (0.0, (10, 15, 30)),     # Dark background
        (15.0, (20, 45, 90)),    # Light rain envelope
        (25.0, (0, 180, 216)),   # Moderate precipitation (cyan)
        (35.0, (0, 245, 160)),   # Green
        (45.0, (255, 190, 11)),  # Yellow (convective)
        (55.0, (255, 87, 51)),   # Red / Orange (severe)
        (65.0, (247, 37, 133)),  # Magenta (violent / hail)
        (75.0, (255, 255, 255)), # White core
    ]

    @staticmethod
    def render_frame(
        grid: np.ndarray,
        tracks: List[StormTrack],
        dna_map: Dict[int, StormDNA],
        timestamp: str,
        frame_idx: int,
        output_path: Optional[str] = None,
        primary_track_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Renders a comprehensive scientific visualization frame.
        """
        fig = plt.figure(figsize=(12, 8), dpi=150, facecolor="#080e1e")
        gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.0])

        # 1. Main Meteorological Display (Left)
        ax_map = fig.add_subplot(gs[:, 0])
        ax_map.set_facecolor("#060a14")

        # Custom meteorological radar colormap
        cmap = matplotlib.colormaps["turbo"]
        im = ax_map.imshow(
            grid,
            cmap=cmap,
            vmin=10.0,
            vmax=70.0,
            origin="upper",
            interpolation="bilinear",
        )

        # Draw storm cells, contours, vectors, and tracks
        for track in tracks:
            t_id = track.track_id
            latest_feat = track.history_features[-1]
            cx, cy = latest_feat.centroid_px

            is_primary = (t_id == primary_track_id) or (primary_track_id is None and t_id == tracks[0].track_id)
            color = "#00f5d4" if is_primary else "#ffbe0b"

            # Draw Historical Centroid Trail
            if len(track.history_centroids) > 1:
                pts = np.array(track.history_centroids)
                ax_map.plot(
                    pts[:, 0],
                    pts[:, 1],
                    color=color,
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.8,
                )
                ax_map.scatter(
                    pts[:-1, 0],
                    pts[:-1, 1],
                    color=color,
                    s=15,
                    alpha=0.5,
                )

            # Draw Current Centroid
            ax_map.scatter(
                cx,
                cy,
                color="#ffffff",
                edgecolors=color,
                s=70,
                linewidths=2,
                zorder=5,
            )

            # Draw Contour Boundary
            cnt = track.latest_contour.squeeze()
            if cnt.ndim == 2 and len(cnt) > 2:
                ax_map.plot(
                    cnt[:, 0],
                    cnt[:, 1],
                    color=color,
                    linewidth=1.8,
                    alpha=0.9,
                )

            # Draw Motion Vector Arrow (Velocity)
            if track.speed_kmh > 0:
                vx_px = track.state[2] * 4.0  # Scale for visibility
                vy_px = track.state[3] * 4.0
                ax_map.arrow(
                    cx,
                    cy,
                    vx_px,
                    vy_px,
                    head_width=8,
                    head_length=10,
                    fc="#ffffff",
                    ec=color,
                    linewidth=1.5,
                    length_includes_head=True,
                    zorder=6,
                )

            # Text Tag
            status_text = f"ID:{t_id} | {latest_feat.max_intensity:.0f} dBZ | {track.speed_kmh:.1f} km/h"
            ax_map.text(
                cx + 12,
                cy - 8,
                status_text,
                color="#ffffff",
                fontsize=8,
                weight="bold",
                bbox=dict(facecolor="#080e1e", edgecolor=color, alpha=0.85, boxstyle="round,pad=0.3"),
                zorder=7,
            )

        ax_map.set_title(
            f"CLIMAQUEST AI — RADAR SURVEILLANCE [{timestamp}]",
            color="#00f5d4",
            fontsize=11,
            pad=10,
            weight="bold",
        )
        ax_map.axis("off")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax_map, orientation="horizontal", fraction=0.045, pad=0.03)
        cbar.set_label("Reflectivity (dBZ) / Precipitation Core Vigour", color="#94a3b8", fontsize=8)
        cbar.ax.tick_params(labelsize=7, colors="#94a3b8")

        # 2. Top-Right: Storm DNA Radar / Spider Chart
        ax_dna = fig.add_subplot(gs[0, 1], polar=True)
        ax_dna.set_facecolor("#0b1329")

        selected_id = primary_track_id if primary_track_id else (tracks[0].track_id if tracks else None)
        selected_dna = dna_map.get(selected_id) if selected_id else None

        if selected_dna is not None:
            labels = selected_dna.GENE_LABELS
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]

            values = selected_dna.vector.tolist()
            values += values[:1]

            ax_dna.plot(angles, values, color="#00f5d4", linewidth=2, linestyle="solid")
            ax_dna.fill(angles, values, color="#00f5d4", alpha=0.25)
            ax_dna.set_xticks(angles[:-1])
            ax_dna.set_xticklabels(labels, color="#94a3b8", size=6.5)
            ax_dna.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax_dna.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], color="#64748b", size=5.5)
            ax_dna.grid(color="#1e293b", linewidth=0.8)
            ax_dna.set_title(
                f"STORM DNA PROFILE (ID: {selected_id})\n[{selected_dna.archetype}]",
                color="#00f5d4",
                size=9,
                weight="bold",
                pad=12,
            )
        else:
            ax_dna.text(0.5, 0.5, "NO ACTIVE STORM", color="#64748b", ha="center")
            ax_dna.axis("off")

        # 3. Bottom-Right: Telemetry Metrics Panel
        ax_stats = fig.add_subplot(gs[1, 1])
        ax_stats.set_facecolor("#0b1329")
        ax_stats.axis("off")

        if selected_dna is not None:
            raw = selected_dna.raw_metrics
            stats_text = (
                f"METEOROLOGICAL TELEMETRY\n"
                f"───────────────────────────────\n"
                f"Event Archetype:   {selected_dna.archetype}\n"
                f"Cell Area:         {raw['area_km2']:.1f} km²\n"
                f"Peak Reflectivity: {raw['max_intensity']:.1f} dBZ\n"
                f"Mean Reflectivity: {raw['mean_intensity']:.1f} dBZ\n"
                f"Ground Speed:      {raw['speed_kmh']:.1f} km/h\n"
                f"Heading (Bearing): {raw['bearing_deg']:.0f}°\n"
                f"Growth Momentum:   {raw['growth_rate_pct_hr']:+.1f}% / hr\n"
                f"Core-to-Shield:    {raw['core_ratio'] * 100:.1f}%\n"
                f"Boundary Solidity: {raw['solidity']:.2f}\n"
                f"───────────────────────────────\n"
                f"Engine 1 Status:   NOMINAL (Tracking)"
            )
            ax_stats.text(
                0.08,
                0.90,
                stats_text,
                color="#cbd5e1",
                fontfamily="monospace",
                fontsize=8.5,
                va="top",
                linespacing=1.4,
            )
            rect = patches.FancyBboxPatch(
                (0.02, 0.02),
                0.96,
                0.96,
                boxstyle="round,pad=0.03",
                edgecolor="#00f5d4",
                facecolor="none",
                linewidth=1.2,
                alpha=0.6,
            )
            ax_stats.add_patch(rect)

        plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05, wspace=0.15)

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")

        # Convert figure to numpy RGB array
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        plt.close(fig)

        return rgb
