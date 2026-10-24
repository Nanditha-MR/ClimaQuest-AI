"""
ClimaQuest AI — Vision Module: Spatiotemporal Multi-Object Storm Tracker
Kalman-assisted Hungarian bipartite matching for deformable convective cells,
computing velocity vectors, bearing, growth rates, and lifecycle events.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from .features import StormCellFeatures
from .detector import DetectedStormCell


class StormTrack:
    """Maintains the continuous spatiotemporal trajectory and lifecycle of a storm."""

    def __init__(self, track_id: int, initial_cell: DetectedStormCell, initial_features: StormCellFeatures, timestamp: str):
        self.track_id = track_id
        self.history_features: List[StormCellFeatures] = [initial_features]
        self.history_timestamps: List[str] = [timestamp]
        self.history_centroids: List[Tuple[float, float]] = [initial_features.centroid_px]
        self.latest_mask: np.ndarray = initial_cell.mask
        self.latest_contour: np.ndarray = initial_cell.contour
        self.missed_frames: int = 0
        self.age_frames: int = 1
        self.status: str = "GENESIS"  # GENESIS, INTENSIFYING, MATURE, DISSIPATING, TERMINATED

        # Kalman Filter state: [x, y, vx, vy]
        cx, cy = initial_features.centroid_px
        self.state = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)
        # Covariance matrix
        self.P = np.eye(4, dtype=np.float32) * 50.0

        # Motion & Trend metrics
        self.velocity_kmh = (0.0, 0.0)
        self.speed_kmh: float = 0.0
        self.bearing_deg: float = 0.0  # Meteorological bearing: 0 = North, 90 = East
        self.growth_rate_pct_hr: float = 0.0
        self.intensity_trend_dbz_hr: float = 0.0

    def predict_position(self, dt: float = 1.0) -> Tuple[float, float]:
        """Predicts centroid position using constant velocity model."""
        pred_x = self.state[0] + self.state[2] * dt
        pred_y = self.state[1] + self.state[3] * dt
        return float(pred_x), float(pred_y)

    def update(
        self,
        cell: DetectedStormCell,
        features: StormCellFeatures,
        timestamp: str,
        dt_hours: float = 0.25,
        resolution_km: float = 2.5,
    ):
        """Updates track with newly matched observation."""
        self.history_features.append(features)
        self.history_timestamps.append(timestamp)
        self.history_centroids.append(features.centroid_px)
        self.latest_mask = cell.mask
        self.latest_contour = cell.contour
        self.missed_frames = 0
        self.age_frames += 1

        # Kalman update: Measurement z = [cx, cy]
        cx, cy = features.centroid_px
        z = np.array([cx, cy], dtype=np.float32)

        # Measurement matrix H
        H = np.zeros((2, 4), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        # Measurement noise R & Process noise Q
        R = np.eye(2, dtype=np.float32) * 4.0
        Q = np.eye(4, dtype=np.float32) * 2.0

        # Prediction step
        F = np.eye(4, dtype=np.float32)
        F[0, 2] = 1.0
        F[1, 3] = 1.0
        x_pred = F @ self.state
        P_pred = F @ self.P @ F.T + Q

        # Kalman Gain
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        y_residual = z - (H @ x_pred)
        self.state = x_pred + K @ y_residual
        self.P = (np.eye(4, dtype=np.float32) - K @ H) @ P_pred

        # Kinematic calculations
        vx_px = self.state[2]
        vy_px = self.state[3]
        vx_kmh = (vx_px * resolution_km) / max(0.001, dt_hours)
        # Note: image y points southwards (down), so -vy corresponds to northward movement
        vy_kmh = (-vy_px * resolution_km) / max(0.001, dt_hours)
        self.velocity_kmh = (float(vx_kmh), float(vy_kmh))
        self.speed_kmh = float(np.sqrt(vx_kmh**2 + vy_kmh**2))

        # Meteorological bearing angle: 0 deg = North, 90 deg = East
        bearing = np.degrees(np.arctan2(vx_kmh, vy_kmh))
        self.bearing_deg = float(bearing % 360.0)

        # Growth Rate (% area change per hour)
        prev_area = self.history_features[-2].area_km2
        curr_area = features.area_km2
        if prev_area > 0 and dt_hours > 0:
            self.growth_rate_pct_hr = float(((curr_area - prev_area) / prev_area) / dt_hours * 100.0)
        else:
            self.growth_rate_pct_hr = 0.0

        # Intensity Trend (dBZ or mm/h change per hour)
        prev_max_i = self.history_features[-2].max_intensity
        curr_max_i = features.max_intensity
        self.intensity_trend_dbz_hr = float((curr_max_i - prev_max_i) / max(0.001, dt_hours))

        # Lifecycle Status Classification
        if self.growth_rate_pct_hr > 25.0 and self.intensity_trend_dbz_hr >= 0:
            self.status = "INTENSIFYING"
        elif features.core_ratio > 0.15 and abs(self.growth_rate_pct_hr) <= 25.0:
            self.status = "MATURE"
        elif self.growth_rate_pct_hr < -20.0 or features.core_ratio < 0.03:
            self.status = "DISSIPATING"
        else:
            self.status = "ACTIVE"


class StormTracker:
    """
    Tracks multiple convective storms across time sequences using Kalman-predicted
    spatial distance, morphological IoU, and intensity consistency.
    """

    def __init__(
        self,
        max_search_radius_px: float = 60.0,
        max_missed_frames: int = 2,
        dt_hours: float = 0.25,
        resolution_km: float = 2.5,
    ):
        self.max_search_radius_px = max_search_radius_px
        self.max_missed_frames = max_missed_frames
        self.dt_hours = dt_hours
        self.resolution_km = resolution_km
        self.active_tracks: List[StormTrack] = []
        self.next_track_id: int = 1

    def update_frame(
        self,
        candidates: List[DetectedStormCell],
        features_list: List[StormCellFeatures],
        timestamp: str,
    ) -> List[StormTrack]:
        """
        Updates tracks with candidates from a new frame.
        """
        if len(self.active_tracks) == 0:
            # Initialize tracks for all candidates in first frame
            for cell, feat in zip(candidates, features_list):
                track = StormTrack(self.next_track_id, cell, feat, timestamp)
                self.active_tracks.append(track)
                self.next_track_id += 1
            return self.active_tracks

        num_tracks = len(self.active_tracks)
        num_candidates = len(candidates)

        if num_candidates == 0:
            for track in self.active_tracks:
                track.missed_frames += 1
            self._prune_dead_tracks()
            return [t for t in self.active_tracks if t.missed_frames == 0]

        # Build Cost Matrix (Tracks x Candidates)
        cost_matrix = np.full((num_tracks, num_candidates), fill_value=1e5, dtype=np.float32)

        for t_idx, track in enumerate(self.active_tracks):
            pred_x, pred_y = track.predict_position(dt=1.0)
            track_last_feat = track.history_features[-1]

            for c_idx, cand_feat in enumerate(features_list):
                cand_cx, cand_cy = cand_feat.centroid_px
                dist = np.sqrt((pred_x - cand_cx) ** 2 + (pred_y - cand_cy) ** 2)

                if dist <= self.max_search_radius_px:
                    # Normalized spatial distance cost [0, 1]
                    dist_cost = dist / self.max_search_radius_px

                    # Morphological IoU cost
                    cand_cell = candidates[c_idx]
                    intersection = np.logical_and(track.latest_mask, cand_cell.mask).sum()
                    union = np.logical_or(track.latest_mask, cand_cell.mask).sum()
                    iou = intersection / union if union > 0 else 0.0
                    iou_cost = 1.0 - iou

                    # Intensity & Area consistency cost
                    delta_int = abs(track_last_feat.max_intensity - cand_feat.max_intensity) / 75.0
                    area_ratio = min(track_last_feat.area_px, cand_feat.area_px) / max(
                        track_last_feat.area_px, cand_feat.area_px
                    )
                    area_cost = 1.0 - area_ratio

                    # Weighted composite cost
                    composite_cost = (
                        0.45 * dist_cost
                        + 0.25 * iou_cost
                        + 0.15 * delta_int
                        + 0.15 * area_cost
                    )
                    cost_matrix[t_idx, c_idx] = composite_cost

        # Global optimal matching using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_candidates = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.85:  # Maximum allowable cost threshold
                self.active_tracks[r].update(
                    cell=candidates[c],
                    features=features_list[c],
                    timestamp=timestamp,
                    dt_hours=self.dt_hours,
                    resolution_km=self.resolution_km,
                )
                matched_tracks.add(r)
                matched_candidates.add(c)

        # Unmatched tracks count as missed
        for t_idx in range(num_tracks):
            if t_idx not in matched_tracks:
                self.active_tracks[t_idx].missed_frames += 1

        # Unmatched candidates become new tracks (Genesis)
        for c_idx in range(num_candidates):
            if c_idx not in matched_candidates:
                new_track = StormTrack(
                    track_id=self.next_track_id,
                    initial_cell=candidates[c_idx],
                    initial_features=features_list[c_idx],
                    timestamp=timestamp,
                )
                self.active_tracks.append(new_track)
                self.next_track_id += 1

        self._prune_dead_tracks()
        return [t for t in self.active_tracks if t.missed_frames == 0]

    def _prune_dead_tracks(self):
        """Marks tracks exceeding missed threshold as TERMINATED and removes them."""
        for track in self.active_tracks:
            if track.missed_frames > self.max_missed_frames:
                track.status = "TERMINATED"
        self.active_tracks = [t for t in self.active_tracks if t.missed_frames <= self.max_missed_frames]
