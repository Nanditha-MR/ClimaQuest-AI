"""
ClimaQuest AI — Vision Module: Morphological, Geometric & Radiometric Feature Extraction
Computes physically meaningful metrics from detected storm cell masks and raw physical grids.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2


class StormCellFeatures:
    """Encapsulates all extracted spatial, geometric, and radiometric attributes."""

    def __init__(self, **kwargs):
        self.attributes: Dict[str, Any] = kwargs

    def to_dict(self) -> Dict[str, Any]:
        return self.attributes

    def __getattr__(self, name: str) -> Any:
        if name in self.attributes:
            return self.attributes[name]
        raise AttributeError(f"'StormCellFeatures' has no attribute '{name}'")


class FeatureExtractor:
    """
    Extracts geometric, morphological, intensity, and georeferenced features
    from segmented storm regions and underlying physical grids.
    """

    def __init__(self, resolution_km: float = 2.5):
        self.resolution_km = resolution_km
        self.pixel_area_km2 = resolution_km * resolution_km

    def extract(
        self,
        grid: np.ndarray,
        mask: np.ndarray,
        contour: np.ndarray,
        core_mask: Optional[np.ndarray] = None,
        bounds: Optional[Dict[str, float]] = None,
    ) -> StormCellFeatures:
        """
        Extracts comprehensive storm features.
        """
        # Geometric Moments
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w / 2.0, y + h / 2.0

        # Intensity-weighted centroid (Center of Mass / Updraft core)
        cell_pixels = grid[mask > 0]
        y_indices, x_indices = np.where(mask > 0)
        total_intensity = np.sum(cell_pixels)

        if total_intensity > 0:
            cx_weighted = float(np.sum(x_indices * cell_pixels) / total_intensity)
            cy_weighted = float(np.sum(y_indices * cell_pixels) / total_intensity)
        else:
            cx_weighted, cy_weighted = float(cx), float(cy)

        # Spatial geometry
        area_px = float(cv2.contourArea(contour))
        area_km2 = area_px * self.pixel_area_km2
        perimeter_px = float(cv2.arcLength(contour, closed=True))
        perimeter_km = perimeter_px * self.resolution_km

        # Bounding boxes
        bx, by, bw, bh = cv2.boundingRect(contour)

        # Convex Hull & Solidity
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(area_px / hull_area) if hull_area > 0 else 1.0

        # Circularity (Isoperimetric Quotient: 1.0 = perfect circle)
        circularity = (
            float(4.0 * np.pi * area_px / (perimeter_px**2))
            if perimeter_px > 0
            else 0.0
        )
        circularity = min(1.0, max(0.0, circularity))

        # Fitted Ellipse / Axis lengths & Eccentricity
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center_e, (d_minor, d_major), orientation_deg) = ellipse
            major_axis_km = float(d_major * self.resolution_km)
            minor_axis_km = float(d_minor * self.resolution_km)
            aspect_ratio = float(d_major / d_minor) if d_minor > 0 else 1.0
            eccentricity = (
                float(np.sqrt(1.0 - (d_minor / d_major) ** 2))
                if d_major >= d_minor and d_major > 0
                else 0.0
            )
        else:
            major_axis_km = float(max(bw, bh) * self.resolution_km)
            minor_axis_km = float(min(bw, bh) * self.resolution_km)
            aspect_ratio = float(bw / bh) if bh > 0 else 1.0
            eccentricity = 0.0
            orientation_deg = 0.0

        # Radiometric Intensity Statistics
        max_intensity = float(np.max(cell_pixels)) if len(cell_pixels) > 0 else 0.0
        mean_intensity = float(np.mean(cell_pixels)) if len(cell_pixels) > 0 else 0.0
        std_intensity = float(np.std(cell_pixels)) if len(cell_pixels) > 0 else 0.0
        p90_intensity = (
            float(np.percentile(cell_pixels, 90)) if len(cell_pixels) > 0 else 0.0
        )
        volumetric_flux = float(np.sum(cell_pixels) * self.pixel_area_km2)

        # Convective Core Metrics
        if core_mask is not None and np.sum(core_mask) > 0:
            core_area_px = float(np.sum(core_mask > 0))
            core_area_km2 = core_area_px * self.pixel_area_km2
            core_ratio = float(core_area_px / area_px) if area_px > 0 else 0.0
        else:
            core_area_km2 = 0.0
            core_ratio = 0.0

        # Georeferencing lon/lat
        lon, lat = None, None
        if bounds is not None:
            h_grid, w_grid = grid.shape
            lon = bounds["lon_min"] + (cx / (w_grid - 1)) * (
                bounds["lon_max"] - bounds["lon_min"]
            )
            lat = bounds["lat_max"] - (cy / (h_grid - 1)) * (
                bounds["lat_max"] - bounds["lat_min"]
            )

        return StormCellFeatures(
            centroid_px=(float(cx), float(cy)),
            centroid_weighted_px=(cx_weighted, cy_weighted),
            lon=lon,
            lat=lat,
            area_px=area_px,
            area_km2=area_km2,
            perimeter_km=perimeter_km,
            bbox_px=(int(bx), int(by), int(bw), int(bh)),
            solidity=solidity,
            circularity=circularity,
            eccentricity=eccentricity,
            aspect_ratio=aspect_ratio,
            orientation_deg=float(orientation_deg),
            major_axis_km=major_axis_km,
            minor_axis_km=minor_axis_km,
            max_intensity=max_intensity,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            p90_intensity=p90_intensity,
            volumetric_flux=volumetric_flux,
            core_area_km2=core_area_km2,
            core_ratio=core_ratio,
        )
