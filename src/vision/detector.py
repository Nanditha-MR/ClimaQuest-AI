"""
ClimaQuest AI — Vision Module: Storm Cell Detection & Segmentation
Implements multi-threshold hierarchical segmentation and morphological decomposition
for convective storm cells and precipitation envelopes (inspired by TITAN / SCIT).
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2


class DetectedStormCell:
    """Represents a segmented storm cell candidate in a single frame."""

    def __init__(
        self,
        cell_id_candidate: int,
        mask: np.ndarray,
        contour: np.ndarray,
        envelope_threshold: float,
        core_threshold: float,
        core_mask: Optional[np.ndarray] = None,
    ):
        self.candidate_id = cell_id_candidate
        self.mask = mask  # Binary mask uint8 (0 or 255)
        self.contour = contour  # OpenCV contour (N, 1, 2)
        self.envelope_threshold = envelope_threshold
        self.core_threshold = core_threshold
        self.core_mask = core_mask  # Mask of the severe convective core inside envelope


class StormDetector:
    """
    Advanced meteorological storm detector using multi-level thresholding,
    adaptive morphology, and convective core isolation.
    """

    def __init__(
        self,
        envelope_threshold: float = 25.0,  # e.g., 25 dBZ / 5 mm/h (rain shield)
        core_threshold: float = 40.0,      # e.g., 40 dBZ / 20 mm/h (convective core)
        min_cell_area_px: int = 25,        # Filter out minor clutter/noise
        kernel_size: int = 5,
        enable_watershed_split: bool = True,
    ):
        self.envelope_threshold = envelope_threshold
        self.core_threshold = core_threshold
        self.min_cell_area_px = min_cell_area_px
        self.kernel_size = kernel_size
        self.enable_watershed_split = enable_watershed_split

        # Morphological structuring elements
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )

    def detect(self, grid: np.ndarray) -> List[DetectedStormCell]:
        """
        Detects distinct storm cells and their convective cores in a 2D physical grid.

        Args:
            grid: 2D numpy array of physical values (dBZ or mm/h).

        Returns:
            List of DetectedStormCell objects.
        """
        if grid.ndim != 2:
            raise ValueError(f"Expected 2D grid, got shape {grid.shape}")

        # Step 1: Pre-filter grid to reduce high-frequency speckle noise
        smoothed = cv2.GaussianBlur(grid, (self.kernel_size, self.kernel_size), 0)

        # Step 2: Binary envelope segmentation
        _, envelope_raw = cv2.threshold(
            smoothed, self.envelope_threshold, 255, cv2.THRESH_BINARY
        )
        envelope_raw = envelope_raw.astype(np.uint8)

        # Step 3: Morphological opening (remove noise) and closing (fill internal voids)
        envelope_clean = cv2.morphologyEx(envelope_raw, cv2.MORPH_OPEN, self.morph_kernel)
        envelope_clean = cv2.morphologyEx(envelope_clean, cv2.MORPH_CLOSE, self.morph_kernel)

        # Step 4: Convective core segmentation
        _, core_raw = cv2.threshold(
            smoothed, self.core_threshold, 255, cv2.THRESH_BINARY
        )
        core_clean = cv2.morphologyEx(core_raw.astype(np.uint8), cv2.MORPH_OPEN, self.morph_kernel)

        # Step 5: Cell separation via Connected Components or Watershed
        if self.enable_watershed_split and np.sum(core_clean) > 0:
            cells = self._watershed_segmentation(smoothed, envelope_clean, core_clean)
        else:
            cells = self._contour_segmentation(smoothed, envelope_clean, core_clean)

        # Filter by minimum area
        valid_cells = [
            c for c in cells if cv2.contourArea(c.contour) >= self.min_cell_area_px
        ]

        return valid_cells

    def _contour_segmentation(
        self, smoothed: np.ndarray, envelope_mask: np.ndarray, core_mask: np.ndarray
    ) -> List[DetectedStormCell]:
        """Standard contour-based object extraction."""
        contours, _ = cv2.findContours(
            envelope_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < self.min_cell_area_px:
                continue

            cell_mask = np.zeros_like(envelope_mask)
            cv2.drawContours(cell_mask, [cnt], -1, 255, thickness=cv2.FILLED)

            # Isolate convective core inside this specific cell
            cell_core = cv2.bitwise_and(core_mask, core_mask, mask=cell_mask)

            detected.append(
                DetectedStormCell(
                    cell_id_candidate=idx + 1,
                    mask=cell_mask,
                    contour=cnt,
                    envelope_threshold=self.envelope_threshold,
                    core_threshold=self.core_threshold,
                    core_mask=cell_core,
                )
            )

        return detected

    def _watershed_segmentation(
        self, smoothed: np.ndarray, envelope_mask: np.ndarray, core_mask: np.ndarray
    ) -> List[DetectedStormCell]:
        """
        Distance-transform & watershed segmentation to cleanly split coalescing storms
        that share a diffuse envelope but possess separate convective updraft cores.
        """
        # Marker determination via core connected components or distance transform
        dist_transform = cv2.distanceTransform(envelope_mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        # Combine with severe core if available
        markers_seed = cv2.bitwise_or(sure_fg, core_mask)
        num_labels, markers = cv2.connectedComponents(markers_seed)

        if num_labels <= 1:
            # Fall back to contour if no distinct centers found
            return self._contour_segmentation(smoothed, envelope_mask, core_mask)

        # Watershed expects 3-channel pseudo image
        norm_img = cv2.normalize(smoothed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        pseudo_bgr = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)

        # Markers: 0 = unknown, 1..N = identified objects
        markers = markers + 1
        markers[envelope_mask == 0] = 0

        cv2.watershed(pseudo_bgr, markers)

        detected = []
        candidate_id = 1
        for label_val in range(2, num_labels + 2):
            cell_mask = np.where(markers == label_val, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(
                cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                if cv2.contourArea(cnt) >= self.min_cell_area_px:
                    cell_core = cv2.bitwise_and(core_mask, core_mask, mask=cell_mask)
                    detected.append(
                        DetectedStormCell(
                            cell_id_candidate=candidate_id,
                            mask=cell_mask,
                            contour=cnt,
                            envelope_threshold=self.envelope_threshold,
                            core_threshold=self.core_threshold,
                            core_mask=cell_core,
                        )
                    )
                    candidate_id += 1

        return detected
