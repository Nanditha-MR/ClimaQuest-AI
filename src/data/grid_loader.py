"""
ClimaQuest AI — Data Module: Grid & Satellite Loader
Standardizes reading of weather radar, precipitation, and satellite IR fields.
"""

import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2


class MeteorologicalGridLoader:
    """
    Loads and calibrates meteorological 2D raster grids from numpy files or images.
    Converts raw radiometric pixels into physical units (dBZ or mm/h or Kelvin).
    """

    @staticmethod
    def load_numpy(filepath: str) -> np.ndarray:
        """Loads a 2D numpy array (.npy) containing calibrated physical values."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        data = np.load(filepath)
        if data.ndim == 3:
            data = data[0]  # Take first channel/time slice
        return data.astype(np.float32)

    @staticmethod
    def load_image_as_grid(
        image_path: str,
        val_min: float = 0.0,
        val_max: float = 75.0,
        invert: bool = False,
    ) -> np.ndarray:
        """
        Loads an 8-bit or 16-bit satellite/radar image and scales it linearly
        into physical units [val_min, val_max].
        
        For Satellite IR (e.g. Brightness Temperature):
        Cold cloud tops have low digital counts in raw IR or inverted scales.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to decode image: {image_path}")
        
        norm = img.astype(np.float32) / 255.0
        if invert:
            norm = 1.0 - norm
        
        physical_grid = val_min + norm * (val_max - val_min)
        return physical_grid

    @staticmethod
    def pixel_to_latlon(
        px: float, py: float, height: int, width: int, bounds: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Transforms pixel space (x, y) into geographic longitude and latitude.
        Assumes standard equirectangular projection over regional bounding box.
        """
        lon = bounds["lon_min"] + (px / (width - 1)) * (bounds["lon_max"] - bounds["lon_min"])
        # Pixel y=0 is typically top (lat_max), y=height-1 is bottom (lat_min)
        lat = bounds["lat_max"] - (py / (height - 1)) * (bounds["lat_max"] - bounds["lat_min"])
        return lon, lat
