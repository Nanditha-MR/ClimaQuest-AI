"""
ClimaQuest AI — Data Module: Realistic Meteorological Radar / Satellite Simulator
Generates physically inspired 2D precipitation / reflectivity grids (dBZ / mm/h)
simulating convective cloud systems with advection, shear, growth, and splitting.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class ConvectiveCellState:
    """Represents a simulated convective storm cell."""
    def __init__(
        self,
        cell_id: int,
        x: float,
        y: float,
        peak_intensity: float,
        sigma_x: float,
        sigma_y: float,
        angle_deg: float,
        vx: float,
        vy: float,
        growth_rate: float = 0.0,
        intensity_growth: float = 0.0,
    ):
        self.cell_id = cell_id
        self.x = x
        self.y = y
        self.peak_intensity = peak_intensity  # e.g., 30 - 65 dBZ (or mm/h)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.angle_deg = angle_deg
        self.vx = vx  # pixels per frame
        self.vy = vy
        self.growth_rate = growth_rate
        self.intensity_growth = intensity_growth

    def step(self, dt: float = 1.0):
        """Advance cell state in time."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.sigma_x = max(3.0, self.sigma_x + self.growth_rate * dt)
        self.sigma_y = max(3.0, self.sigma_y + self.growth_rate * dt * 0.8)
        self.peak_intensity = max(10.0, min(75.0, self.peak_intensity + self.intensity_growth * dt))


class MeteorologicalSequenceGenerator:
    """
    Simulates a sequence of meteorological radar reflectivity (dBZ) or
    satellite-derived precipitation fields with real physical phenomena.
    """

    def __init__(
        self,
        height: int = 400,
        width: int = 400,
        lon_min: float = 72.0,
        lon_max: float = 80.0,
        lat_min: float = 12.0,
        lat_max: float = 20.0,
        seed: int = 42,
    ):
        self.height = height
        self.width = width
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.rng = np.random.RandomState(seed)

        # Coordinate meshgrid
        y_coords = np.arange(height)
        x_coords = np.arange(width)
        self.X, self.Y = np.meshgrid(x_coords, y_coords)

    def generate_scenario(self, num_frames: int = 15) -> List[Dict[str, Any]]:
        """
        Generates a multi-frame meteorological sequence featuring:
        - Storm Alpha: A severe supercell developing and moving northeast.
        - Storm Beta: A moderate convective cluster with wind shear elongation.
        - Storm Gamma: A fast-moving squall line fragment appearing midway.
        """
        cells: List[ConvectiveCellState] = [
            ConvectiveCellState(
                cell_id=1,
                x=80.0,
                y=120.0,
                peak_intensity=52.0,  # Severe core
                sigma_x=16.0,
                sigma_y=12.0,
                angle_deg=35.0,
                vx=4.5,
                vy=3.0,
                growth_rate=0.4,
                intensity_growth=0.8,
            ),
            ConvectiveCellState(
                cell_id=2,
                x=260.0,
                y=280.0,
                peak_intensity=44.0,  # Moderate multicell
                sigma_x=22.0,
                sigma_y=15.0,
                angle_deg=-20.0,
                vx=-2.0,
                vy=1.8,
                growth_rate=0.1,
                intensity_growth=-0.4,
            ),
        ]

        frames: List[Dict[str, Any]] = []

        for t in range(num_frames):
            # Dynamic event at t=6: Storm 3 (squall genesis)
            if t == 6:
                cells.append(
                    ConvectiveCellState(
                        cell_id=3,
                        x=140.0,
                        y=320.0,
                        peak_intensity=40.0,
                        sigma_x=10.0,
                        sigma_y=8.0,
                        angle_deg=10.0,
                        vx=3.2,
                        vy=-3.5,
                        growth_rate=0.7,
                        intensity_growth=1.2,
                    )
                )

            # Base background noise (diffuse atmospheric moisture / clutter)
            grid = self.rng.gamma(shape=1.2, scale=1.5, size=(self.height, self.width))

            # Synthesize each cell as anisotropic 2D rotated Gaussian core + anvil
            for cell in cells:
                theta = np.radians(cell.angle_deg)
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)

                dx = self.X - cell.x
                dy = self.Y - cell.y

                x_rot = cos_t * dx + sin_t * dy
                y_rot = -sin_t * dx + cos_t * dy

                # Anvil cloud envelope (low intensity, broad coverage)
                anvil = (cell.peak_intensity * 0.45) * np.exp(
                    -0.5 * ((x_rot / (cell.sigma_x * 2.2)) ** 2 + (y_rot / (cell.sigma_y * 2.2)) ** 2)
                )

                # Convective core (high intensity, sharp gradient)
                core = (cell.peak_intensity * 0.55) * np.exp(
                    -0.5 * ((x_rot / cell.sigma_x) ** 2 + (y_rot / cell.sigma_y) ** 2)
                )

                cell_field = anvil + core
                grid = np.maximum(grid, cell_field)

                # Advance cell dynamics
                cell.step(dt=1.0)

            # Clip to standard meteorological radar bounds (0 to 75 dBZ)
            grid = np.clip(grid, 0.0, 75.0).astype(np.float32)

            # Timestamp simulation (e.g. 15-minute observation cycles)
            timestamp_str = f"2026-10-19T{10 + (t * 15) // 60:02d}:{(t * 15) % 60:02d}:00Z"

            frames.append({
                "frame_idx": t,
                "timestamp": timestamp_str,
                "grid": grid,
                "unit": "dBZ",
                "bounds": {
                    "lon_min": self.lon_min,
                    "lon_max": self.lon_max,
                    "lat_min": self.lat_min,
                    "lat_max": self.lat_max,
                },
                "resolution_km": 2.5,
            })

        return frames
