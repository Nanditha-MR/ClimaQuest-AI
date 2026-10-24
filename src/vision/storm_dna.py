"""
ClimaQuest AI — Vision Module: Storm DNA Fingerprinting Engine
Synthesizes multi-scale physical, geometric, kinematic, and textural features
into an 8-dimensional normalized fingerprint vector and categorical archetype.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from .features import StormCellFeatures
from .tracker import StormTrack
from .cnn_embedder import DeepStormEmbedder


class StormDNA:
    """
    The signature 8-dimensional normalized fingerprint vector of an extreme weather event:
    1. Scale (Coverage)
    2. Vigour (Core Intensity)
    3. Compactness (Solidity)
    4. Asymmetry (Eccentricity / Shear)
    5. Core Density (Core-to-Envelope Ratio)
    6. Kinematics (Ground Speed)
    7. Growth Momentum (Evolution Rate)
    8. Textural Gradient (Internal Turbulence / Spatial Complexity)
    """

    GENE_LABELS = [
        "Spatial Scale",
        "Convective Vigour",
        "Structural Compactness",
        "Shear Asymmetry",
        "Core Density",
        "Ground Speed",
        "Growth Momentum",
        "Textural Gradient",
    ]

    def __init__(
        self,
        vector: np.ndarray,
        archetype: str,
        raw_metrics: Dict[str, float],
        deep_embedding: Optional[np.ndarray] = None,
    ):
        self.vector = np.clip(vector, 0.0, 1.0).astype(np.float32)
        self.archetype = archetype
        self.raw_metrics = raw_metrics
        self.deep_embedding = deep_embedding

    def to_dict(self) -> Dict[str, Any]:
        """Serializes Storm DNA into structured JSON-compatible dictionary."""
        return {
            "archetype": self.archetype,
            "vector": [round(float(v), 4) for v in self.vector],
            "genes": {
                label: round(float(self.vector[i]), 4)
                for i, label in enumerate(self.GENE_LABELS)
            },
            "deep_embedding": [round(float(x), 4) for x in self.deep_embedding]
            if self.deep_embedding is not None
            else [],
            "raw_metrics": {k: round(float(v), 2) for k, v in self.raw_metrics.items()},
        }

    def distance_to(self, other: "StormDNA") -> float:
        """Euclidean distance in DNA space (0.0 = identical, 1.0+ = divergent)."""
        return float(np.linalg.norm(self.vector - other.vector))

    def similarity_to(self, other: "StormDNA") -> float:
        """Cosine similarity in DNA space [0.0 to 1.0]."""
        dot = np.dot(self.vector, other.vector)
        norm_a = np.linalg.norm(self.vector)
        norm_b = np.linalg.norm(other.vector)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(max(0.0, dot / (norm_a * norm_b)))


class StormDNAGenerator:
    """
    Computes normalized Storm DNA vectors and deep visual embeddings
    from detected features, track kinematics, and raw physical grids.
    """

    _embedder = DeepStormEmbedder(embedding_dim=16)

    @staticmethod
    def extract_dna(
        grid: np.ndarray,
        track: StormTrack,
        features: StormCellFeatures,
    ) -> StormDNA:
        """
        Extracts 8-dimensional normalized DNA vector and deep visual embedding from a tracked storm.
        """
        # Gene 1: Spatial Scale [0, 1] (0 to 10,000 km^2)
        g_scale = min(1.0, features.area_km2 / 10000.0)

        # Gene 2: Convective Vigour [0, 1] (25 to 75 dBZ)
        g_vigour = min(1.0, max(0.0, (features.max_intensity - 25.0) / 50.0))

        # Gene 3: Structural Compactness [0, 1] (Solidity)
        g_compact = min(1.0, max(0.0, features.solidity))

        # Gene 4: Shear Asymmetry [0, 1] (Eccentricity)
        g_asymmetry = min(1.0, max(0.0, features.eccentricity))

        # Gene 5: Core Density [0, 1] (Normalized core-to-envelope ratio)
        g_core = min(1.0, max(0.0, features.core_ratio * 3.0))

        # Gene 6: Ground Speed [0, 1] (0 to 100 km/h)
        g_speed = min(1.0, max(0.0, track.speed_kmh / 100.0))

        # Gene 7: Growth Momentum [0, 1] (-50% to +50% mapped to [0, 1])
        g_growth = min(1.0, max(0.0, 0.5 + (track.growth_rate_pct_hr / 100.0)))

        # Gene 8: Textural Gradient [0, 1] (Laplacian gradient of intensity within mask)
        g_texture = StormDNAGenerator._compute_texture_entropy(grid, track.latest_mask)

        dna_vector = np.array(
            [
                g_scale,
                g_vigour,
                g_compact,
                g_asymmetry,
                g_core,
                g_speed,
                g_growth,
                g_texture,
            ],
            dtype=np.float32,
        )

        archetype = StormDNAGenerator._classify_archetype(dna_vector, features, track)

        # Extract 16-dimensional deep visual structural embedding
        deep_emb = StormDNAGenerator._embedder.extract_embedding(grid, track.latest_mask)

        raw_metrics = {
            "area_km2": features.area_km2,
            "max_intensity": features.max_intensity,
            "mean_intensity": features.mean_intensity,
            "speed_kmh": track.speed_kmh,
            "bearing_deg": track.bearing_deg,
            "growth_rate_pct_hr": track.growth_rate_pct_hr,
            "solidity": features.solidity,
            "core_ratio": features.core_ratio,
        }

        return StormDNA(dna_vector, archetype, raw_metrics, deep_embedding=deep_emb)

    @staticmethod
    def _compute_texture_entropy(grid: np.ndarray, mask: np.ndarray) -> float:
        """Measures spatial intensity turbulence / roughness using Laplacian operator."""
        if np.sum(mask) == 0:
            return 0.0

        # Sub-crop bounding region
        y_idx, x_idx = np.where(mask > 0)
        y0, y1 = max(0, y_idx.min() - 2), min(grid.shape[0], y_idx.max() + 3)
        x0, x1 = max(0, x_idx.min() - 2), min(grid.shape[1], x_idx.max() + 3)

        sub_grid = grid[y0:y1, x0:x1]
        laplacian = cv2.Laplacian(sub_grid, cv2.CV_32F)
        sub_mask = mask[y0:y1, x0:x1]

        valid_laplacian = laplacian[sub_mask > 0]
        if len(valid_laplacian) == 0:
            return 0.0

        var = float(np.var(valid_laplacian))
        # Log-scale normalization for variance [0, 1]
        norm_entropy = min(1.0, np.log1p(var) / 6.0)
        return float(norm_entropy)

    @staticmethod
    def _classify_archetype(
        v: np.ndarray, feat: StormCellFeatures, track: StormTrack
    ) -> str:
        """
        Classifies storm cell into recognized meteorological archetypes.
        """
        g_scale, g_vigour, g_compact, g_asymmetry, g_core, g_speed, g_growth, _ = v

        if g_vigour >= 0.65 and g_core >= 0.4 and g_compact >= 0.7:
            return "Severe Supercell"
        elif g_asymmetry >= 0.7 and g_speed >= 0.35:
            return "Squall Line / Linear Multicell"
        elif g_scale >= 0.6 and feat.area_km2 >= 3000.0:
            return "Mesoscale Convective System (MCS)"
        elif g_growth >= 0.7 and feat.area_km2 < 1500.0:
            return "Rapidly Intensifying Pulse Cell"
        elif g_growth <= 0.3 and g_core <= 0.15:
            return "Decaying Stratiform Shield"
        else:
            return "Isolated Convective Cluster"
