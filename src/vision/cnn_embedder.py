"""
ClimaQuest AI — Vision Module: Deep Visual Structural Embedder
Extracts deep spatial structural embeddings from cropped storm cell patches
to complement the 8 physical genes with a high-dimensional visual latent representation.
"""

from typing import List, Tuple
import numpy as np
import cv2


class DeepStormEmbedder:
    """
    Computes a 16-dimensional visual latent representation of storm cloud morphology
    (spiral band curvature, texture frequency, and internal structural density)
    using multi-scale convolutional spatial filters (mimicking early CNN filter banks).
    """

    def __init__(self, embedding_dim: int = 16, patch_size: Tuple[int, int] = (64, 64)):
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.filters = self._build_multiscale_filter_bank()

    def _build_multiscale_filter_bank(self) -> List[np.ndarray]:
        """Builds multi-orientation, multi-frequency convolutional filter kernels."""
        filters = []
        ksize = 9
        # 4 orientations (0, 45, 90, 135 deg) x 4 frequencies (wavelengths)
        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            for lambd in [4.0, 8.0, 12.0, 16.0]:
                kernel = cv2.getGaborKernel(
                    ksize=(ksize, ksize),
                    sigma=2.5,
                    theta=theta,
                    lambd=lambd,
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_32F,
                )
                kernel /= 1.5 * kernel.sum() if kernel.sum() != 0 else 1.0
                filters.append(kernel)
        return filters[: self.embedding_dim]

    def extract_embedding(self, grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extracts a 16-dimensional normalized visual embedding vector from a masked storm cell.
        """
        if np.sum(mask) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Crop bounding box of the storm
        y_idx, x_idx = np.where(mask > 0)
        y0, y1 = max(0, y_idx.min()), min(grid.shape[0], y_idx.max() + 1)
        x0, x1 = max(0, x_idx.min()), min(grid.shape[1], x_idx.max() + 1)

        patch = grid[y0:y1, x0:x1].copy()
        patch_mask = mask[y0:y1, x0:x1]

        # Zero out background
        patch[patch_mask == 0] = 0.0

        # Resize to fixed canonical patch size (64x64)
        resized = cv2.resize(patch, self.patch_size, interpolation=cv2.INTER_LINEAR)
        norm_patch = resized / 75.0  # Normalize physical range [0, 75 dBZ] to [0, 1]

        # Convolve with multi-scale filter bank and extract spatial pooling responses
        embedding = []
        for kernel in self.filters:
            filtered = cv2.filter2D(norm_patch, cv2.CV_32F, kernel)
            # Global Average Pooling (GAP) of response energy
            energy = float(np.mean(np.abs(filtered)))
            embedding.append(energy)

        embedding_vec = np.array(embedding, dtype=np.float32)

        # L2 Normalize the embedding vector
        norm = np.linalg.norm(embedding_vec)
        if norm > 0:
            embedding_vec /= norm

        return embedding_vec
