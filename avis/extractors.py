"""
Lightweight feature extractors for AVIS — no CLIP/GPU needed.

Uses OpenCV-based features that run in <1ms per frame on CPU:
  - ColorHistogram: HSV color distribution (robust to lighting)
  - EdgeHistogram:  Canny edge orientation histogram (shape descriptor)
  - Combined:       concatenation of both (~200-dim)

These are NOT as semantically rich as CLIP, but they:
  - Run instantly on CPU
  - Are invariant to small lighting changes
  - Capture enough to distinguish scenes/shapes
  - Perfect for demonstrating the Feature-Stream format
"""

import numpy as np
import cv2


class ColorHistogram:
    """HSV color histogram feature extractor."""
    
    def __init__(self, h_bins=18, s_bins=12, v_bins=8):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.dim = h_bins + s_bins + v_bins
        self._bins = [h_bins, s_bins, v_bins]
        self._ranges = [0, 180, 0, 256, 0, 256]
    
    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # Compute per-channel histograms
        features = []
        channels = [0, 1, 2]  # H, S, V
        for ch, nbins in zip(channels, self._bins):
            hist = cv2.calcHist([hsv], [ch], None, [nbins], 
                               self._ranges[ch*2:ch*2+2])
            hist = cv2.normalize(hist, hist).flatten()
            features.append(hist)
        
        feat = np.concatenate(features).astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(feat)
        if norm > 1e-8:
            feat = feat / norm
        return feat


class EdgeHistogram:
    """Edge orientation histogram using Canny + Sobel."""
    
    def __init__(self, n_bins=36, canny_low=50, canny_high=150):
        self.n_bins = n_bins
        self.dim = n_bins
        self.canny_low = canny_low
        self.canny_high = canny_high
    
    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi % 180
        
        # Canny edge mask (only count strong edges)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # Build orientation histogram over edge pixels
        hist = np.zeros(self.n_bins, dtype=np.float64)
        edge_mask = edges > 0
        if edge_mask.sum() == 0:
            return np.zeros(self.dim, dtype=np.float32)
        
        angles = angle[edge_mask]
        weights = mag[edge_mask]
        
        bin_width = 180.0 / self.n_bins
        for a, w in zip(angles, weights):
            b = int(a / bin_width) % self.n_bins
            hist[b] += w
        
        hist = hist / (hist.sum() + 1e-8)
        feat = hist.astype(np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(feat)
        if norm > 1e-8:
            feat = feat / norm
        return feat


class CombinedExtractor:
    """Combined color + edge features (~200-dim, runs in <1ms/frame)."""
    
    def __init__(self):
        self.color = ColorHistogram(h_bins=18, s_bins=12, v_bins=8)
        self.edge = EdgeHistogram(n_bins=36)
        self.dim = self.color.dim + self.edge.dim
    
    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        c = self.color.extract(frame_bgr)
        e = self.edge.extract(frame_bgr)
        feat = np.concatenate([c, e]).astype(np.float32)
        # Re-normalize combined
        norm = np.linalg.norm(feat)
        if norm > 1e-8:
            feat = feat / norm
        return feat
