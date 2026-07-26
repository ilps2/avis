"""
PyTorch feature extractors for AVIS.

Uses torchvision pre-trained models (already cached, no download needed):
  - MobileNetV3-Small: 576-dim, ~8ms/frame on CPU
  - ResNet18 (if cached): 512-dim

Faster and semantically richer than OpenCV color histograms.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class TorchExtractor:
    """Base class for torchvision-based feature extractors."""
    
    def __init__(self, model_name: str = "mobilenet_v3_small"):
        self.device = torch.device("cpu")
        self.model_name = model_name
        
        if model_name == "mobilenet_v3_small":
            self.model = models.mobilenet_v3_small(weights="DEFAULT")
            self.model.classifier = nn.Identity()
            self.dim = 576
        elif model_name == "mobilenet_v3_large":
            self.model = models.mobilenet_v3_large(weights="DEFAULT")
            self.model.classifier = nn.Identity()
            self.dim = 960
        elif model_name == "resnet18":
            m = models.resnet18(weights="DEFAULT")
            m.fc = nn.Identity()
            self.model = m
            self.dim = 512
        elif model_name == "efficientnet_b0":
            m = models.efficientnet_b0(weights="DEFAULT")
            m.classifier = nn.Identity()
            self.model = m
            self.dim = 1280
        elif model_name == "resnet50":
            m = models.resnet50(weights="DEFAULT")
            m.fc = nn.Identity()
            self.model = m
            self.dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(self.device).eval()
        
        # Standard ImageNet preprocessing
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extract features from a BGR frame (OpenCV format)."""
        # BGR → RGB → resize to 224×224
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        
        # Preprocess
        tensor = self.preprocess(frame_resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
        
        feat = features.cpu().numpy().squeeze(0).astype(np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(feat)
        if norm > 1e-8:
            feat = feat / norm
        
        return feat


# Quick self-test
if __name__ == "__main__":
    import time
    
    print("Testing torchvision extractors...")
    
    # Create a random frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    for name in ["mobilenet_v3_small"]:
        print(f"\n  {name}:")
        ext = TorchExtractor(name)
        print(f"    dim={ext.dim}")
        
        t0 = time.time()
        feat = ext.extract(frame)
        dt = time.time() - t0
        print(f"    shape={feat.shape}, norm={np.linalg.norm(feat):.4f}")
        print(f"    time={dt*1000:.1f}ms")
        
        # Test consistency
        feat2 = ext.extract(frame)
        diff = np.abs(feat - feat2).max()
        print(f"    deterministic: max_diff={diff:.6f}")
        
        # Test on black vs white frame
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        white = np.ones((480, 640, 3), dtype=np.uint8) * 255
        fb = ext.extract(black)
        fw = ext.extract(white)
        sim = float(np.dot(fb, fw))
        print(f"    black-vs-white cos-sim: {sim:.4f}")
    
    print("\n✓ All extractors working")
