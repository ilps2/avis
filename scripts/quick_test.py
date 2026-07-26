#!/usr/bin/env python3
"""Quick test: 2-second video, 60 frames."""
import sys, os, time, tempfile
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from avis import AVISEncoder, AVISDecoder, AVISQuery

# Generate short video
tmp = tempfile.mkdtemp(prefix="avis_")
video_path = f"{tmp}/test.mp4"
avis_path = f"{tmp}/test.avis"

w, h, fps = 320, 240, 30
nframes = 60
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

for i in range(nframes):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    t = i / fps
    if t < 1.0:
        frame[:] = (30, 30, 30)
        cx = int(50 + 200 * t)
        cv2.circle(frame, (cx, 120), 30, (255, 100, 0), -1)
    else:
        frame[:] = (60, 100, 60)
        cx = int(250 - 200 * (t - 1.0))
        cv2.circle(frame, (cx, 120), 30, (0, 200, 200), -1)
    writer.write(frame)
writer.release()
print(f"Video: {nframes} frames, {Path(video_path).stat().st_size/1024:.0f}KB")

# Encode
print("Encoding...")
t0 = time.time()
enc = AVISEncoder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", keyframe_interval=30, device="cpu")
stats = enc.encode(video_path, avis_path)
enc.close()
dt = time.time() - t0
print(f"Encoded in {dt:.1f}s")

# Stats
print(f"\nSource: {stats.source_size_bytes/1024:.0f}KB → AVIS: {stats.avis_size_bytes/1024:.0f}KB")
print(f"I-frames: {stats.i_frames}, Δ-frames: {stats.delta_frames}")
print(f"AVIS/source: {stats.compression_vs_raw*100:.1f}%")

# Decode
dec = AVISDecoder(avis_path)
feats = dec.all_features()
dec.close()
print(f"Decoded {len(feats)} features, dim={feats[0].shape[0]}")

# Validate
sim_01 = float(np.dot(feats[0], feats[1]))
sim_cross = float(np.dot(feats[5], feats[50]))
print(f"Same-scene cos-sim: {sim_01:.4f}, Cross-scene: {sim_cross:.4f}")

# Query
q = AVISQuery(avis_path)
similar = q.find_similar_frames(10, top_k=3)
print(f"Frames similar to #10: {similar}")
q.close()

# Token economics
total_trad = nframes * stats.latent_dim
total_avis = stats.i_frames * stats.latent_dim + int(stats.delta_frames * stats.latent_dim * 0.1)
print(f"\nToken reduction: {total_trad:,} → {total_avis:,} ({total_trad/total_avis:.1f}x)")

print(f"\nDone. AVIS: {avis_path}")
