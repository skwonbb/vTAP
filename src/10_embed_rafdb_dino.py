"""Embed RAF-DB with DINOv2 ViT-B/14."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "face" / "rafdb" / "rafdb_basic" / "Image" / "aligned"
OUT = ROOT / "cache" / "embeddings"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "facebook/dinov2-base"
BATCH = 64

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(dev).eval()

df = pd.read_csv(ROOT / "data" / "face" / "raf_labels.csv")
paths = [IMG / f"{s}_aligned.jpg" for s in df["stem"]]

vecs = []
with torch.no_grad():
    for i in range(0, len(paths), BATCH):
        imgs = [Image.open(p).convert("RGB") for p in paths[i : i + BATCH]]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
        out = model(pixel_values=px).last_hidden_state[:, 0]      
        vecs.append(out.float().cpu().numpy())
        if (i // BATCH) % 20 == 0:
            print(f"  {i + len(imgs):>6,}/{len(paths):,}", flush=True)

U = np.concatenate(vecs, 0)
np.save(OUT / "rafdb_dino.npy", U)
print(f"saved rafdb_dino.npy  shape={U.shape}")
