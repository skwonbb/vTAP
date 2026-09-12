"""Embed RAF-DB with CLIP ViT-L/14."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "face" / "rafdb" / "rafdb_basic" / "Image" / "aligned"
OUT = ROOT / "cache" / "embeddings"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "openai/clip-vit-large-patch14"
BATCH = 64

dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = CLIPImageProcessor.from_pretrained(MODEL)
model = CLIPVisionModelWithProjection.from_pretrained(MODEL).to(dev).eval()

df = pd.read_csv(ROOT / "data" / "face" / "raf_labels.csv")


paths = [IMG / f"{stem}_aligned.jpg" for stem in df["stem"]]
missing = [p for p in paths if not p.exists()]
if missing:
    raise SystemExit(f"missing {len(missing)} aligned images, e.g. {missing[0]}")

vecs = []
with torch.no_grad():
    for i in range(0, len(paths), BATCH):
        imgs = [Image.open(p).convert("RGB") for p in paths[i : i + BATCH]]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(dev)
        out = model(pixel_values=px).image_embeds     
        vecs.append(out.float().cpu().numpy())
        if (i // BATCH) % 20 == 0:
            print(f"  {i + len(imgs):>6,}/{len(paths):,}", flush=True)

U = np.concatenate(vecs, 0)
np.save(OUT / "rafdb_clip.npy", U)
print(f"saved rafdb_clip.npy  shape={U.shape}  dtype={U.dtype}")
