#!/usr/bin/env python3
"""Overlay perf-cgroup measurement fences on the service pipeline figure.
Non-destructive: reads docs/service_pipeline.png, writes docs/service_pipeline_fenced.png.
Fences = where perf attached (capture_tiers.sh SPECS): the FastAPI/orchestrator pod,
Milvus, MongoDB, SeaweedFS, the LLM-d inference-gateway pod, and the vLLM pod."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

# --- step 1 (PIL): clone a GPU cuboid into the free right end of the hardware band as the CPU box
pim = Image.open("docs/service_pipeline.png").convert("RGB")
CX0, CY0, CX1, CY1 = 505, 812, 648, 895      # first GPU cuboid incl. shadow
PX0 = 1180                                    # paste x (right end of band, inside the border)
tile = pim.crop((CX0, CY0, CX1, CY1))
pim.paste(tile, (PX0, CY0))
img = np.asarray(pim).astype(float) / 255.0
H, W = img.shape[0], img.shape[1]

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

RED = "#c0392b"
M = 7  # margin outside each card

def fence(x0, y0, x1, y1):
    ax.add_patch(FancyBboxPatch((x0 - M, y0 - M), (x1 - x0) + 2 * M, (y1 - y0) + 2 * M,
                                boxstyle="round,pad=2,rounding_size=8",
                                fc="none", ec=RED, lw=1.8, ls=(0, (4, 3)), zorder=10))

# (x0, y0, x1, y1) in image pixels — card outlines eyeballed from the 1448x1086 render
fence(445, 167, 1229, 273)    # FastAPI gateway + orchestrator: ONE pod (llm-service-kernel), one fence
fence(430, 406, 632, 513)     # Milvus
fence(683, 406, 914, 513)     # MongoDB
fence(966, 406, 1198, 513)    # SeaweedFS (filer + volume pods)
fence(410, 622, 614, 718)     # LLM-d Scheduler (inference-gateway pod)
fence(697, 621, 954, 718)     # Model-serving Workers (vLLM pod)

# --- step 2b: relabel the cloned cuboid CPU and drop a dashed runs-on line into it
OFF = PX0 - CX0
ax.add_patch(plt.Rectangle((546 + OFF, 838), 60, 38, fc="#f2f4f7", ec="none", zorder=10))
ax.text(573 + OFF, 858, "CPU", fontsize=15, fontweight="bold", color="#333333",
        ha="center", va="center", zorder=11)
cpu_cx = 573 + OFF
ax.add_patch(FancyArrowPatch((cpu_cx, 748), (cpu_cx, 812), arrowstyle="-|>", mutation_scale=13,
                             lw=1.6, color="#555555", ls=(0, (4, 3)), zorder=10))
ax.text(cpu_cx, 775, "all fenced pods", fontsize=10.5, ha="center", va="center", zorder=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", lw=0.8))

# legend sub-caption: hardware layer now holds CPU too
ax.add_patch(plt.Rectangle((1170, 1003), 185, 24, fc="white", ec="none", zorder=10))
ax.text(1171, 1015, "Single-node GPU + CPU", fontsize=11.5, color="#3a3a3a",
        ha="left", va="center", zorder=11)

ax.text(1345, 950, "red dashes: perf cgroup fences (one per pod), all measured in the same windows",
        fontsize=13, color=RED, ha="right", va="center", style="italic", zorder=11)

fig.savefig("docs/service_pipeline_fenced.png")
print("wrote docs/service_pipeline_fenced.png")
