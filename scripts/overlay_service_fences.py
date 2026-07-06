#!/usr/bin/env python3
"""Overlay perf-cgroup measurement fences on the service pipeline figure.
Non-destructive: reads docs/service_pipeline.png, writes docs/service_pipeline_fenced.png.
Fences = where perf attached (capture_tiers.sh SPECS): the FastAPI/orchestrator pod,
Milvus, MongoDB, SeaweedFS, the LLM-d inference-gateway pod, and the vLLM pod."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

img = mpimg.imread("docs/service_pipeline.png")
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

ax.text(1345, 950, "red dashes: perf cgroup fences (one per pod), all measured in the same windows",
        fontsize=13, color=RED, ha="right", va="center", style="italic", zorder=11)

fig.savefig("docs/service_pipeline_fenced.png")
print("wrote docs/service_pipeline_fenced.png")
