#!/usr/bin/env python3
"""Time-split figures for the LOCAL service run (SYSTEM python3), from the 12-cell timing grid
(3 output tiers x 4 input buckets, n=20, concurrency 1, exact tokens).
(a) per-tier GPU-vs-CPU wall-clock donuts (house palette: GPU purple / CPU teal, like grand_timesplit)
(b) CPU-side stage decomposition per bucket (embed / vector search / other) — the bucket effect."""
import os, csv, glob, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "timing")
OUT  = os.path.join(HERE, "..", "plots"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 12, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False,
})
GPU, CPU = "#6a51a3", "#1b9e77"
TIERS = ["tok64", "tok192", "tok320"]
BUCKETS = ["short", "medium", "long", "very_long"]

def cell(tier, bucket):
    f = os.path.join(DATA, tier, f"rag_{bucket}_{tier}.csv")
    rows = list(csv.DictReader(open(f)))
    def med(n): return st.median([float(r[n]) for r in rows if r.get(n) not in (None, "", "None")])
    return {"gpu": med("model_backend_http_ms"), "cpu": med("frontend_overhead_ms"),
            "embed": med("rag_embed_ms"), "milvus": med("rag_milvus_ms")}

# ---- (a) per-tier donuts (sum over buckets = the tier's aggregate split) ----
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4))
for ax, tier in zip(axes, TIERS):
    cells = [cell(tier, b) for b in BUCKETS]
    g = sum(c["gpu"] for c in cells); c_ = sum(c["cpu"] for c in cells)
    shares = [g/(g+c_)*100, c_/(g+c_)*100]
    w, _ = ax.pie(shares, colors=[GPU, CPU], startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5))
    ax.text(0, 0.12, tier, ha="center", fontweight="bold", fontsize=13)
    ax.text(0, -0.18, f"GPU {shares[0]:.1f}%", ha="center", fontsize=10.5, color=GPU)
    ax.text(0, -0.42, f"CPU {shares[1]:.1f}%", ha="center", fontsize=10.5, color=CPU)
fig.suptitle("Wall-clock split per output tier: GPU generation vs CPU-side work",
             fontsize=12.5, y=1.02)
fig.legend(handles=[plt.Rectangle((0,0),1,1,color=GPU), plt.Rectangle((0,0),1,1,color=CPU)],
           labels=["GPU generation", "CPU-side (embed, search, orchestration)"],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
fig.savefig(os.path.join(OUT, "timing_donuts.png")); plt.close(fig)

# ---- (b) CPU-side stage decomposition per bucket (averaged over tiers; tier-independent) ----
fig, ax = plt.subplots(figsize=(8.2, 4.6))
X = np.arange(len(BUCKETS))
emb = [st.mean([cell(t, b)["embed"] for t in TIERS]) for b in BUCKETS]
mil = [st.mean([cell(t, b)["milvus"] for t in TIERS]) for b in BUCKETS]
oth = [st.mean([cell(t, b)["cpu"] for t in TIERS]) - e - m for b, e, m in zip(BUCKETS, emb, mil)]
ax.bar(X, emb, 0.6, label="Query embedding (BGE, CPU)", color=CPU, edgecolor="white")
ax.bar(X, mil, 0.6, bottom=emb, label="Vector search (Milvus)", color="#66c2a4", edgecolor="white")
ax.bar(X, oth, 0.6, bottom=[e+m for e, m in zip(emb, mil)], label="Other CPU-side (fetch, format, routing)",
       color="#c7e9c0", edgecolor="white")
for i, (e, m, o) in enumerate(zip(emb, mil, oth)):
    ax.text(i, e+m+o+2, f"{e+m+o:.0f} ms", ha="center", fontsize=10.5, fontweight="bold")
ax.set_xticks(X); ax.set_xticklabels([b.replace("_", " ") for b in BUCKETS])
ax.set_xlabel("input bucket (query length)"); ax.set_ylabel("median CPU-side time per request (ms)")
ax.set_title("CPU-side time per request, by input bucket")
ax.legend(loc="upper left", fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(OUT, "timing_cpu_stages.png")); plt.close(fig)

# ---- grid table to stdout ----
print(f"{'cell':24} {'GPU ms':>8} {'CPU ms':>8} {'GPU %':>7}")
for t in TIERS:
    for b in BUCKETS:
        c = cell(t, b)
        print(f"{t+' x '+b:24} {c['gpu']:8.0f} {c['cpu']:8.0f} {c['gpu']/(c['gpu']+c['cpu'])*100:6.1f}%")
print("wrote timing_donuts.png + timing_cpu_stages.png ->", OUT)
