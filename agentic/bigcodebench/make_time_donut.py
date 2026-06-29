#!/usr/bin/env python3
"""CPU-vs-GPU TIME donut for BigCodeBench: generation (GPU/inference) vs execution (CPU/tool)."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, os
def wall(f,a,b):
    d={w.split()[1]:float(w.split()[0]) for w in open(f) if len(w.split())==2}
    return d.get(b,0)-d.get(a,0)
gen=wall("runs/perf/gen_markers.txt","gen_start","gen_done")
exe=wall("runs/perf/markers.txt","perf_start","eval_done")
tot=gen+exe
fig,ax=plt.subplots(figsize=(7.5,5.5))
vals=[gen,exe]; cols=["#d62728","#1f77b4"]
labs=[f"GPU — LLM generation\n{gen:.0f}s ({gen/tot*100:.0f}%)",
      f"CPU — code execution\n{exe:.0f}s ({exe/tot*100:.0f}%)"]
ax.pie(vals,colors=cols,startangle=90,counterclock=False,wedgeprops=dict(width=0.42,edgecolor="white"),
       autopct=lambda p:f"{p:.0f}%",pctdistance=0.79,textprops=dict(color="white",fontweight="bold",fontsize=12))
ax.legend(labs,loc="center left",bbox_to_anchor=(0.97,0.5),fontsize=10,frameon=False)
ax.text(0,0,f"BigCodeBench\nhard (148)\n{tot:.0f}s total",ha="center",va="center",fontsize=11,fontweight="bold")
ax.set_title("BigCodeBench — CPU vs GPU time\n(during-inference GPU vs outside-inference CPU)")
fig.tight_layout(); fig.savefig("figures/bcb_cpu_vs_gpu_time.png",dpi=130); plt.close(fig)
print(f"gen(GPU)={gen:.1f}s  exec(CPU)={exe:.1f}s  -> GPU {gen/tot*100:.0f}% / CPU {exe/tot*100:.0f}%")
print("wrote figures/bcb_cpu_vs_gpu_time.png")
