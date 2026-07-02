#!/usr/bin/env python3
"""Per-task SWE-agent tool-exec (OUTSIDE inference) microarch. SYSTEM python3.
Reads <root>/tool_<inst>/tool_{core,fp,mem,stall}.csv (perf -x, -G aggregate format:
value,unit,event,cgroup,runtime,pct). Per-task signature barhs + an app-dependence comparison."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_orchestration as P

def parse_swe_tool_csv(path):
    t={}
    if not os.path.exists(path): return t
    for ln in open(path):
        if ln.startswith("#") or not ln.strip(): continue
        c=ln.split(",")
        if len(c)<3: continue
        v,e=c[0].strip(),c[2].strip()
        if not e or v in ("<not counted>","<not supported>",""): continue
        try: t[e]=t.get(e,0.0)+float(v)
        except ValueError: pass
    return t
def task_metrics(taskdir):
    t={}
    for g in ["core","fp","mem","stall"]:
        t.update(parse_swe_tool_csv(os.path.join(taskdir,f"tool_{g}.csv")))
    return P.full_metrics(t)

if __name__=="__main__":
    root=sys.argv[1] if len(sys.argv)>1 else "h100/data_swe"
    outdir="h100/plots/swe"; os.makedirs(outdir,exist_ok=True)
    tasks={}
    for d in sorted(glob.glob(os.path.join(root,"tool_*"))):
        name=os.path.basename(d).replace("tool_","").split("__")[0]
        tasks[name]=task_metrics(d)
    for name,m in tasks.items():
        P.make_signature_barh(m, f"{outdir}/swe_tool_{name}.png",
                              f"SWE-agent tool-exec [{name}]: CPU micro-arch signature", color=P.GREEN)
        print(name, {k:round(m.get(k,0),2) for k in ["IPC","cache_MPKI","branch_MPKI","memBound_pct","vec_pct","avx512_pct","MFLOP"]})
    import matplotlib.pyplot as plt
    names=list(tasks.keys()); cols=[P.GREEN,"#d94801",P.BLUE,P.ACCENT]
    fig,axs=plt.subplots(1,3,figsize=(13,4.5))
    for ax,(key,lab) in zip(axs,[("IPC","IPC"),("vec_pct","vectorized FP %"),("MFLOP","MFLOP")]):
        vals=[tasks[n].get(key,0) for n in names]
        b=ax.bar(names,vals,color=cols[:len(names)]); ax.bar_label(b,fmt="%.1f",fontsize=10)
        ax.set_title(lab);
        if key=="MFLOP": ax.set_yscale("log")
    fig.suptitle("SWE-agent tool-exec is app/task-dependent (per SWE-bench task)",fontsize=13,weight="bold")
    fig.tight_layout(); fig.savefig(f"{outdir}/swe_tool_compare.png",dpi=140); plt.close(fig)
    print("figs ->",outdir)
