#!/usr/bin/env python3
"""Parse H100 co-located orchestration captures and build per-agent figures. SYSTEM python3.
    plot_orchestration.py <data_root> [app]     # app in {bcb,swe,openclaw}; default bcb
Reads <root>/<app>_spin_core, <app>_block_core, <app>_block_fp ; writes <root>/../plots/<app>/.
Formats:
  *_timeline.csv : perf stat -I -x, -> time,value,unit,event,run-time,pct,ratio,ratio-unit (+ '# started')
  perf_flat.txt  : perf report -g none --no-children -> "  NN.NN%  cmd  dso  [.] symbol"
  engine_pyspy.folded : py-spy raw -> "frame1;...;leaf <count>"  (frames = "func (file:line)")
  agent.log      : "<APP>/<id> SOLVED turn N" | "<APP>/<id> unsolved after N turns"
"""
import os, sys, re, json, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as _plt
# ---- agentic/inference/plots house theme (serif, light grid, Okabe-Ito colorblind-safe + purple accent) ----
_plt.rcParams.update({
    "font.family":"serif","font.serif":["DejaVu Serif"],"mathtext.fontset":"dejavuserif",
    "font.size":12,"axes.titlesize":13.5,"axes.labelsize":12,"xtick.labelsize":11,"ytick.labelsize":11,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.color":"#cccccc","grid.linewidth":0.5,"grid.alpha":0.6,"axes.axisbelow":True,
    "legend.frameon":False,"figure.dpi":150,"savefig.bbox":"tight",
})
ACCENT="#6a51a3"; NEUTRAL="#bdbdbd"
GREEN,BLUE,VERM,ORANGE="#009E73","#0072B2","#D55E00","#E69F00"
OKABE=[GREEN,BLUE,VERM,ORANGE,"#CC79A7","#56B4E9","#F0E442","#999999",ACCENT,"#66c2a5"]

# ---------- parsing ----------
def parse_timeline(path):
    tot = {}
    if not os.path.exists(path): return tot
    for ln in open(path):
        if ln.startswith("#") or not ln.strip(): continue
        f = ln.rstrip("\n").split(",")
        if len(f) < 4: continue
        val, ev = f[1].strip(), f[3].strip()
        if not ev: continue
        if val in ("<not counted>", "<not supported>", ""): tot.setdefault(ev, 0.0); continue
        try: tot[ev] = tot.get(ev, 0.0) + float(val)
        except ValueError: pass
    return tot

def microarch(tl):
    ins, cyc = tl.get("instructions",0.0), tl.get("cycles",0.0)
    return {"IPC": ins/cyc if cyc else 0.0,
            "cache_MPKI": tl.get("cache-misses",0.0)*1000/ins if ins else 0.0,
            "branch_MPKI": tl.get("branch-misses",0.0)*1000/ins if ins else 0.0,
            "cache_miss_rate": 100*tl.get("cache-misses",0.0)/tl.get("cache-references",1.0) if tl.get("cache-references") else 0.0,
            "core_sec": tl.get("task-clock",0.0)/1000.0}

def fp_stats(tl):
    s=tl.get("fp_arith_inst_retired.scalar_double",0.0); p1=tl.get("fp_arith_inst_retired.128b_packed_double",0.0)
    p2=tl.get("fp_arith_inst_retired.256b_packed_double",0.0); p5=tl.get("fp_arith_inst_retired.512b_packed_double",0.0)
    flops=s+p1*2+p2*4+p5*8; vec=p1*2+p2*4+p5*8
    return {"flops":flops,"scalar_ops":s,"p128":p1,"p256":p2,"p512":p5,
            "vectorized_pct":100*vec/flops if flops else 0.0,"avx512_pct":100*p5*8/flops if flops else 0.0}

NATIVE_BUCKETS = [
    ("ctx-switch / sched", re.compile(r"finish_task_switch|__schedule|\bschedule\b|resched|sched_|try_to_wake", re.I)),
    ("futex / cond-wait",  re.compile(r"futex|cond_wait|pthread_cond|_raw_spin|mutex", re.I)),
    ("cuda / driver",      re.compile(r"cuda|libcuda|nvidia|cuEvent|cuStream|Synchronize|EvtHandlr", re.I)),
    ("torch / aten",       re.compile(r"libtorch|libc10|aten|c10::|\btorch", re.I)),
    ("python interp",      re.compile(r"_PyEval|PyObject|PyNumber|_Py|ceval|python3\.1", re.I)),
    ("alloc / memcpy",     re.compile(r"malloc|\bfree\b|memcpy|memset|tcmalloc|memmove", re.I)),
]
PY_BUCKETS = [
    ("output bookkeeping", re.compile(r"update_async_output|output_token|gpu_input_batch|output_processor|process_outputs|_update_states", re.I)),
    ("sampler",            re.compile(r"sampl|top_k|topk|top_p|topp|penalt|token_bin|logits|exponential_noise", re.I)),
    ("input prep",         re.compile(r"_prepare_inputs|make_tensor|copy_to_gpu|input_ids|_prepare\b", re.I)),
    ("model / gpu exec",   re.compile(r"execute_model|gemm|model_runner|graphs\.py|replay|_ops\.py|attention|forward|linear|rotary|layernorm|rmsnorm", re.I)),
    ("scheduler / loop",   re.compile(r"schedul|run_busy_loop|run_engine_core|_process_input|add_request|\bcore\.py|abort|\bstep\b", re.I)),
    ("wait / idle",        re.compile(r"threading\.py|queue\.py|selectors|epoll|\bpoll\b|cond_wait|futex|acquire|\bwait\b|\bget\b", re.I)),
]
def bucketize(pairs, buckets):
    out={b[0]:0.0 for b in buckets}; out["other"]=0.0
    for pct,label in pairs:
        for name,rx in buckets:
            if rx.search(label): out[name]+=pct; break
        else: out["other"]+=pct
    return out

def parse_perf_flat(path):
    pairs=[]
    if not os.path.exists(path): return pairs
    for ln in open(path):
        m=re.match(r"\s*([\d.]+)%\s+\S+\s+(\S+)\s+\[[.k]\]\s+(.*)", ln)
        if m: pairs.append((float(m.group(1)), m.group(3).strip()+" "+m.group(2)))
    return pairs

def parse_pyspy_folded(path):
    leaf={}; total=0
    if not os.path.exists(path): return []
    for ln in open(path):
        m=re.match(r"(.*)\s+(\d+)$", ln.rstrip("\n"))
        if not m: continue
        frames,cnt=m.group(1).split(";"),int(m.group(2))
        lf=frames[-1] if frames else ""; leaf[lf]=leaf.get(lf,0)+cnt; total+=cnt
    return [(100*c/total,f) for f,c in leaf.items()] if total else []

def parse_agent_log(path):
    tasks=[]
    if not os.path.exists(path): return tasks
    for ln in open(path):
        m=re.search(r"/(\d+)\s+SOLVED turn (\d+)", ln)
        if m: tasks.append({"task":m.group(1),"solved":True,"turns":int(m.group(2))}); continue
        m=re.search(r"/(\d+)\s+unsolved after (\d+) turns", ln)
        if m: tasks.append({"task":m.group(1),"solved":False,"turns":int(m.group(2))})
    return tasks

def parse_toolexec_dir(path):
    """Sum the per-subprocess `perf stat -x,` CSVs (value,,event,runtime,pct,...) across all tool runs."""
    tot={}
    if not os.path.isdir(path): return tot
    for fn in glob.glob(os.path.join(path,"*.csv")):
        for ln in open(fn):
            if ln.startswith("#") or not ln.strip(): continue
            f=ln.rstrip("\n").split(",")
            if len(f)<3: continue
            val,ev=f[0].strip(),f[2].strip()
            if not ev or val in ("<not counted>","<not supported>",""): continue
            try: tot[ev]=tot.get(ev,0.0)+float(val)
            except ValueError: pass
    return tot

def summarize(run_dir):
    tl=parse_timeline(os.path.join(run_dir,"engine_timeline.csv"))
    fp=parse_timeline(os.path.join(run_dir,"engine_fp_timeline.csv"))
    return {"name":os.path.basename(run_dir),"microarch":microarch(tl) if tl else {},
            "fp":fp_stats(fp) if fp else {},
            "native":parse_perf_flat(os.path.join(run_dir,"perf_flat.txt")),
            "python":parse_pyspy_folded(os.path.join(run_dir,"engine_pyspy.folded"))}

# ---------- figures ----------
PALETTES={"tab10":OKABE,"Set2":OKABE}
def _clabel(c, spin, block): return "spin (baseline)" if c is spin else "evblock (blocked)"
def _stacked(ax, datas, labels, buckets_def, cmap, title, xlabel):
    import numpy as np
    bnames=[b[0] for b in buckets_def]+["other"]; cols=PALETTES.get(cmap, PALETTES["tab10"])
    left=np.zeros(len(datas)); y=np.arange(len(datas))
    for j,bk in enumerate(bnames):
        vals=np.array([bucketize(d,buckets_def)[bk] for d in datas])
        ax.barh(y, vals, left=left, height=0.55, label=bk, color=cols[j%len(cols)], edgecolor="white")
        for i,v in enumerate(vals):
            if v>=4.0: ax.text(left[i]+v/2, y[i], f"{v:.0f}", ha="center", va="center", fontsize=8)
        left+=vals
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0,100)
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=11, weight="bold")
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5,1.12), frameon=False)

def _donut_fig(datas, labels, buckets_def, cmap, title):
    import matplotlib.pyplot as plt, numpy as np
    bnames=[b[0] for b in buckets_def]+["other"]; cols=PALETTES.get(cmap, PALETTES["tab10"])
    colmap=[cols[i%len(cols)] for i in range(len(bnames))]
    n=len(datas); fig,axs=plt.subplots(1,n,figsize=(4.9*n,5.6)); axs=np.atleast_1d(axs)
    wedges=None
    for ax,d,lab in zip(axs,datas,labels):
        b=bucketize(d,buckets_def); vals=[b[k] for k in bnames]
        wedges,_=ax.pie(vals,colors=colmap,startangle=90,counterclock=False,
                        wedgeprops=dict(width=0.42,edgecolor="white"))
        cum=0
        for v in vals:
            if v>=4:
                ang=np.deg2rad(90-(cum+v/2)/100.0*360)
                ax.text(0.79*np.cos(ang),0.79*np.sin(ang),f"{v:.0f}",ha="center",va="center",fontsize=9,weight="bold")
            cum+=v
        ax.set_title(lab,fontsize=12,weight="bold")
    fig.legend(wedges,bnames,loc="lower center",ncol=min(4,len(bnames)),fontsize=9,frameon=False)
    fig.suptitle(title,fontsize=12,weight="bold")
    fig.tight_layout(rect=[0,0.10,1,0.95])
    return fig

def make_figures(sums, outdir, app, during=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
    os.makedirs(outdir, exist_ok=True)
    byname={s["name"]:s for s in sums}
    spin=byname.get(f"{app}_spin_core"); block=byname.get(f"{app}_block_core")
    COL={"spin (baseline)":ACCENT,"evblock (blocked)":BLUE}
    conds=[c for c in (spin,block) if c and c["microarch"]]
    Dm=full_metrics(during) if during else {}
    if Dm.get("IPC"):  # 01 microarch (during generation, replay basis -> consistent with table)
        mk=[("IPC","IPC"),("cache_MPKI","cache-MPKI"),("branch_MPKI","branch-MPKI")]; x=np.arange(len(mk))
        fig,ax=plt.subplots(figsize=(7,5))
        b=ax.bar(x,[Dm[k] for k,_ in mk],0.5,color=ACCENT)
        ax.bar_label(b,fmt="%.2f",padding=2,fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels([n for _,n in mk]); ax.set_ylabel("value")
        ax.set_title(f"{app.upper()}: vLLM engine microarch during generation (Coder-32B/H100, replay)",fontsize=11,weight="bold")
        fig.tight_layout(); fig.savefig(f"{outdir}/{app}_01_microarch.png",dpi=140); plt.close(fig)
    if conds:  # core-seconds
        fig,ax=plt.subplots(figsize=(6,5)); labs=[_clabel(c,spin,block) for c in conds]; vals=[c["microarch"]["core_sec"] for c in conds]
        b=ax.bar(labs,vals,color=[COL.get(l) for l in labs],width=0.55); ax.bar_label(b,fmt="%.0f",padding=3,fontsize=11)
        ax.set_ylabel("vLLM CPU core-seconds (during inference)"); ax.set_title(f"{app.upper()}: during-inference CPU cost — busy-wait vs blocked",fontsize=11,weight="bold")
        if len(conds)==2 and vals[0]>0:
            ax.text(0.5,max(vals)*0.9,f"evblock reclaims {100*(vals[0]-vals[1])/vals[0]:.0f}%",ha="center",fontsize=10,bbox=dict(boxstyle="round",fc="#eee"))
        ax.margins(y=0.15); fig.tight_layout(); fig.savefig(f"{outdir}/{app}_02_coresec.png",dpi=140); plt.close(fig)
    # attribution from the CORE REPLAY (pure generation) for consistency with the replay microarch;
    # fall back to the live record if the replay attribution isn't present.
    attr = byname.get(f"{app}_spin_core_rp") or spin
    if attr and attr.get("python"):
        fig=_donut_fig([attr["python"]],["during generation (engine, replay)"],PY_BUCKETS,"tab10",
                       f"{app.upper()}: what the vLLM engine CPU orchestrates (Python, py-spy)")
        fig.savefig(f"{outdir}/{app}_03_python_attrib.png",dpi=140); plt.close(fig)
    if attr and attr.get("native"):
        fig=_donut_fig([attr["native"]],["during generation (engine, replay)"],NATIVE_BUCKETS,"Set2",
                       f"{app.upper()}: native CPU — sync/scheduling vs actual work (perf task-clock)")
        fig.savefig(f"{outdir}/{app}_04_native_attrib.png",dpi=140); plt.close(fig)
    # FP (from the *_block_fp run if present)
    fps=[s for s in sums if s["fp"] and s["fp"].get("flops",0)>0]
    if fps:
        fp=fps[0]["fp"]; fig,ax=plt.subplots(figsize=(7,4.6))
        cats=["scalar","128b","256b","512b"]; ops=[fp["scalar_ops"],fp["p128"],fp["p256"],fp["p512"]]
        b=ax.bar(cats,ops,color=[NEUTRAL,"#56B4E9",BLUE,"#08306b"]); ax.set_yscale("log")
        ax.bar_label(b,fmt="%.2g",padding=3,fontsize=8); ax.set_ylabel("fp_arith ops (log)")
        ax.set_title(f"{app.upper()}: engine FP during inference — {fp['vectorized_pct']:.0f}% vectorized, {fp['avx512_pct']:.0f}% AVX-512",fontsize=11,weight="bold")
        fig.tight_layout(); fig.savefig(f"{outdir}/{app}_06_fp.png",dpi=140); plt.close(fig)

def make_task_figure(agent_log, outdir, app):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
    from matplotlib.patches import Patch
    tasks=parse_agent_log(agent_log)
    if not tasks: return
    tasks=tasks[::-1]  # first task at top
    labels=[f"{app.upper()}/{t['task']}" for t in tasks]; turns=[t["turns"] for t in tasks]
    cols=[GREEN if t["solved"] else VERM for t in tasks]
    n_solved=sum(t["solved"] for t in tasks); total_turns=sum(t["turns"] for t in tasks)
    fig,ax=plt.subplots(figsize=(7,0.42*len(tasks)+1.6)); y=np.arange(len(tasks))
    ax.barh(y,turns,color=cols,height=0.6,edgecolor="white")
    for i,t in enumerate(tasks):
        ax.text(t["turns"]+0.05,y[i],"solved@t"+str(t["turns"]) if t["solved"] else "unsolved",va="center",fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=8); ax.set_xlabel("agent loops (turns) used")
    ax.set_xlim(0,max(turns)+2)
    ax.set_title(f"{app.upper()} tasks — Coder-32B: {n_solved}/{len(tasks)} solved, {total_turns} loops (=tool-exec runs)",fontsize=11,weight="bold")
    ax.legend(handles=[Patch(color=GREEN,label="solved"),Patch(color=VERM,label="unsolved")],loc="lower right",fontsize=9)
    fig.tight_layout(); fig.savefig(f"{outdir}/{app}_05_tasks_loops.png",dpi=140); plt.close(fig)

def parse_markers(path):
    t0=t1=None; tool=0.0; starts={}
    if not os.path.exists(path): return None
    for ln in open(path):
        f=ln.split()
        if len(f)<2: continue
        try: ts=float(f[0])
        except ValueError: continue
        tag=f[1]
        if tag=="RUN_START": t0=ts
        elif tag=="RUN_END": t1=ts
        elif tag=="toolexec_start": starts[f[2] if len(f)>2 else "_"]=ts
        elif tag=="toolexec_end":
            k=f[2] if len(f)>2 else "_"
            if k in starts: tool+=ts-starts.pop(k)
    if t0 is None or t1 is None: return None
    return {"wall":t1-t0,"tool":tool,"inference":(t1-t0)-tool}

def full_metrics(t):
    """All derived microarch measures from a merged event dict (replay -> counters consistent across groups)."""
    ins=t.get("instructions",0.0); cyc=t.get("cycles",0.0)
    l1=t.get("mem_load_retired.l1_hit",0.0); l2=t.get("mem_load_retired.l2_hit",0.0)
    l3=t.get("mem_load_retired.l3_hit",0.0); l3m=t.get("mem_load_retired.l3_miss",0.0); tot=l1+l2+l3+l3m
    stl=t.get("cycle_activity.stalls_total",0.0); sl3=t.get("cycle_activity.stalls_l3_miss",0.0)
    mp=t.get("l1d_pend_miss.pending",0.0); mpc=t.get("l1d_pend_miss.pending_cycles",0.0)
    s=t.get("fp_arith_inst_retired.scalar_double",0.0); p1=t.get("fp_arith_inst_retired.128b_packed_double",0.0)
    p2=t.get("fp_arith_inst_retired.256b_packed_double",0.0); p5=t.get("fp_arith_inst_retired.512b_packed_double",0.0)
    fl=s+p1*2+p2*4+p5*8; vec=p1*2+p2*4+p5*8
    m={}
    if cyc: m["IPC"]=ins/cyc
    if ins: m["cache_MPKI"]=t.get("cache-misses",0.0)*1000/ins; m["branch_MPKI"]=t.get("branch-misses",0.0)*1000/ins
    if tot: m["L1_pct"]=100*l1/tot; m["L2_pct"]=100*l2/tot; m["L3_pct"]=100*l3/tot; m["L3miss_pct"]=100*l3m/tot
    if tot: m["AMAT_cyc"]=(l1*5+l2*15+l3*50+l3m*250)/tot
    m["DRAM_MB"]=l3m*64/1e6
    if ins: m["DRAM_MPKI"]=l3m*1000/ins   # instruction-normalized DRAM traffic (duration-independent)
    if cyc: m["stall_pct"]=100*stl/cyc; m["memBound_pct"]=100*sl3/cyc
    if mpc: m["MLP"]=mp/mpc
    if fl: m["vec_pct"]=100*vec/fl; m["avx512_pct"]=100*p5*8/fl
    m["MFLOP"]=fl/1e6
    return m

def during_dict(root, cond):
    """Merge the 4 engine counter-groups. Prefer the CORE REPLAY (bcb_{cond}_core_rp) so the core
    group shares the same pure-generation basis as fp/mem/stall (all replays); fall back to the live
    record's core if the replay isn't present."""
    t={}
    core=os.path.join(root,f"bcb_{cond}_core_rp","engine_timeline.csv")
    if not os.path.exists(core): core=os.path.join(root,f"bcb_{cond}_core","engine_timeline.csv")
    t.update(parse_timeline(core))
    for g,fn in [("fp","engine_fp_timeline.csv"),("mem","engine_mem_timeline.csv"),("stall","engine_stall_timeline.csv")]:
        t.update(parse_timeline(os.path.join(root,f"bcb_{cond}_{g}",fn)))
    return t
def outside_dict(root):
    t={}
    for g in ["core","fp","mem","stall"]: t.update(parse_toolexec_dir(os.path.join(root,f"tool_{g}")))
    return t

def make_two_view(during, outside, outdir, app):
    """DURING (engine orchestration) vs OUTSIDE (tool-exec) — bars for comparable metrics + full table + hierarchy."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
    D=full_metrics(during); O=full_metrics(outside)
    if not D.get("IPC") or not O.get("IPC"): return
    # 08: grouped bars for comparable-scale metrics
    keys=[("IPC","IPC"),("cache_MPKI","cache-MPKI"),("branch_MPKI","branch-MPKI"),("DRAM_MPKI","DRAM-MPKI"),
          ("memBound_pct","mem-bound %"),("MLP","MLP"),("vec_pct","vectorized %FP")]
    x=np.arange(len(keys)); w=0.38; fig,ax=plt.subplots(figsize=(10,5))
    b1=ax.bar(x,[D.get(k,0) for k,_ in keys],w,label="DURING inference (vLLM orchestration)",color=ACCENT)
    b2=ax.bar(x+w,[O.get(k,0) for k,_ in keys],w,label="OUTSIDE inference (tool-exec)",color=GREEN)
    ax.bar_label(b1,fmt="%.1f",fontsize=8,padding=2); ax.bar_label(b2,fmt="%.1f",fontsize=8,padding=2)
    ax.set_xticks(x+w/2); ax.set_xticklabels([n for _,n in keys]); ax.set_ylabel("value")
    ax.set_title(f"{app.upper()}: CPU during vs outside inference — same box (Coder-32B/H100)",fontsize=11,weight="bold")
    ax.legend(fontsize=9); ax.margins(y=0.14); fig.tight_layout()
    fig.savefig(f"{outdir}/{app}_08_during_vs_outside.png",dpi=140); plt.close(fig)
    # 09: full microarch table
    rows=[("IPC","%.2f"),("cache_MPKI","%.1f"),("branch_MPKI","%.1f"),
          ("L1_pct","%.0f"),("L2_pct","%.0f"),("L3_pct","%.0f"),("L3miss_pct","%.1f"),
          ("AMAT_cyc","%.1f"),("DRAM_MPKI","%.2f"),("DRAM_MB","%.0f"),("stall_pct","%.0f"),("memBound_pct","%.1f"),
          ("MLP","%.2f"),("vec_pct","%.0f"),("avx512_pct","%.0f"),("MFLOP","%.0f")]
    lbl={"IPC":"IPC","cache_MPKI":"cache-MPKI","branch_MPKI":"branch-MPKI","L1_pct":"L1 hit %","L2_pct":"L2 hit %",
         "L3_pct":"L3 hit %","L3miss_pct":"L3-miss %","AMAT_cyc":"AMAT (cyc, est)","DRAM_MPKI":"DRAM-MPKI (norm.)",
         "DRAM_MB":"DRAM read (MB, dur-dep)","stall_pct":"stall %","memBound_pct":"mem-bound %","MLP":"MLP",
         "vec_pct":"vectorized %","avx512_pct":"AVX-512 %","MFLOP":"MFLOP"}
    cells=[[lbl[k], (f%D[k]) if k in D else "-", (f%O[k]) if k in O else "-"] for k,f in rows]
    fig,ax=plt.subplots(figsize=(7,0.42*len(cells)+1)); ax.axis("off")
    tbl=ax.table(cellText=cells,colLabels=["metric","DURING (engine)","OUTSIDE (tool-exec)"],loc="center",cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.35)
    for j in range(3): tbl[0,j].set_facecolor(ACCENT); tbl[0,j].set_text_props(color="white",weight="bold")
    ax.set_title(f"{app.upper()}: CPU microarch — during vs outside inference (Coder-32B/H100, cap6, replay)",fontsize=11,weight="bold",pad=14)
    fig.tight_layout(); fig.savefig(f"{outdir}/{app}_09_microarch_table.png",dpi=140,bbox_inches="tight"); plt.close(fig)
    # 10: cache hierarchy donuts
    fig,axs=plt.subplots(1,2,figsize=(9,4.8)); cols=[GREEN,"#66c2a5",ORANGE,VERM]
    for ax,dat,lab in zip(axs,[D,O],["DURING (engine)","OUTSIDE (tool-exec)"]):
        vals=[dat.get("L1_pct",0),dat.get("L2_pct",0),dat.get("L3_pct",0),dat.get("L3miss_pct",0)]
        ax.pie(vals,labels=["L1","L2","L3","DRAM"],colors=cols,autopct=lambda p:f"{p:.0f}" if p>=3 else "",
               startangle=90,counterclock=False,wedgeprops=dict(width=0.42,edgecolor="white"),pctdistance=0.79)
        ax.set_title(lab,fontsize=11,weight="bold")
    fig.suptitle(f"{app.upper()}: where loads are served (cache hierarchy) — during vs outside",fontsize=12,weight="bold")
    fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{outdir}/{app}_10_hierarchy.png",dpi=140); plt.close(fig)

def make_timesplit_fig(markers_path, outdir, app):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ts=parse_markers(markers_path)
    if not ts or ts["wall"]<=0: return
    fig,ax=plt.subplots(figsize=(6.4,5.8))
    vals=[ts["inference"],ts["tool"]]
    labs=[f"GPU generation (inference)\n{ts['inference']:.0f}s",f"CPU tool-exec\n{ts['tool']:.0f}s"]
    ax.pie(vals,labels=labs,colors=[ACCENT,VERM],autopct=lambda p:f"{p:.0f}%",
           startangle=90,counterclock=False,wedgeprops=dict(width=0.42,edgecolor="white"),
           pctdistance=0.80,textprops=dict(fontsize=10))
    ax.set_title(f"{app.upper()}: wall-clock CPU-vs-GPU time — Coder-32B/H100 ({ts['wall']:.0f}s total)",fontsize=12,weight="bold")
    fig.tight_layout(); fig.savefig(f"{outdir}/{app}_07_timesplit.png",dpi=140); plt.close(fig)

# ============ inf_thesis-style figures (perf-native, no py-spy) ============
NATIVE_ROLES = [
    ("OS kernel — thread scheduling", "#cb181d", re.compile(r"finish_task_switch|__schedule|\bschedule\b|schedule_tail|resched|try_to_wake|pick_next_task|__switch_to|context_switch", re.I)),
    ("OS kernel — futex / wait",      "#fc9272", re.compile(r"futex|hrtimer|\bpoll\b|epoll|do_sys_poll|do_syscall|syscall", re.I)),
    ("sync / spinlock / condvar",         "#6baed6", re.compile(r"_raw_spin|osq_lock|rwsem|\bmutex|cond_wait|pthread_cond", re.I)),
    ("CUDA GPU-sync (busy-wait)",      "#6a51a3", re.compile(r"cuda|libcuda|nvidia|cuEvent|cuStream|Synchronize|EvtHandlr|\[vdso\]", re.I)),
    ("Python interpreter",                "#d94801", re.compile(r"_PyEval|PyObject|PyNumber|_Py[A-Z]|ceval|python3\.1", re.I)),
    ("PyTorch / ATen",                    "#74c476", re.compile(r"libtorch|libc10|aten|c10::|\btorch", re.I)),
    ("C library — memcpy/malloc",      "#2171b5", re.compile(r"libc\.so|\bmalloc|\bfree\b|memcpy|memmove|memset|tcmalloc", re.I)),
]
def _roles(pairs):
    out={r[0]:0.0 for r in NATIVE_ROLES}; out["other"]=0.0
    for pct,label in pairs:
        for name,col,rx in NATIVE_ROLES:
            if rx.search(label): out[name]+=pct; break
        else: out["other"]+=pct
    return out

def make_attrib_donut(pairs, outpath, title):
    import matplotlib.pyplot as plt, math
    from matplotlib.patches import Patch
    r=_roles(pairs); items=sorted([(k,v) for k,v in r.items() if v>0.04], key=lambda x:-x[1])
    if not items: return
    colmap={n:c for n,c,_ in NATIVE_ROLES}; colmap["other"]=NEUTRAL
    vals=[v for _,v in items]; cols=[colmap.get(k,NEUTRAL) for k,_ in items]
    fig,ax=plt.subplots(figsize=(11,5.6))
    ax.pie(vals,colors=cols,startangle=90,counterclock=False,wedgeprops=dict(width=0.45,edgecolor="white",linewidth=1.5))
    cum=0.0
    for v in vals:
        if v>=6:
            ang=math.radians(90-(cum+v/2)/100.0*360)
            ax.text(0.77*math.cos(ang),0.77*math.sin(ang),f"{v:.0f}%",ha="center",va="center",color="white",fontweight="bold",fontsize=13)
        cum+=v
    ax.legend([Patch(color=colmap.get(k,NEUTRAL)) for k,_ in items],[f"{k}   ({v:.1f}%)" for k,v in items],
              title="Software component",loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=10,title_fontsize=11)
    ax.set_title(title,loc="left"); fig.tight_layout(); fig.savefig(outpath); plt.close(fig)

_SIG_ROWS=[("IPC (of 4.0 retire width)","IPC",4.0,"%.2f"),("L1 data-cache hit","L1_pct",100,"%.1f%%"),
           ("branch-MPKI (of 10)","branch_MPKI",10.0,"%.2f"),("DRAM-MPKI (of 1)","DRAM_MPKI",1.0,"%.2f"),
           ("memory-bound cycles","memBound_pct",100,"%.1f%%"),("MLP (of 4, outstanding)","MLP",4.0,"%.2f"),
           ("vectorized FP (AVX)","vec_pct",100,"%.0f%%")]
def make_signature_barh(m, outpath, title, color=ACCENT):
    import matplotlib.pyplot as plt, numpy as np
    labels=[r[0] for r in _SIG_ROWS]; norm=[100*min(m.get(k,0)/sc,1.0) for _,k,sc,_ in _SIG_ROWS]
    txt=[(fmt % m.get(k,0)) for _,k,sc,fmt in _SIG_ROWS]; y=np.arange(len(_SIG_ROWS))[::-1]
    fig,ax=plt.subplots(figsize=(9,5.6))
    ax.barh(y,norm,color=color,alpha=0.9,edgecolor="white",height=0.62)
    for yi,n,t in zip(y,norm,txt): ax.text(min(n+1.5,101),yi,t,va="center",fontweight="bold",fontsize=11,color="#222")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0,118); ax.set_xlabel("fraction of each metric's scale (%)")
    ax.set_title(title); fig.tight_layout(); fig.savefig(outpath); plt.close(fig)

_TV_ROWS=[("IPC (of 4.0)","IPC",4.0,"%.2f"),("branch-MPKI (of 10)","branch_MPKI",10.0,"%.2f"),
          ("DRAM-MPKI (of 1)","DRAM_MPKI",1.0,"%.2f"),("mem-bound %","memBound_pct",100,"%.1f"),
          ("MLP (of 4)","MLP",4.0,"%.2f"),("vectorized FP %","vec_pct",100,"%.1f")]
def make_twoview_barh(D, O, outpath, title):
    import matplotlib.pyplot as plt, numpy as np
    labels=[r[0] for r in _TV_ROWS]; y=np.arange(len(_TV_ROWS))[::-1]; h=0.38
    fig,ax=plt.subplots(figsize=(9.5,6))
    for off,dat,col,lab in [(h/2,D,ACCENT,"DURING inference (orchestration)"),(-h/2,O,GREEN,"OUTSIDE inference (tool-exec)")]:
        norm=[100*min(dat.get(k,0)/sc,1.0) for _,k,sc,_ in _TV_ROWS]
        ax.barh(y+off,norm,h,color=col,edgecolor="white",label=lab)
        for yi,n,(_,k,sc,fmt) in zip(y,norm,_TV_ROWS): ax.text(min(n+1.2,101),yi+off,(fmt%dat.get(k,0)),va="center",fontsize=9,fontweight="bold",color="#333")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0,118); ax.set_xlabel("fraction of each metric's scale (%)")
    ax.set_title(title); ax.legend(loc="lower right",fontsize=9); fig.tight_layout(); fig.savefig(outpath); plt.close(fig)

def make_timesplit_donut(markers_path, outpath, app):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ts=parse_markers(markers_path)
    if not ts or ts["wall"]<=0: return
    fig,ax=plt.subplots(figsize=(6.6,5.6))
    ax.pie([ts["inference"],ts["tool"]],colors=[ACCENT,NEUTRAL],startangle=90,counterclock=False,
           wedgeprops=dict(width=0.42,edgecolor="white",linewidth=2),autopct=lambda p:f"{p:.0f}%",
           pctdistance=0.78,textprops=dict(fontsize=14,fontweight="bold",color="white"))
    ax.legend(handles=[Patch(color=ACCENT,label=f"GPU generation (inference)  {ts['inference']:.0f}s"),
                       Patch(color=NEUTRAL,label=f"CPU tool-execution  {ts['tool']:.0f}s")],
              loc="lower center",bbox_to_anchor=(0.5,-0.08))
    ax.set_title(f"{app.upper()}: wall-clock CPU-vs-GPU time ({ts['wall']:.0f}s)")
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)

def make_tasks_barh(agent_log, outpath, app):
    import matplotlib.pyplot as plt, numpy as np
    from matplotlib.patches import Patch
    tasks=parse_agent_log(agent_log)
    if not tasks: return
    tasks=tasks[::-1]; y=np.arange(len(tasks)); turns=[t["turns"] for t in tasks]
    cols=[GREEN if t["solved"] else VERM for t in tasks]; ns=sum(t["solved"] for t in tasks); tt=sum(turns)
    fig,ax=plt.subplots(figsize=(7,0.4*len(tasks)+1.4))
    ax.barh(y,turns,color=cols,edgecolor="white",height=0.62)
    for yi,t in zip(y,tasks): ax.text(t["turns"]+0.06,yi,("solved @t"+str(t["turns"])) if t["solved"] else "unsolved",va="center",fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([f"{app.upper()}/{t['task']}" for t in tasks],fontsize=8)
    ax.set_xlabel("agent loops (turns) used"); ax.set_xlim(0,max(turns)+2)
    ax.set_title(f"{app.upper()} tasks — {ns}/{len(tasks)} solved, {tt} loops")
    ax.legend(handles=[Patch(color=GREEN,label="solved"),Patch(color=VERM,label="unsolved")],loc="lower right",fontsize=9)
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)

if __name__=="__main__":
    root=sys.argv[1] if len(sys.argv)>1 else "."
    app=sys.argv[2] if len(sys.argv)>2 else "bcb"
    base=os.path.dirname(os.path.abspath(root.rstrip("/")))
    outdir=os.path.join(base,"plots",app); os.makedirs(outdir,exist_ok=True)
    cond="spin" if os.path.isdir(os.path.join(root,"bcb_spin_core")) else "block"
    D=full_metrics(during_dict(root,cond)); O=full_metrics(outside_dict(root))
    natdir=f"bcb_{cond}_core_rp" if os.path.isdir(os.path.join(root,f"bcb_{cond}_core_rp")) else f"bcb_{cond}_core"
    natpairs=parse_perf_flat(os.path.join(root,natdir,"perf_flat.txt"))
    corelive=os.path.join(root,f"bcb_{cond}_core")
    try:
        make_signature_barh(D, f"{outdir}/{app}_01_signature_during.png",
            f"{app.upper()}: CPU micro-arch signature — during inference (Coder-32B/H100)")
        make_signature_barh(O, f"{outdir}/{app}_01b_signature_toolexec.png",
            f"{app.upper()}: CPU micro-arch signature — tool-exec, outside inference", color=GREEN)
        make_attrib_donut(natpairs, f"{outdir}/{app}_02_cpu_components.png",
            f"{app.upper()}: host-CPU time during inference, by software component")
        make_twoview_barh(D,O, f"{outdir}/{app}_03_during_vs_outside.png",
            f"{app.upper()}: CPU during vs outside inference — same box (Coder-32B/H100)")
        if os.path.exists(os.path.join(corelive,"markers.txt")):
            make_timesplit_donut(os.path.join(corelive,"markers.txt"), f"{outdir}/{app}_04_timesplit.png", app)
        if os.path.exists(os.path.join(corelive,"agent.log")):
            make_tasks_barh(os.path.join(corelive,"agent.log"), f"{outdir}/{app}_05_tasks.png", app)
        print("figures ->", outdir)
        print("DURING :", {k:round(v,3) for k,v in D.items()})
        print("OUTSIDE:", {k:round(v,3) for k,v in O.items()})
    except Exception as e:
        import traceback; traceback.print_exc(); print("figure gen error:", e)
    json.dump({"during":D,"outside":O,"native_roles":_roles(natpairs)},
              open(os.path.join(outdir,"summary.json"),"w"),indent=2)
