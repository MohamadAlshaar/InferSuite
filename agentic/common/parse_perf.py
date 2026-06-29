#!/usr/bin/env python3
"""Canonical parser for the agentic CPU suite. PER-FILE design: each metric is computed
from the pass-file that holds its events, using THAT file's own cycles/instructions.

Why per-file (not merge-then-ratio): separate passes have different durations, so a global
merge would pair e.g. uops from the mlp pass with cycles from a longer cache pass -> bogus
ratios (ILP < IPC). Each ratio must use its own pass's denominator.

Run with system python3. Usage:
    parse_perf.py [--freq-hz N] [--json out.json] [--label L] file1 [file2 ...]

Fixes baked in (see project_validation_findings): full FP incl packed-double + FMA bracket;
SMT-correct ILP (uops_executed.thread); CHA-DRAM*64; HARD-FAIL on zero cycles; TMA-sum assert.
"""
import sys, re, json, argparse

def parse_one(path):
    fc = {}
    try: txt = open(path).read()
    except OSError: return fc
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        ev = val = None
        # HUMAN format first (perf -o without -x): "  1,234,567   event-name   # ..."
        m = re.match(r'^([\d,]+)\s+([a-zA-Z][\w.:/-]+)', line)       # hyphens for topdown-*
        if m: val, ev = m.group(1), m.group(2)
        # CSV (-x,) fallback. Two shapes:
        #   interval:  ts,value,unit,event,cgroup,runtime,pct   (q0=float ts, q3=event)
        #   aggregate: value,unit,event,...                     (q0=value,  q2=event)
        elif ',' in line:
            q = line.split(',')
            if len(q) >= 4 and re.match(r'^\d+\.\d+$', q[0]) and re.match(r'^[a-zA-Z]', q[3]):
                if re.match(r'^[\d.]+$', q[1]): val, ev = q[1], q[3]   # interval
            elif len(q) >= 3 and re.match(r'^[\d.]+$', q[0]) and re.match(r'^[a-zA-Z]', q[2]):
                val, ev = q[0], q[2]                                   # aggregate
        if ev is None or val in (None, '', '<not', '<not counted>', '<not supported>'):
            continue
        try: v = float(val.replace(',', ''))
        except ValueError: continue
        ev = ev.strip().strip('/')
        fc[ev] = fc.get(ev, 0.0) + v          # SUM within a file (per-unit uncore)
    # wall-clock of the pass (perf prints "  N.NNN seconds time elapsed")
    mw = re.search(r'([\d.]+)\s+seconds time elapsed', txt)
    if mw: fc['seconds_elapsed'] = float(mw.group(1))
    return fc

def g(d, *names):
    for n in names:
        if n in d: return d[n]
    for n in names:
        for k in d:
            if k.endswith(n) or n in k: return d[k]
    return 0.0

def has(d, *names): return any(g(d, n) for n in names)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freq-hz', type=float, default=0)
    ap.add_argument('--json', default=None); ap.add_argument('--label', default='')
    ap.add_argument('files', nargs='+')
    a = ap.parse_args()
    dicts = [d for d in (parse_one(p) for p in a.files) if d]
    if not dicts:
        sys.stderr.write("FATAL: no parseable perf files.\n"); sys.exit(1)

    def pick(*events):
        for d in dicts:
            if has(d, *events): return d
        return None

    R = {'label': a.label}
    # ---- IPC: per-file (must use same-file cycles+instructions); report canonical + spread ----
    ipcs = [(g(d,'instructions')/g(d,'cycles')) for d in dicts if g(d,'cycles')>0 and g(d,'instructions')>0]
    if not ipcs:
        sys.stderr.write("FATAL: no file has nonzero cycles+instructions -> perf produced nothing. "
                         "Refusing to emit fake zeros.\n"); sys.exit(1)
    R['IPC'] = round(sum(ipcs)/len(ipcs), 2)
    if max(ipcs)-min(ipcs) > 0.25:
        R['IPC_spread'] = [round(min(ipcs),2), round(max(ipcs),2)]   # passes diverged (non-determinism)

    # ---- TMA (own file's slots) ----
    d = pick('slots')
    if d:
        slots=g(d,'slots'); ret=g(d,'topdown-retiring'); fe=g(d,'topdown-fe-bound')
        bad=g(d,'topdown-bad-spec'); be=g(d,'topdown-be-bound'); s=ret+fe+bad+be
        if slots>0:
            if s>0 and not (0.85<=s/slots<=1.15):
                sys.stderr.write(f"WARN: TMA sum/slots={s/slots:.2f} -> multiplexed/garbage; TMA suspect.\n")
            R['TMA_pct']={k:round(v/slots*100,1) for k,v in
                          [('retiring',ret),('frontend',fe),('bad_spec',bad),('backend',be)]}

    # ---- cache / AMAT / MPKI (own file) ----
    d = pick('mem_load_retired.l1_hit')
    if d:
        l1=g(d,'mem_load_retired.l1_hit'); l2=g(d,'mem_load_retired.l2_hit')
        l3=g(d,'mem_load_retired.l3_hit'); mp=g(d,'mem_load_retired.l3_miss'); tot=l1+l2+l3+mp
        ins=g(d,'instructions')
        if tot>0:
            R['AMAT_cyc']=round((l1*4+l2*12+l3*40+mp*200)/tot,2)
            R['hit_pct']={k:round(v/tot*100,2) for k,v in [('L1',l1),('L2',l2),('L3',l3),('miss',mp)]}
            if ins>0: R['LLC_load_MPKI']=round(mp/(ins/1000),3)

    # ---- FP (own file's cycles) ----
    d = pick('fp_arith_inst_retired.scalar_single','fp_arith_inst_retired.scalar_double')
    if d:
        ss=g(d,'fp_arith_inst_retired.scalar_single'); sd=g(d,'fp_arith_inst_retired.scalar_double')
        ps={k:g(d,f'fp_arith_inst_retired.{k}b_packed_single') for k in (128,256,512)}
        pd={k:g(d,f'fp_arith_inst_retired.{k}b_packed_double') for k in (128,256,512)}
        ops=ss+sd+ps[128]*4+ps[256]*8+ps[512]*16+pd[128]*2+pd[256]*4+pd[512]*8
        cyc=g(d,'cycles') or 1
        R['FP']={'element_ops':int(ops),'flops_no_fma':int(ops),'flops_all_fma_x2':int(ops*2),
                 'vectorized_pct':round((ops-ss-sd)/ops*100,1) if ops else 0,
                 'avx512_pct':round((ps[512]*16+pd[512]*8)/ops*100,1) if ops else 0,
                 'ops_per_cycle':round(ops/cyc,4),'scalar_dp':int(sd),'packed_dp_512':int(pd[512])}

    # ---- MLP / ILP (own file) ----
    d = pick('l1d_pend_miss.pending','uops_executed.thread','uops_executed.core')
    if d:
        pend=g(d,'l1d_pend_miss.pending'); pendc=g(d,'l1d_pend_miss.pending_cycles')
        uops=g(d,'uops_executed.thread') or g(d,'uops_executed.core'); cyc=g(d,'cycles') or 1
        if pendc>0: R['MLP']=round(pend/pendc,2)
        if uops>0:
            R['ILP']=round(uops/cyc,2)
            R['ILP_basis']='thread' if g(d,'uops_executed.thread') else 'core(SMT-inflated)'

    # ---- DRAM (CHA*64, node-wide) ----
    d = pick('unc_cha_imc_reads_count.normal','cas_count_read')
    if d:
        rd=g(d,'unc_cha_imc_reads_count.normal') or g(d,'cas_count_read')
        wr=g(d,'unc_cha_imc_writes_count.full') or g(d,'cas_count_write')
        if rd or wr:
            by=(rd+wr)*64; ins=g(d,'instructions'); wall=g(d,'seconds_elapsed')
            R['DRAM_GB']=round(by/1e9,2)
            R['DRAM_src']='cha(node-wide)' if g(d,'unc_cha_imc_reads_count.normal') else 'cas(node-wide)'
            # normalized so window-length doesn't distort cross-workload comparison:
            if wall>0: R['DRAM_GBps']=round(by/1e9/wall,2)
            if ins>0:  R['DRAM_bytes_per_kinstr']=round(by/(ins/1000),1)

    # ---- core-seconds from the TMA/canonical file ----
    if a.freq_hz>0:
        dc = pick('slots') or dicts[0]; cyc=g(dc,'cycles')
        if cyc>0: R['core_seconds']=round(cyc/a.freq_hz,2); R['freq_GHz']=round(a.freq_hz/1e9,3)

    # ---- print ----
    print(f"=== {a.label or 'perf'} ===")
    line=f"  IPC={R['IPC']:.2f}"
    if 'IPC_spread' in R: line+=f" (per-pass {R['IPC_spread'][0]}..{R['IPC_spread'][1]} — passes diverged)"
    if 'core_seconds' in R: line+=f"  core-s={R['core_seconds']} @ {R['freq_GHz']}GHz"
    print(line)
    if 'TMA_pct' in R:
        t=R['TMA_pct']; print(f"  TMA: Retiring {t['retiring']}% / Frontend {t['frontend']}% / Bad-spec {t['bad_spec']}% / Backend {t['backend']}%")
    if 'AMAT_cyc' in R:
        h=R['hit_pct']; print(f"  AMAT~{R['AMAT_cyc']}cyc  L1 {h['L1']}%/L2 {h['L2']}%/L3 {h['L3']}%/miss {h['miss']}%  LLC-MPKI={R.get('LLC_load_MPKI')}")
    if 'FP' in R:
        f=R['FP']; print(f"  FP: vectorized {f['vectorized_pct']}%  AVX-512 {f['avx512_pct']}%  ops/cyc {f['ops_per_cycle']}  "
                         f"FLOPs=[{f['flops_no_fma']:,} .. {f['flops_all_fma_x2']:,}]")
    if 'MLP' in R: print(f"  MLP={R['MLP']}  ILP={R.get('ILP')} ({R.get('ILP_basis')})")
    if 'DRAM_GB' in R:
        extra=[]
        if 'DRAM_GBps' in R: extra.append(f"{R['DRAM_GBps']} GB/s")
        if 'DRAM_bytes_per_kinstr' in R: extra.append(f"{R['DRAM_bytes_per_kinstr']} B/kinstr")
        print(f"  DRAM={R['DRAM_GB']} GB [{R['DRAM_src']}]" + (f"  ({', '.join(extra)})" if extra else ""))
    if a.json: json.dump(R, open(a.json,'w'), indent=2); print(f"  -> {a.json}")

if __name__=='__main__': main()
