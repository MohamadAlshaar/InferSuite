#!/usr/bin/env python3
"""oc_lineage_watcher.py <container-scope-cgroup-dir> <lineage-log.tsv> [--dry]

Rung-2 replacement for oc_cgroup_watcher.sh: LINEAGE-based agent/tool separation via the
kernel's netlink proc connector (PROC_EVENT_FORK/EXEC/EXIT pushed within microseconds —
no 20 ms poll, no name guessing).

Why: the comm-based watcher measured 2026-07-11 (ground truth): tool-side 0.00% contamination
(kept), but ~68% of short-lived-process CPU stuck agent-side (fork->exec window + poll), and
spawned NODE tools 100% misattributed (comm == gateway family). Lineage fixes both:

  boot phase   : everything in the scope = /agent (comm family sort, as before; no tools
                 spawn before the gateway is up). Gateway identified by comm 'openclaw-gatewa'
                 (verified 2026-07-08) — or first long-lived node/bun if it never renames.
  lineage phase: FORK by gateway  -> child PENDING (stays in /agent, it is born there)
                 EXEC by PENDING  -> TOOL ROOT: move to /toolexec (regardless of comm — a
                                    spawned node tool is a tool), log with kernel timestamp
                 FORK by TOOL     -> TOOL (born in /toolexec already — cgroup inheritance)
                 EXIT             -> logged (lifetime completeness for post-hoc attribution)

The TSV log (ns timestamps, pid, ppid, comm, class, action) is the ground truth consumed by
the validator's E4 PID-set purity check and by post-hoc perf-sample re-attribution.

Robustness: netlink can drop events under spawn storms (ENOBUFS) -> large SO_RCVBUF + on
overflow a full /proc resync sweep re-classifies unknowns by PPid walk. A 1 s safety sweep
also catches anything missed. MUST run as root. Pin to housekeeping CPUs from the caller."""
import os, sys, socket, struct, time, errno

CN_IDX_PROC, CN_VAL_PROC = 1, 1
NETLINK_CONNECTOR = 11
PROC_CN_MCAST_LISTEN = 1
EV_FORK, EV_EXEC, EV_COMM, EV_EXIT = 0x1, 0x2, 0x200, 0x80000000
AGENT_FAMILY = ("node", "bun")          # + openclaw* prefix
GATEWAY_COMM = "openclaw-gatewa"

DRY = "--dry" in sys.argv
SCOPE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else None
LOGPATH = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "/tmp/lineage.tsv"

def comm_of(pid):
    try:
        return open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        return ""

def ppid_of(pid):
    try:
        for ln in open(f"/proc/{pid}/status"):
            if ln.startswith("PPid:"):
                return int(ln.split()[1])
    except OSError:
        pass
    return 0

def is_agent_comm(c):
    return c in AGENT_FAMILY or c.startswith("openclaw")

class Watcher:
    def __init__(self, scope):
        self.scope = scope
        self.cls = {}            # tgid -> 'agent' | 'tool'
        self.pending = set()     # forked by gateway, awaiting exec
        self.gateway = set()     # gateway tgids
        self.log = open(LOGPATH, "a", buffering=1)
        self.log.write("ts_ns\tevent\tpid\tppid\tcomm\tclass\taction\n")
        # clock reference: event ts_ns is CLOCK_MONOTONIC-domain; log both clocks so
        # post-hoc joins with cpustat (EPOCHREALTIME) and perf can convert exactly.
        self.log.write(f"{time.monotonic_ns()}\tclockref\t0\t0\trealtime={time.time_ns()}\t-\t-\n")

    def scope_procs(self, sub=""):
        try:
            return [int(x) for x in open(f"{self.scope}{sub}/cgroup.procs")]
        except OSError:
            return []

    def move(self, pid, side):
        if DRY: return "dry"
        try:
            open(f"{self.scope}/{side}/cgroup.procs", "w").write(str(pid))
            return f"moved:{side}"
        except OSError:
            return "gone"

    def emit(self, ts, ev, pid, ppid, comm, cls, action):
        self.log.write(f"{ts}\t{ev}\t{pid}\t{ppid}\t{comm}\t{cls}\t{action}\n")

    def seed(self):
        """Initial snapshot: everything currently in the scope tree = agent side."""
        if not self.scope: return
        for sub in ("", "/agent", "/toolexec"):
            for pid in self.scope_procs(sub):
                c = comm_of(pid)
                self.cls[pid] = "agent"
                if c == GATEWAY_COMM:
                    self.gateway.add(pid)
                if sub != "/agent":
                    self.move(pid, "agent")
                self.emit(time.time_ns(), "seed", pid, ppid_of(pid), c, "agent", "seeded")

    def maybe_promote_gateway(self, pid, comm):
        if comm == GATEWAY_COMM and pid not in self.gateway:
            self.gateway.add(pid)
            self.emit(time.time_ns(), "gateway", pid, ppid_of(pid), comm, "agent", "identified")

    def on_fork(self, ts, ppid, pid):
        pcls = self.cls.get(ppid)
        if pcls is None:
            if DRY:                                  # observe-only mode: log everything
                self.emit(ts, "fork", pid, ppid, comm_of(pid), "?", "observe")
            return                                   # outside our episode
        if pcls == "tool":
            self.cls[pid] = "tool"
            # do NOT trust cgroup inheritance: the child may have been born in the window
            # before its parent's own move landed (event lag ~1 ms; measured 2026-07-12 —
            # 60/60 storm children stranded agent-side). Explicit move is idempotent.
            action = self.move(pid, "toolexec")
            self.emit(ts, "fork", pid, ppid, comm_of(pid), "tool", action)
        else:
            self.cls[pid] = "agent"
            # spawn boundary: a fork by any agent-side gateway-family process (node/bun/
            # openclaw*) is a tool candidate — the explicit gateway set is a logging nicety,
            # not the discriminator (the fake octest gateway and early real gateways keep
            # comm 'node'). Framework helpers are empirically zero (three-way join).
            if ppid in self.gateway or is_agent_comm(comm_of(ppid)):
                self.pending.add(pid)                # tool candidate: waits for exec
            self.emit(ts, "fork", pid, ppid, comm_of(pid), "agent",
                      "pending" if pid in self.pending else "inherit")

    def on_exec(self, ts, pid):
        if pid not in self.cls:
            if DRY:
                self.emit(ts, "exec", pid, ppid_of(pid), comm_of(pid), "?", "observe")
            return
        comm = comm_of(pid)
        if pid in self.pending:
            self.pending.discard(pid)
            self.cls[pid] = "tool"                   # lineage rule: gateway-spawned exec = tool
            action = self.move(pid, "toolexec")
            self.emit(ts, "exec", pid, ppid_of(pid), comm, "tool", action)
        else:
            if self.cls.get(pid) == "agent":
                self.maybe_promote_gateway(pid, comm)
            self.emit(ts, "exec", pid, ppid_of(pid), comm, self.cls.get(pid, "?"), "none")

    def on_exit(self, ts, pid):
        if pid in self.cls:
            self.emit(ts, "exit", pid, 0, "", self.cls.pop(pid), "none")
            self.pending.discard(pid)
            self.gateway.discard(pid)

    def resync(self):
        """Full sweep: classify any scope pid we do not know via PPid walk (storm recovery)."""
        if not self.scope: return
        for sub, side in (("", None), ("/agent", "agent"), ("/toolexec", "tool")):
            for pid in self.scope_procs(sub):
                if pid in self.cls: continue
                p, hops = pid, 0
                cls = None
                while hops < 32:
                    p = ppid_of(p)
                    if p <= 1: break
                    if p in self.gateway: cls = "tool"; break     # descendant of gateway spawn
                    if self.cls.get(p) == "tool": cls = "tool"; break
                    if self.cls.get(p) == "agent": cls = "agent"; break
                    hops += 1
                cls = cls or ("agent" if is_agent_comm(comm_of(pid)) else "tool")
                self.cls[pid] = cls
                action = self.move(pid, "toolexec" if cls == "tool" else "agent")
                self.emit(time.time_ns(), "resync", pid, ppid_of(pid), comm_of(pid), cls, action)

def main():
    if SCOPE and not DRY:
        os.makedirs(f"{SCOPE}/agent", exist_ok=True)
        os.makedirs(f"{SCOPE}/toolexec", exist_ok=True)
    w = Watcher(SCOPE)
    w.seed()

    s = socket.socket(socket.AF_NETLINK, socket.SOCK_DGRAM, NETLINK_CONNECTOR)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    s.bind((os.getpid(), CN_IDX_PROC))
    # subscribe: nlmsghdr + cn_msg + u32 op
    op = struct.pack("=I", PROC_CN_MCAST_LISTEN)
    cn = struct.pack("=IIIIHH", CN_IDX_PROC, CN_VAL_PROC, 0, 0, len(op), 0) + op
    NLMSG_DONE = 3
    nl = struct.pack("=IHHII", 16 + len(cn), NLMSG_DONE, 0, 0, os.getpid()) + cn
    s.send(nl)

    last_sweep = time.time()
    while True:
        try:
            data = s.recv(65536)
        except OSError as e:
            if e.errno == errno.ENOBUFS:
                w.emit(time.time_ns(), "overflow", 0, 0, "", "", "resync")
                w.resync()
                continue
            raise
        off = 0
        while off + 16 <= len(data):
            (mlen, mtype, _f, _seq, _pid) = struct.unpack_from("=IHHII", data, off)
            if mlen < 16: break
            payload = data[off + 16: off + mlen]
            off += (mlen + 3) & ~3
            if len(payload) < 20 + 16: continue
            what, _cpu, ts = struct.unpack_from("=II", payload, 20) + (struct.unpack_from("=Q", payload, 28)[0],)
            body = payload[36:]
            if what == EV_FORK and len(body) >= 16:
                pp, pt, cp, ct = struct.unpack_from("=IIII", body, 0)
                if cp == ct:                       # process (not thread) creation
                    w.on_fork(ts, pt, ct)
            elif what == EV_EXEC and len(body) >= 8:
                pp, pt = struct.unpack_from("=II", body, 0)
                w.on_exec(ts, pt)
            elif what == EV_EXIT and len(body) >= 8:
                pp, pt = struct.unpack_from("=II", body, 0)
                if pp == pt:
                    w.on_exit(ts, pt)
        if time.time() - last_sweep > 1.0:
            w.resync()
            last_sweep = time.time()
        if SCOPE and not DRY and not os.path.isdir(SCOPE):
            break                                   # container gone -> episode over

if __name__ == "__main__":
    main()
