#!/usr/bin/env python3
"""Unified resource-aware TherMOO scheduler. Scientific engine files are untouched.

Linux/Python 3.12; standard library only beyond the existing TherMOO environment.
Starts one process per trial to release native caches and memory on completion.
Default scope is the entire unfinished plan, including unknown and long trials.
--take-over drains old full runners without discarding their running trials.
"""
import argparse
from collections import Counter
import ctypes
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

GIB = 1024 ** 3
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"


def read(path, default=""):
    try:
        return Path(path).read_text()
    except OSError:
        return default


def numbers(path):
    result = {}
    for line in read(path).splitlines():
        fields = line.replace(":", "").split()
        if len(fields) >= 2:
            try:
                result[fields[0]] = int(fields[1])
            except ValueError:
                pass
    return result


def case(job):
    params = asdict(job)
    params.pop("seed")
    params.pop("block")
    return json.dumps(params, sort_keys=True)


def cgroup_paths():
    root = Path("/sys/fs/cgroup")
    paths = {root}
    for line in read("/proc/self/cgroup").splitlines():
        if line.startswith("0::"):
            leaf = (root / line[3:].lstrip("/")).resolve()
            paths.update(p for p in (leaf, *leaf.parents) if p.is_relative_to(root))
    return sorted(p for p in paths if p.is_dir())


def process_info(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        status = numbers(f"/proc/{pid}/status")
        return dict(ticks=int(fields[11]) + int(fields[12]),
                    rss=int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
                    peak=status.get("VmHWM", 0) * 1024,
                    state=fields[0], ppid=int(fields[1]), start=int(fields[19]))
    except (OSError, ValueError, IndexError):
        return dict(ticks=0, rss=0, peak=0, state="X", ppid=0, start=0)


def same_process(pid, start):
    info = process_info(pid)
    return info["start"] == start and info["state"] not in ("X", "Z")


def send_signal(pid, start, sig):
    """Check process identity; use a pidfd when the kernel supports it."""
    descriptor = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            descriptor = os.pidfd_open(pid)
        if not same_process(pid, start):
            return False
        if descriptor is None:
            os.kill(pid, sig)
        else:
            signal.pidfd_send_signal(descriptor, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def guard():
    """On controller failure, resume the old processes it temporarily stopped."""
    watched = {}
    for line in sys.stdin:
        row = json.loads(line)
        if row["op"] == "watch":
            watched[row["pid"]] = row["start"]
        else:
            watched.pop(row["pid"], None)
    for pid, start in watched.items():
        if process_info(pid)["state"] in ("T", "t"):
            if send_signal(pid, start, signal.SIGCONT):
                print(f"[RESTORE] resumed old process pid={pid}", flush=True)
    return 0


def old_runners(root):
    """Only recognize known TherMOO full-run commands owned by this Unix user."""
    found = {}
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            if folder.stat().st_uid != os.getuid():
                continue
            args = (folder / "cmdline").read_bytes().decode(errors="replace").split("\0")
            cwd = (folder / "cwd").resolve()
        except OSError:
            continue
        if "--worker-json" in args or "--guard" in args:
            continue
        module = any(args[i:i+3] == ["-m", "thermoo", "run"] for i in range(len(args)))
        inline = "-c" in args and any("thermoo.cli" in a and "execute(" in a.replace(" ", "")
                                     and "full" in a for a in args)
        adaptive = any(Path(a).name == "thermoo_dynamic.py" for a in args if a)
        if not (module or inline or adaptive):
            continue
        location = cwd
        try:
            for item in (folder / "environ").read_bytes().split(b"\0"):
                if item.startswith(b"THERMOO_ROOT="):
                    location = Path(os.fsdecode(item.split(b"=", 1)[1])).resolve()
                    break
        except OSError:
            pass
        if "--root" in args:
            try:
                location = Path(args[args.index("--root") + 1]).resolve()
            except (ValueError, IndexError):
                continue
        if location != root:
            continue
        if module or inline or adaptive:
            pid = int(folder.name)
            info = process_info(pid)
            if info["state"] not in ("X", "Z"):
                found[pid] = dict(start=info["start"], kind="adaptive" if adaptive else "fixed", children={})
    return found


def job_lock(pid, base):
    try:
        for descriptor in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(descriptor)
            except PermissionError:
                raise RuntimeError(f"Cannot inspect worker pid={pid}; it will not be retired")
            except FileNotFoundError:
                continue
            if "/outputs/full/" in target and target.endswith("/run.lock") and not target.startswith(str(base) + "/"):
                raise RuntimeError(f"Worker pid={pid} belongs to a different experiment fingerprint; leaving old runs intact")
            if target.startswith(str(base) + "/") and target.endswith("/run.lock"):
                return Path(target).parent.name
    except PermissionError:
        raise RuntimeError(f"Cannot inspect worker pid={pid}; it will not be retired")
    except FileNotFoundError:
        pass
    return None


class Handoff:
    """Freeze submissions in old fixed pools; keep their busy workers alive.

    A worker is stopped and inspected again before retirement. An open trial
    lock means objective computation may be in progress, so that worker resumes.
    Workers without trial locks have no unfinished trial state to discard.
    """
    def __init__(self, root, base, plan, keys, enabled):
        self.root, self.base, self.plan, self.keys = root, base, plan, keys
        self.enabled = enabled
        self.parents, self.watched, self.records = {}, {}, {}
        self.pipe = self.guard_proc = None
        self.last_discovery = -1e20

    def watch(self, pid, start):
        if self.pipe is None:
            reader, writer = os.pipe()
            try:
                self.guard_proc = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--guard"],
                    stdin=reader, start_new_session=True)
            finally:
                os.close(reader)
            self.pipe = os.fdopen(writer, "w", buffering=1)
        self.watched[pid] = start
        self.pipe.write(json.dumps(dict(op="watch", pid=pid, start=start)) + "\n")
        self.pipe.flush()

    def forget(self, pid):
        self.watched.pop(pid, None)
        if self.pipe is not None:
            self.pipe.write(json.dumps(dict(op="forget", pid=pid)) + "\n")
            self.pipe.flush()

    def stop(self, pid, start):
        self.watch(pid, start)
        if not send_signal(pid, start, signal.SIGSTOP):
            self.forget(pid)
            return False
        for _ in range(50):
            info = process_info(pid)
            if info["state"] in ("T", "t"):
                return True
            if not same_process(pid, start):
                return False
            time.sleep(0.01)
        return False

    def retire(self, pid, start):
        send_signal(pid, start, signal.SIGTERM)
        send_signal(pid, start, signal.SIGCONT)  # deliver pending TERM to stopped process

    def refresh(self):
        now = time.monotonic()
        if now - self.last_discovery >= 30:
            found = old_runners(self.root)
            if found and not self.enabled:
                raise RuntimeError("Old TherMOO runners are present; use --take-over for a unified queue")
            for pid, row in found.items():
                if pid in self.parents:
                    continue
                if job_lock(pid, self.base) is not None:
                    continue  # A forked trial is not a coordinator.
                self.parents[pid] = row
                if row["kind"] == "fixed":
                    if not self.stop(pid, row["start"]):
                        if not same_process(pid, row["start"]):
                            self.parents.pop(pid, None)
                            continue
                        raise RuntimeError(f"Could not pause old coordinator pid={pid}; no workers were killed")
                    print(f"[HANDOFF] old coordinator pid={pid}: submissions paused", flush=True)
                else:
                    send_signal(pid, row["start"], signal.SIGINT)
                    print(f"[HANDOFF] old adaptive coordinator pid={pid}: draining", flush=True)
            self.last_discovery = now
        table = {}
        if self.parents:
            for folder in Path("/proc").iterdir():
                if folder.name.isdigit():
                    pid = int(folder.name)
                    info = process_info(pid)
                    if info["ppid"] in self.parents:
                        args = read(folder / "cmdline").replace("\0", " ")
                        if "spawn_main" in args or "--worker-json" in args:
                            table[pid] = info
                            self.parents[info["ppid"]]["children"][pid] = info["start"]
        records = {}
        for parent, row in list(self.parents.items()):
            alive = 0
            for pid, start in list(row["children"].items()):
                if not same_process(pid, start):
                    row["children"].pop(pid, None)
                    self.forget(pid)
                    continue
                signature = job_lock(pid, self.base)
                if row["kind"] == "fixed" and signature is None:
                    if self.stop(pid, start):
                        signature = job_lock(pid, self.base)
                        if signature is None:
                            self.retire(pid, start)
                            continue
                        send_signal(pid, start, signal.SIGCONT)
                        self.forget(pid)
                if signature is None and row["kind"] == "adaptive":
                    try:
                        from thermoo.plan import Job
                        args = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
                        signature = Job(**json.loads(args[args.index("--worker-json")+1])).signature()
                    except (OSError, ValueError, IndexError):
                        pass
                alive += 1
                group = self.keys.get(signature, "startup")
                records[f"legacy:{pid}"] = dict(proc=SimpleNamespace(pid=pid), case=group,
                                                 signature=signature, parent=parent)
            if row["kind"] == "fixed" and not alive:
                self.retire(parent, row["start"])
            if not same_process(parent, row["start"]) and not alive:
                self.forget(parent)
                del self.parents[parent]
                print(f"[HANDOFF] old coordinator pid={parent}: retired", flush=True)
        self.records = records
        return records

    def close(self):
        # Exception recovery is reversible even if the guardian itself failed.
        for pid, start in self.watched.items():
            if process_info(pid)["state"] in ("T", "t"):
                send_signal(pid, start, signal.SIGCONT)
        if self.pipe is not None:
            self.pipe.close()
        if self.guard_proc is not None:
            self.guard_proc.wait(timeout=5)


def snapshot(pids, cpus, groups):
    total = busy = 0
    for line in read("/proc/stat").splitlines():
        fields = line.split()
        if fields and fields[0].startswith("cpu") and fields[0][3:].isdigit():
            if int(fields[0][3:]) in cpus:
                v = list(map(int, fields[1:9]))
                total += sum(v)
                busy += sum(v) - v[3] - v[4]  # steal counts as unavailable CPU
    mem = numbers("/proc/meminfo")
    available = mem.get("MemAvailable", mem.get("MemFree", 0)) * 1024
    mem_total = mem.get("MemTotal", 0) * 1024
    quotas = {}
    pid_slots = 1_000_000
    for group in groups:
        fields = read(group / "cpu.max").split()
        if len(fields) == 2 and fields[0] != "max":
            quotas[str(group)] = (int(fields[0]) / int(fields[1]),
                                  numbers(group / "cpu.stat").get("usage_usec", 0))
        limit = read(group / "memory.max").strip()
        current = read(group / "memory.current").strip()
        if limit.isdigit() and current.isdigit():
            mem_total = min(mem_total, int(limit))
            available = min(available, max(0, int(limit) - int(current)))
        pid_limit = read(group / "pids.max").strip()
        pid_current = read(group / "pids.current").strip()
        if pid_limit.isdigit() and pid_current.isdigit():
            pid_slots = min(pid_slots, max(0, int(pid_limit) - int(pid_current) - 16))
    pressure = 0.0
    for line in read("/proc/pressure/memory").splitlines():
        if line.startswith("full "):
            pressure = float(dict(v.split("=") for v in line.split()[1:])["avg10"])
    return dict(now=time.monotonic(), total=total, busy=busy,
                procs={pid: process_info(pid) for pid in pids}, quotas=quotas,
                available=available, mem_total=mem_total, pressure=pressure, pid_slots=pid_slots,
                swapout=numbers("/proc/vmstat").get("pswpout", 0))


def cpu_slots(previous, current, cpu_count, reserve, ceiling, smoothing):
    dt = max(0.001, current["now"] - previous["now"])
    ticks = os.sysconf("SC_CLK_TCK")
    own = sum(max(0, info["ticks"] - previous["procs"][pid]["ticks"])
              for pid, info in current["procs"].items() if pid in previous["procs"])
    own = own / ticks / dt
    delta = current["total"] - previous["total"]
    host_busy = cpu_count * max(0, current["busy"] - previous["busy"]) / max(1, delta)
    bounds = [("host", float(cpu_count), host_busy)]
    for name, (limit, usage) in current["quotas"].items():
        if name in previous["quotas"]:
            used = max(0, usage - previous["quotas"][name][1]) / 1e6 / dt
            bounds.append((name, limit, used))
    budgets = []
    for name, limit, used in bounds:
        external = max(0.0, used - own)
        old = smoothing.get(name, external)
        weight = 0.7 if external > old else 0.3
        smoothing[name] = weight * external + (1 - weight) * old
        keep = min(reserve, limit * 0.10)
        budgets.append(limit - keep - smoothing[name])
    capacity = max(0.0, min(budgets))
    slots = math.floor(capacity + 0.001)
    if slots == 0 and capacity >= 0.75:
        slots = 1
    return min(ceiling, slots), host_busy


def memory_headroom(current, running, estimates, keys, floor, reserve_gb):
    reserve = min(reserve_gb * GIB, current["mem_total"] * 0.20)
    growth = 0
    for signature, entry in running.items():
        info = current["procs"].get(entry["proc"].pid, dict(rss=0, peak=0))
        group = entry.get("case", keys.get(signature))
        predicted = max(floor * GIB, estimates.get(group, 0), info["peak"] * 1.3)
        growth += max(0, predicted - info["rss"])
    return current["available"] - reserve - growth, reserve


def unlocked(folder):
    path = folder / "run.lock"
    try:
        with path.open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True
    except FileNotFoundError:
        return True


def worker(args):
    # Parent stays in the foreground; Ctrl-C drains its isolated worker groups.
    # Unexpected parent death must not leave orphan experiment processes.
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if parent == 1 or os.getppid() != parent:
        return 2
    os.nice(args.nice)
    from thermoo.engine import trial, atomic_json
    from thermoo.plan import Job
    try:
        result = trial(Job(**json.loads(args.worker_json)), "full", args.fe_progress)
    except BaseException:
        result = dict(status="failed", error=traceback.format_exc())
    result.update(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  worker_pid=os.getpid())
    atomic_json(Path(args.report), result)
    return 1 if result["status"] == "failed" else 0


def run(args):
    from thermoo.engine import experiment_fingerprint, atomic_json
    from thermoo.plan import jobs
    root = Path(args.root).resolve()
    fingerprint = experiment_fingerprint()
    base = root / "outputs" / "full" / fingerprint
    reports = root / "reports" / "scheduler" / fingerprint
    base.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    lock = (base / "scheduler.lock").open("a+")
    if not args.dry_run:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("[BUSY] A unified scheduler is already running for this fingerprint.", flush=True)
            return 2
    compatibility_lock = (base / "dynamic.lock").open("a+")
    compatibility_owned = False
    plan = {job.signature(): job for job in jobs()}
    keys = {sig: case(job) for sig, job in plan.items()}
    done, failed, cost, estimates, running, retry = set(), set(), {}, {}, {}, {}
    counts = Counter()
    cpus = set(os.sched_getaffinity(0))
    groups = cgroup_paths()
    ceiling = min(args.max_workers or len(cpus), len(cpus))

    def scan():
        for folder in base.iterdir():
            signature = folder.name
            if signature not in plan or signature in done:
                continue
            path = folder / "result.json"
            if not path.is_file() or not (folder / "population.npz").is_file():
                continue
            row = json.loads(path.read_text())
            if row.get("status") != "complete" or row.get("fingerprint") != fingerprint:
                continue
            done.add(signature)
            group = keys[signature]
            cost[group] = max(cost.get(group, 0.0), float(row["seconds"]))

    def eligible(signature):
        return signature not in done and signature not in failed and (
            args.max_seconds == 0 or
            keys[signature] in cost and cost[keys[signature]] <= args.max_seconds)

    def heavy(signature):
        job = plan[signature]
        return (job.method == "SMS" and job.objectives >= 5) or cost.get(keys[signature], 0) >= 1800

    scan()
    memory_reports = list(reports.glob("*.json"))
    memory_reports.extend((root / "reports" / "dynamic" / fingerprint).glob("*.json"))
    for path in memory_reports:
        if path.stem not in plan:
            continue
        row = json.loads(path.read_text())
        group = keys[path.stem]
        estimates[group] = max(estimates.get(group, 0), row.get("peak_rss_bytes", 0) * 1.3)
    initial = len(done)
    print(f"[GLOBAL] complete={initial}/{len(plan)} fingerprint={fingerprint}", flush=True)
    scope = "ALL unfinished trials" if args.max_seconds == 0 else f"measured <= {args.max_seconds:g}s"
    print(f"[UNIFIED] remaining={sum(map(eligible, plan))} scope={scope} "
          f"max_workers={ceiling} RAM_reserve={args.reserve_gb:g}GiB "
          f"CPU_reserve={args.cpu_reserve:g} growth={args.ramp}/sample", flush=True)
    if args.dry_run:
        previous = snapshot([], cpus, groups)
        print(f"[RAM] available={previous['available']/GIB:.2f}GiB "
              f"effective_total={previous['mem_total']/GIB:.2f}GiB", flush=True)
        print(f"[QUOTAS] {previous['quotas']}", flush=True)
        return 0
    handoff = Handoff(root, base, plan, keys, args.take_over)
    args._handoff = handoff
    legacy = handoff.refresh()
    previous = snapshot([e["proc"].pid for e in legacy.values()], cpus, groups)
    stopping = False
    smoothing = {}
    target = 0
    started = last_scan = last_log = time.monotonic()
    last_sample = started
    while True:
        try:
            now = time.monotonic()
            for signature, entry in list(running.items()):
                proc = entry["proc"]
                code = proc.poll()
                if code is None:
                    continue
                result = json.loads(entry["report"].read_text()) if entry["report"].is_file() else dict(
                    status="failed", error=f"Worker exited {code} without a result report")
                status = result.get("status", "failed")
                group = keys[signature]
                estimates[group] = max(estimates.get(group, 0), result.get("peak_rss_bytes", 0) * 1.3)
                if status in ("complete", "cached"):
                    done.add(signature)
                    if "seconds" in result:
                        cost[group] = max(cost.get(group, 0), float(result["seconds"]))
                elif status == "locked":
                    retry[signature] = now + 15
                else:
                    failed.add(signature)
                    stopping = True
                    print(f"[ERROR] {result.get('error', result)}", flush=True)
                counts[status] += 1
                job = plan[signature]
                print(f"[{status.upper()}] {job.block} {job.problem} M={job.objectives} "
                      f"D={job.variables} {job.label} seed={job.seed} "
                      f"seconds={result.get('seconds', 0):.2f} "
                      f"peak={result.get('peak_rss_bytes', 0)/GIB:.2f}GiB", flush=True)
                del running[signature]
            if now - last_scan >= args.refresh:
                scan()
                last_scan = now
            if now - last_sample >= args.poll:
                legacy = handoff.refresh()
                if not compatibility_owned:
                    try:
                        fcntl.flock(compatibility_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        compatibility_owned = True
                    except BlockingIOError:
                        pass
                all_running = {**running, **legacy}
                current = snapshot([e["proc"].pid for e in all_running.values()], cpus, groups)
                target, busy = cpu_slots(previous, current, len(cpus), args.cpu_reserve, ceiling, smoothing)
                target = min(target, len(all_running) + current["pid_slots"])
                headroom, reserve = memory_headroom(current, all_running, estimates, keys,
                                                    args.worker_gb, args.reserve_gb)
                pressure = current["pressure"] > 1.0 or current["swapout"] > previous["swapout"]
                if pressure or headroom < 0 or stopping:
                    target = 0
                active_signatures = set(running) | {e["signature"] for e in legacy.values() if e["signature"] in plan}
                active_cases = Counter(keys[sig] for sig in active_signatures)
                unknown_busy = sum(keys[sig] not in cost and not heavy(sig) for sig in active_signatures)
                probe_limit = min(4, max(1, target // 6))
                candidates = [sig for sig in plan if eligible(sig) and sig not in running
                              and sig not in active_signatures and retry.get(sig, 0) <= now]

                def priority(sig):
                    group = keys[sig]
                    unknown = group not in cost
                    if unknown and not heavy(sig) and active_cases[group] == 0 and unknown_busy < probe_limit:
                        return (0, 0, 0, plan[sig].seed)
                    if not unknown and not heavy(sig):
                        return (1, cost[group], active_cases[group], plan[sig].seed)
                    if not heavy(sig):
                        return (2, active_cases[group], cost.get(group, math.inf), plan[sig].seed)
                    return (3, active_cases[group], cost.get(group, math.inf), plan[sig].seed)

                # Times choose order, never exclude unknown or slow jobs in full mode.
                # Prefer a few unknown representatives, then known fast jobs. Long
                # jobs backfill available capacity and eventually fill the whole pool.
                candidates.sort(key=priority)
                launches = 0
                for signature in candidates:
                    if len(running) + len(legacy) >= target or launches >= args.ramp or stopping:
                        break
                    estimate = max(args.worker_gb * GIB, estimates.get(keys[signature], 0))
                    if estimate > headroom:
                        continue
                    if not unlocked(base / signature):
                        retry[signature] = now + 15
                        continue
                    report = reports / f"{signature}.json"
                    report.unlink(missing_ok=True)
                    cmd = [sys.executable, str(Path(__file__).resolve()), "--root", str(root),
                           "--worker-json", json.dumps(asdict(plan[signature])),
                           "--report", str(report), "--nice", str(args.nice)]
                    if args.fe_progress:
                        cmd.append("--fe-progress")
                    try:
                        proc = subprocess.Popen(cmd, start_new_session=True)
                    except OSError as error:
                        print(f"[WAIT] Could not start a worker: {error}; retrying next sample", flush=True)
                        break
                    running[signature] = dict(proc=proc, report=report, case=keys[signature])
                    launches += 1
                    headroom -= estimate
                    job = plan[signature]
                    timing = f"{cost[keys[signature]]:.1f}s" if keys[signature] in cost else "unmeasured"
                    print(f"[START] pid={proc.pid} active={len(running)+len(legacy)}/{target} "
                          f"{job.problem} {job.label} seed={job.seed} "
                          f"estimate={timing}", flush=True)
                if now - last_log >= args.log_every or launches:
                    state = dict(complete=len(done), planned=len(plan), active=len(running)+len(legacy),
                                 owned=len(running), inherited=len(legacy), old_coordinators=len(handoff.parents),
                                 target=target, cpu_busy_cores=round(busy, 2), cpu_count=len(cpus),
                                 ram_available_gb=round(current["available"]/GIB, 2),
                                 ram_reserve_gb=round(reserve/GIB, 2), pressure=pressure,
                                 eligible_remaining=sum(map(eligible, plan)), stopping=stopping)
                    atomic_json(reports / "status.json", state)
                    print(f"[AUTO] global={len(done)}/{len(plan)} active={len(running)+len(legacy)} "
                          f"owned={len(running)} inherited={len(legacy)} "
                          f"target={target} CPU={busy:.1f}/{len(cpus)} "
                          f"RAM_available={current['available']/GIB:.1f}GiB "
                          f"reserve={reserve/GIB:.1f}GiB "
                          f"eligible={state['eligible_remaining']} "
                          f"elapsed={(now-started)/60:.1f}min "
                          f"{'DRAINING' if stopping or len(running)+len(legacy)>target else ''}", flush=True)
                    last_log = now
                previous, last_sample = current, now
            scope_left = any(eligible(sig) for sig in plan)
            if not running and not legacy and not handoff.parents and (stopping or not scope_left):
                scan()
                if stopping or not any(eligible(sig) for sig in plan):
                    break
            time.sleep(0.5)
        except KeyboardInterrupt:
            stopping = True
            print("[DRAIN] No new starts; waiting for active and inherited trials to finish.", flush=True)
    scan()
    print(f"[DONE] global={len(done)}/{len(plan)} gained={len(done)-initial} "
          f"this_controller={dict(counts)} deferred={len(plan)-len(done)}", flush=True)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-seconds", type=float, default=0, help="0: all remaining trials (default)")
    parser.add_argument("--take-over", action="store_true", help="Drain existing TherMOO full runners without cancelling active trials")
    parser.add_argument("--max-workers", type=int, default=0, help="0: affinity CPU count")
    parser.add_argument("--reserve-gb", type=float, default=8)
    parser.add_argument("--worker-gb", type=float, default=1, help="Conservative initial worker memory estimate")
    parser.add_argument("--cpu-reserve", type=float, default=2)
    parser.add_argument("--poll", type=float, default=3)
    parser.add_argument("--refresh", type=float, default=30)
    parser.add_argument("--log-every", type=float, default=15)
    parser.add_argument("--ramp", type=int, default=2)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--fe-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-json", help=argparse.SUPPRESS)
    parser.add_argument("--report", help=argparse.SUPPRESS)
    parser.add_argument("--guard", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_workers < 0 or args.reserve_gb < 0 or args.cpu_reserve < 0 or args.worker_gb <= 0:
        parser.error("Resource limits must be nonnegative; worker-gb must be positive")
    if args.poll < 0.5 or args.refresh < 1 or args.ramp < 1 or args.max_seconds < 0:
        parser.error("poll>=0.5, refresh>=1, ramp>=1 and max-seconds>=0 are required")
    if not 0 <= args.nice <= 19:
        parser.error("nice must be between 0 and 19")
    if args.guard:
        return guard()
    args.root = args.root.resolve()
    if not (args.root / "thermoo" / "engine.py").is_file():
        parser.error("--root must point to the existing TherMOO project")
    os.environ["THERMOO_ROOT"] = str(args.root)
    sys.path.insert(0, str(args.root))
    if args.worker_json:
        return worker(args)
    try:
        return run(args)
    finally:
        if hasattr(args, "_handoff"):
            args._handoff.close()


if __name__ == "__main__":
    raise SystemExit(main())
