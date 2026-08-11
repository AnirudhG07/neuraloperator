"""
bench_jax.py — compare FFT engines by what is genuinely MEASURED, nothing estimated.

Real TPU behavior diverges so far from CPU and from XLA's static cost model that estimated
bytes/flops (cost_analysis, analytical VMEM formulas) broke more assumptions than they
confirmed.  So this harness keeps only measured signals:

  compare()  -> fast cross-engine summary: rel_err (vs numpy) + median blocked wall-clock
                TIME (+ x factor + cv% run-to-run noise + best-run min).  Measured on-device.
  timings()  -> record EVERY run's wall-clock (in order) -> CSV + a run#-vs-time PNG, so you
                can SEE the noise (real signal vs Colab jitter) and download both.
  profile()  -> capture a REAL xprof device trace per (engine, N).  The source of truth for
                mem / MXU% / HBM bytes.  One capture folder per (engine, N).
  explain()  -> read a captured trace and print, in plain terms, WHERE the device time went
                (MXU vs VPU/fusion vs reshape/copy, + top ops).  The quick human read of a
                profile; TensorBoard's memory_viewer / op_profile give the exact counters.

An "engine" turns a real (N, K) input into a spectrum; a "case" is a named list of engines.

    compare(case="rfft")                        # correctness + time
    runs = profile(case="rfft", Ns=(1024, 16384))   # xprof capture per (engine, N)
    explain(runs[0])                            # human read of one captured trace

Run on a Colab TPU:  python -m tpu.profiling.bench_jax   (or import and call).
"""
import glob
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "tpu")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np

from tpu.fft1d.jax_fft import (
    fft_iter,
    fft_recur,
    jax_rfft,
    jax_rfft_int8,
    rfft_trickA,
    rfft_trickB,
)
from tpu.fft1d.pallas_fft import pallas_full, pallas_half, pallas_trickB
from tpu.fft_core import BF, F

C = jnp.complex64
K_TILE = 128  # Pallas block width (must be a multiple of 128)
BACKEND = jax.devices()[0].platform
# Pallas runs the REAL kernel on TPU (interpret=False); on CPU only the emulation exists, so
# any CPU time here is emulation (not representative) — trust `time` only on a real v5e run.
_ITP = (BACKEND != "tpu")
TIME_REPS = 15  # median over this many blocked runs


# ─── engine registry ────────────────────────────────────────────────────────
def _jax(fn):
    return {"pallas": False, "fn": fn}


def _pal(kernel, half, dt=F):
    return {"pallas": True, "kernel": kernel, "half": half, "dt": dt}


ENGINES = {
    "jnp.fft":     _jax(lambda xr: jnp.fft.fft(xr.astype(C), axis=0)),  # full
    "jnp.rfft":    _jax(lambda xr: jnp.fft.rfft(xr, axis=0)),  # half, native
    "fft_recur":   _jax(lambda xr: fft_recur(xr.astype(C))),  # recursive radix-B baseline
    "fft_iter":    _jax(lambda xr: fft_iter(xr.astype(C))),  # full
    "trickA":      _jax(lambda xr: rfft_trickA(xr, fft_iter)),  # half, on fft_iter
    "trickB":      _jax(lambda xr: rfft_trickB(xr, fft_iter)),  # half, on fft_iter
    "jax_half":      _jax(lambda xr: jax_rfft(xr, True, F)),  # pure-JAX real/imag, f32
    "jax_half_bf16": _jax(lambda xr: jax_rfft(xr.astype(BF), True, BF)),  # pure-JAX, bf16
    "jax_half_int8": _jax(lambda xr: jax_rfft_int8(xr, True)),  # int8 direct DFT, int32 acc
    "pallas_full":     _pal(pallas_full, half=False),  # full, VMEM
    "pallas_half":     _pal(pallas_half, half=True),  # half via output truncation, VMEM
    "pallas_half_bf16": _pal(pallas_half, half=True, dt=BF),  # bf16 'complex32'
    "pallas_trickB":   _pal(pallas_trickB, half=True),  # half via self-made reversal
}

# Named line-ups. The first entry is the baseline (x = 1.00).
CASES = {
    "full":             ["jnp.fft", "fft_recur", "fft_iter", "pallas_full"],
    "rfft":             ["jnp.rfft", "trickA", "trickB", "pallas_half"],
    "pallas_vs_native": ["jnp.rfft", "pallas_half"],
    "pallas_rfft":      ["pallas_half", "pallas_trickB"],
    "bf16":             ["pallas_half", "pallas_half_bf16", "jax_half", "jax_half_bf16"],
    # datatype comparison: f32 vs bf16 ('complex32') vs int8 ('complex16') on the JAX path
    "dtype":            ["jax_half", "jax_half_bf16", "jax_half_int8"],
}


# ─── shared helpers ─────────────────────────────────────────────────────────
def _to_cplx(out):
    if isinstance(out, tuple):
        return np.asarray(out[0], np.float32) + 1j * np.asarray(out[1], np.float32)
    return np.asarray(out)


def _relerr(out, ref):
    g = _to_cplx(out)
    return float(np.max(np.abs(g - ref)) / np.max(np.abs(ref)))


def _ft(t):
    if t < 1e-3:
        return f"{t*1e6:.1f}us"
    if t < 1:
        return f"{t*1e3:.2f}ms"
    return f"{t:.2f}s"


def _build(spec, xj):
    """(jitted callable, input array) for an engine — input cast to the kernel's dtype."""
    if spec["pallas"]:
        kern, dt = spec["kernel"], spec["dt"]
        return jax.jit(lambda x, kern=kern, dt=dt: kern(x, K_TILE, _ITP, dt)), xj.astype(dt)
    return jax.jit(spec["fn"]), xj


def _times(fn, x, reps=TIME_REPS):
    """Sorted list of `reps` blocked wall-clock times, after one warmup (compile not timed)."""
    jax.block_until_ready(fn(x))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        ts.append(time.perf_counter() - t0)
    return sorted(ts)


def _median(ts):
    return ts[len(ts) // 2]


def _cv(ts):
    """Coefficient of variation (%) = std/mean — the run-to-run noise (e.g. Colab jitter)."""
    m = sum(ts) / len(ts)
    if len(ts) < 2 or m == 0:
        return 0.0
    var = sum((t - m) ** 2 for t in ts) / len(ts)
    return var ** 0.5 / m * 100


# ─── correctness + real time ────────────────────────────────────────────────
def compare(engines=None, case=None, Ns=(256, 1024, 16384), K=256, baseline=None):
    """Correctness (rel_err vs numpy) + median blocked wall-clock TIME per (N, engine).
    x = time ratio vs `baseline` (defaults to the first engine listed)."""
    names = engines or (CASES[case] if case else list(ENGINES))
    baseline = baseline or names[0]
    title = case or ", ".join(names)
    tnote = "real" if BACKEND == "tpu" else f"{BACKEND} (pallas=emulation)"
    print("=" * 78)
    print(f"[{title}]   K={K}   backend={BACKEND}   time={tnote}   reps={TIME_REPS}   "
          f"(x = vs {baseline})")
    print(f"  time = median of {TIME_REPS} runs;  cv% = std/mean (run-to-run noise);  "
          f"min = best run")
    print("=" * 78)
    print(f"  {'N':>8} {'engine':>14} {'rel_err':>10} {'time':>10} {'x':>6} "
          f"{'cv%':>6} {'min':>10}")
    for N in Ns:
        xr = np.random.randn(N, K).astype(np.float32)
        ref = np.fft.fft(xr, axis=0)  # full reference; slice per output rows
        xj = jnp.asarray(xr)
        res = {}
        for n in names:
            try:
                fn, x = _build(ENGINES[n], xj)
                res[n] = (fn(x), _times(fn, x))
            except Exception as e:  # e.g. VMEM OOM at large N — don't kill the run
                res[n] = ("ERR", repr(e)[:44])
        m0 = _median(res[baseline][1]) if res[baseline][0] != "ERR" else 1.0
        for n in names:
            out, ts = res[n]
            if out == "ERR":
                print(f"  {N:>8,} {n:>14}   ERR: {ts}")
                continue
            rows = (out[0] if isinstance(out, tuple) else out).shape[0]
            rel = _relerr(out, ref[:rows])
            med = _median(ts)
            print(f"  {N:>8,} {n:>14} {rel:>10.1e} {_ft(med):>10} {med/m0:>5.2f}x "
                  f"{_cv(ts):>5.1f}% {_ft(ts[0]):>10}")
        print()


# ─── real xprof profile capture (the ONLY measured mem / MXU% / bytes) ───────
def _tf_present():
    try:
        import tensorflow as tf
        return f"tensorflow {tf.__version__} present (op_profile / memory_viewer will populate)"
    except Exception:
        return ("!! tensorflow MISSING -> op_profile / memory_viewer will be EMPTY.  "
                "run:  pip install -q tensorflow")


def _profile_files(logdir):
    out = []
    for root, _dirs, files in os.walk(logdir):
        if "plugins/profile" in root.replace(os.sep, "/"):
            out += [os.path.join(root, f) for f in files]
    return out


def profile(engines=None, case=None, Ns=(1024,), K=256, logdir="/tmp/tpu_prof", reps=15):
    """Capture a REAL device profile per (engine, N) with xprof — the only source of
    MEASURED mem / MXU% / VPU% / HBM bytes here.  Each (engine, N) is written to its OWN
    subdir `logdir/<engine>_N<N>/`, so each is a separate TensorBoard run you can flip
    between and compare.  Warmup/compile happens OUTSIDE the trace; `reps` runs are timed
    inside it (block_until_ready so device work is actually captured).

    View:      %load_ext tensorboard ;  %tensorboard --logdir <logdir>   (PROFILE tab)
    Download:  !zip -r <logdir>.zip <logdir>
    """
    print(_tf_present())
    if BACKEND != "tpu":
        print(f"!! backend={BACKEND}: not v5e — this profile is emulation.  run on a Colab TPU.")
    names = engines or (CASES[case] if case else list(ENGINES))

    written = []
    for N in Ns:
        xj = jnp.asarray(np.random.randn(N, K).astype(np.float32))
        for n in names:
            try:
                fn, x = _build(ENGINES[n], xj)
                jax.block_until_ready(fn(x))  # warmup / compile OUTSIDE the trace
            except Exception as e:
                print(f"  skip {n:>14} N={N:<7} : {repr(e)[:55]}")
                continue
            sub = os.path.join(logdir, f"{n}_N{N}")
            with jax.profiler.trace(sub):
                for _ in range(reps):
                    jax.block_until_ready(fn(x))
            written.append(sub)
            print(f"  captured {n:>14} N={N:<7} -> {sub}")

    files = _profile_files(logdir)
    print(f"\n{len(written)} runs, {len(files)} profile files under {logdir}")
    for f in files[:6]:
        print("   ", f)
    if len(files) > 6:
        print(f"    ... (+{len(files) - 6} more)")
    print(f"\nview:      %load_ext tensorboard ;  %tensorboard --logdir {logdir}   (PROFILE tab)")
    print(f"download:  !zip -r {logdir}.zip {logdir}")
    return written


# ─── explain a captured trace in plain terms (measured, not modelled) ────────
def _bar(frac, width=28):
    frac = max(0.0, min(1.0, frac))
    return "█" * round(frac * width) + "·" * (width - round(frac * width))


def _hlo_bucket(cat):
    """Roll XLA's REAL `hlo_category` up into MXU/VPU/copy/DMA/custom (the ONE bucket list).
    e.g. 'convolution fusion'->MXU (our einsum-DFT), 'loop/custom fusion'->VPU,
    'slice'/'data formatting'/'reverse'->copy, 'copy-done'->DMA.  Category strings come from the
    `hlo_category` XStat on /device:TPU:0 of the *.xplane.pb (xprof — https://openxla.org/xprof;
    proto tensorflow.tsl.profiler.protobuf.xplane_pb2); the labels are XLA's, the roll-up is ours."""
    c = cat.lower()
    if not c:
        return "other"
    if "convolution" in c or "dot" in c or "gemm" in c:
        return "MXU matmul"
    if c in ("copy-start", "copy-done") or any(k in c for k in
                                               ("all-reduce", "infeed", "outfeed", "dma")):
        return "DMA / HBM"
    if "custom-call" in c:
        return "custom-call (fft)"
    if "fusion" in c:  # loop/custom/input/output fusion -> VPU element-wise (incl. gather_fusion)
        return "VPU / fusion"
    if any(k in c for k in ("slice", "data formatting", "reverse", "transpose", "copy",
                            "broadcast", "gather", "concatenate", "pad", "reshape", "bitcast")):
        return "reshape / copy"
    return c  # keep XLA's own label for anything unmapped


def _rv(b, i):  # read one protobuf varint at byte i -> (value, next_i)
    v = sh = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7f) << sh
        if not c & 0x80:
            return v, i
        sh += 7


def _mem_breakdown(buf):
    """Decode a memory_access_breakdown blob -> list of (op_type 1=read/2=write, mem_space, bytes).
    mem_space 1 = HBM (verified: for jnp.rfft it equals N*K*4 = the input), 3 = on-chip VMEM."""
    out, i = [], 0
    while i < len(buf):
        t, i = _rv(buf, i)
        if (t >> 3) == 1 and (t & 7) == 2:  # repeated MemoryAccess sub-message
            ln, i = _rv(buf, i)
            sub, i, j, d = buf[i:i + ln], i + ln, 0, {}
            while j < len(sub):
                t2, j = _rv(sub, j)
                val, j = _rv(sub, j)
                d[t2 >> 3] = val
            out.append((d.get(1, 0), d.get(2, 0), d.get(3, 0)))
        elif (t & 7) == 0:
            _, i = _rv(buf, i)
        elif (t & 7) == 2:
            ln, i = _rv(buf, i)
            i += ln
        else:
            break
    return out


def _xplane_stats(path):
    """HBM bytes + flops from the captured *.xplane.pb (the /device:TPU:0 plane's per-op cost
    model — what TensorBoard's Memory Viewer / Op Profile read).  Returns per-TRACE totals
    {hbm, bytes, flops, nruns, ai, peak_hbm, opaque}, or None if no xplane / no TensorFlow.
      hbm    = mem_space-1 (HBM) bytes from memory_access_breakdown  (SANE bandwidth)
      bytes  = total bytes_accessed, ALL mem levels (incl. VMEM reuse -> can exceed HBM peak)
    Opaque FFT / Pallas custom-calls report no breakdown -> their HBM write side is UNDER-counted
    (opaque=True flags it)."""
    xp = path if path.endswith(".xplane.pb") else next(
        iter(sorted(glob.glob(os.path.join(path, "**", "*.xplane.pb"), recursive=True))[-1:]), None)
    if not xp:
        return None
    try:  # tf import is heavy + optional; skip HBM cleanly if absent
        from tensorflow.tsl.profiler.protobuf import xplane_pb2
    except Exception:
        return None
    sp = xplane_pb2.XSpace()
    sp.ParseFromString(open(xp, "rb").read())
    tpu = next((p for p in sp.planes if p.name == "/device:TPU:0"), None)
    if tpu is None:
        return None
    sid = {k: v.name for k, v in tpu.stat_metadata.items()}

    def _val(s):
        f = s.WhichOneof("value")
        return getattr(s, f) if f else 0

    # bytes_accessed / flops / breakdown / hlo_category live on event_metadata (per unique op)
    eb = {}
    for mid, em in tpu.event_metadata.items():
        d = {sid.get(s.metadata_id, ""): _val(s) for s in em.stats}
        eb[mid] = (d.get("bytes_accessed", 0) or 0, d.get("flops", 0) or 0,
                   d.get("memory_access_breakdown", b""), d.get("hlo_category", "") or "")
    peak = {sid[s.metadata_id]: _val(s) for s in tpu.stats if s.metadata_id in sid}
    names = {m: em.name.split(" = ")[0].lstrip("%") for m, em in tpu.event_metadata.items()}
    tb = tf = hbm = total_ps = 0.0
    occ, opaque, by_cat, by_raw, durs = {}, False, {}, {}, {}
    for ln in tpu.lines:
        if ln.name != "XLA Ops":  # the device HLO-op line — already device-only, no host noise
            continue
        for ev in ln.events:
            b, fl, mab, cat = eb.get(ev.metadata_id, (0, 0, b"", ""))
            tb, tf, total_ps = tb + b, tf + fl, total_ps + ev.duration_ps
            by_cat[_hlo_bucket(cat)] = by_cat.get(_hlo_bucket(cat), 0.0) + ev.duration_ps
            by_raw[cat or "unknown"] = by_raw.get(cat or "unknown", 0.0) + ev.duration_ps
            durs.setdefault(names.get(ev.metadata_id, "?"), []).append(ev.duration_ps / 1e6)
            if mab:
                hbm += sum(by for _ot, ms, by in _mem_breakdown(mab) if ms == 1)
            elif "custom-call" in cat.lower():
                opaque = True  # custom-call (no breakdown) -> HBM write under-counted (by category)
            occ[ev.metadata_id] = occ.get(ev.metadata_id, 0) + 1
    nruns = max(occ.values(), default=1)
    return {"hbm": hbm, "bytes": tb, "flops": tf, "nruns": nruns,
            "ai": tf / tb if tb else 0.0, "opaque": opaque, "peak_hbm": peak.get(
                "peak_hbm_bw_gigabytes_per_second", 0.0), "durs": durs,
            "by_cat": {k: v / 1e6 for k, v in by_cat.items()},         # rolled up (us)
            "by_rawcat": {k: v / 1e6 for k, v in by_raw.items()},      # raw hlo_category (us)
            "total": total_ps / 1e6, "per_run": total_ps / 1e6 / nruns if nruns else 0.0}


def explain(path, top=12, wall_us=None):
    """Where the DEVICE time went — GROUND TRUTH from the *.xplane.pb /device:TPU:0 plane (the
    same source TensorBoard's Op Profile / Memory Viewer read):
      • device time / run          (summed XLA-Ops event durations)
      • where the time goes        by XLA's real `hlo_category`
      • top ops (+ cv% jitter)
      • HBM traffic + operand-AI    (memory_access_breakdown / bytes_accessed / flops)
    Needs TensorFlow to read the xplane.  Pass `wall_us` (from timings.csv) to show wall−device."""
    mem = _xplane_stats(path)
    if mem is None:
        print(f"no *.xplane.pb under {path}, or TensorFlow missing (needed to read it).")
        return
    durs, by_raw, total, nruns = mem["durs"], mem["by_rawcat"], mem["total"] or 1.0, mem["nruns"]
    run = os.path.basename(path.rstrip("/")) if not path.endswith(".gz") else path
    print(f"\n── {run}  (xprof /device:TPU:0 — real hlo_category) " + "─" * 14)
    print(f"  device time  {mem['per_run']:8.1f} us / run   "
          f"({total/1e3:.2f} ms over {nruns} runs, {sum(len(v) for v in durs.values())} ops)")
    if wall_us is not None:
        gap = wall_us - mem["per_run"]
        print(f"  wall-clock   {wall_us:8.1f} us / run   "
              f"(host dispatch/sync = wall − device = {gap:7.1f} us, {gap/wall_us*100:4.0f}%)")
    dev_s, pk = mem["per_run"] / 1e6, mem["peak_hbm"] or 819.0
    if mem["hbm"] == 0 and mem["opaque"]:
        print("  HBM traffic      n/a  (opaque custom-call — cost model reports no breakdown)")
    else:
        hbr = mem["hbm"] / nruns
        gbs = hbr / dev_s / 1e9 if dev_s else 0.0
        tag = "  [read-side; opaque custom-call under-counts writes]" if mem["opaque"] else ""
        print(f"  HBM traffic  {hbr/1e6:8.2f} MB / run   ({gbs:6.1f} GB/s = {gbs/pk*100:3.0f}% "
              f"of {pk:.0f} peak){tag}")
    print(f"  operand bytes {mem['bytes']/nruns/1e6:7.2f} MB/run all-levels (VMEM reuse incl.);  "
          f"AI = {mem['ai']:.0f} flop/byte, ridge ~240 -> "
          f"{'MEMORY' if mem['ai'] < 240 else 'COMPUTE'}-bound")
    print("  cv% = per-op run-to-run noise (std/mean) — high cv% = jitter.")
    print("\n  device time by RAW hlo_category (xprof Op-Profile categories, no rollup):")
    for b, dd in sorted(by_raw.items(), key=lambda kv: -kv[1]):
        print(f"    {b:>22} {dd/total*100:5.1f}%  [{_bar(dd/total)}]")
    print(f"\n  top {top} ops:   (n = occurrences ~= #runs)")
    print(f"    {'share':>6} {'total':>9} {'mean':>9} {'cv%':>6} {'n':>4}  op")
    for nm, ds in sorted(durs.items(), key=lambda kv: -sum(kv[1]))[:top]:
        tot, mean = sum(ds), sum(ds) / len(ds)
        print(f"    {tot/total*100:5.1f}% {tot/1e3:8.2f}ms {mean/1e3:8.3f}ms "
              f"{_cv(ds):>5.1f}% {len(ds):>4}  {nm[:44]}")


def _read_timings(csv):
    """timings.csv (engine,N,run,seconds) -> {(engine, int N): sorted [seconds]}."""
    out = {}
    if not os.path.exists(csv):
        return out
    import csv as _csv
    with open(csv) as fh:
        rdr = _csv.reader(fh)
        next(rdr, None)  # header
        for row in rdr:
            if len(row) < 4:
                continue
            try:
                out.setdefault((row[0], int(row[1])), []).append(float(row[3]))
            except ValueError:
                continue
    for k in out:
        out[k].sort()
    return out


def report(logdir="tpu/profiling/tpu_prof", timings="tpu/profiling/timings.csv",
           out="tpu/profiling/PROFILING.md"):
    """Write ONE combined report (Markdown) from the captured *.xplane.pb traces + timings.csv:
      1. a 'verify in TensorBoard' map (which tab → each number),
      2. a summary table (sorted by N, ★ = fastest device time; memops = all data-movement),
      3. per (engine, N) RAW hlo_category breakdown — xprof Op-Profile categories, NO rollup.
    All values are XLA's own device fields (device_duration_ps / hlo_category / bytes_accessed /
    flops) — nothing estimated.  Returns the output path."""
    wall = _read_timings(timings)
    data = {}  # (engine, N) -> _xplane_stats
    for f in (sorted(os.listdir(logdir)) if os.path.isdir(logdir) else []):
        p = os.path.join(logdir, f)
        eng, sep, ns = f.rpartition("_N")
        if not (os.path.isdir(p) and sep and ns.isdigit()):
            continue
        m = _xplane_stats(p)
        if m:
            data[(eng, int(ns))] = m
    Ns = sorted({N for (_e, N) in data})
    L = [f"# TPU FFT profiling — {logdir}/ (xprof /device:TPU:0) + {os.path.basename(timings)}\n"]

    L += ["## methodology",
          ("- capture: `bench_jax.profile()` = `jax.profiler.trace()` around 15 "
          "`block_until_ready(fn(x))` per (engine, N)."),
          ("- device µs = Σ `XLA Ops` event `device_duration_ps` on /device:TPU:0 ÷ runs.  "
          "wall µs = host `perf_counter` (= device + dispatch)."),
          ("- HBM = `memory_access_breakdown` mem-space-1 bytes ÷ runs.  "
          "AI = `flops` ÷ `bytes_accessed`.  category = XLA `hlo_category`.\n")]

    L += ["## verify in TensorBoard (open this trace; each number's source)",
          "| number here | TensorBoard tool → field |", "|---|---|",
          "| device µs/run | Trace Viewer → 'XLA Modules' block duration; or Op Profile → total self-time ÷ runs |",
          "| wall µs/run | not in xprof — host `perf_counter` (= device + ~180µs dispatch) |",
          "| HBM MB/run | Memory Viewer → peak/bytes; or Op Profile → 'Bytes accessed' (HBM), ÷ runs |",
          "| AI (flop/byte) | Op Profile → 'FLOPs' ÷ 'Bytes accessed' |",
          "| MXU/VPU/memops/custom % | Op Profile → group by Category (rolled up from hlo_category) |",
          "| the RAW category rows below | Op Profile → the 'Category' column, ungrouped |",
          "| a single op (fusion.N) | Op Profile → expand a category; or Trace Viewer → click the op |\n"]

    def g(x, f="{:.1f}"):
        return f.format(x) if x is not None else "—"

    L += [("## summary  (sorted by N;  ★ = fastest device µs at that N;  "
          "memops = copy+reshape+transpose+slice+gather+reverse+DMA)\n"),
          "| N | engine | device µs | wall µs | gap µs | HBM MB | AI | MXU% | VPU% | memops% | custom% |",
          "|" + "---|" * 11]
    for N in Ns:
        engs = [(e, n) for (e, n) in data if n == N]
        best = min(engs, key=lambda k: data[k]["per_run"])
        for (e, n) in sorted(engs, key=lambda k: data[k]["per_run"]):
            m = data[(e, n)]
            tot, c = m["total"] or 1.0, m["by_cat"]
            w = wall.get((e, N))
            wm = _median(w) * 1e6 if w else None
            gap = (wm - m["per_run"]) if wm else None
            hbm = None if (m["hbm"] == 0 and m["opaque"]) else m["hbm"] / m["nruns"] / 1e6
            memops = (c.get("reshape / copy", 0) + c.get("DMA / HBM", 0)) / tot * 100
            L.append(f"| {N} | {'★ ' if (e, n) == best else ''}{e} | {m['per_run']:.1f} | "
                     f"{g(wm)} | {g(gap)} | {g(hbm, '{:.2f}')} | {m['ai']:.0f} | "
                     f"{c.get('MXU matmul', 0)/tot*100:.0f} | {c.get('VPU / fusion', 0)/tot*100:.0f} | "
                     f"{memops:.0f} | {c.get('custom-call (fft)', 0)/tot*100:.0f} |")

    L.append("\n## per (engine, N) — RAW `hlo_category` (xprof Op-Profile categories, no rollup)\n")
    for N in Ns:
        for (e, n) in sorted([(e, n) for (e, n) in data if n == N]):
            m = data[(e, n)]
            tot = m["total"] or 1.0
            L.append(f"**{e}  N={N}** — device {m['per_run']:.1f} us/run, {m['nruns']} runs")
            L.append("```")
            for cat, us in sorted(m["by_rawcat"].items(), key=lambda kv: -kv[1]):
                L.append(f"{cat:22} {us/tot*100:5.1f}%  {_bar(us/tot)}")
            L.append("```")
    L += ["\n## sources",
          "- xprof / JAX profiling: https://openxla.org/xprof/jax_profiling",
          ("- xplane proto (device_duration_ps / hlo_category / bytes_accessed / flops / "
          "memory_access_breakdown): `tensorflow.tsl.profiler.protobuf.xplane_pb2`")]
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out, "-", len(data), "engine×N rows")
    return out


def timings(engines=None, case=None, Ns=(1024,), K=256, reps=15,
            out="/tmp/timings.csv", plot=True):
    """Record EVERY run's wall-clock (in order, not just the median) so you can SEE the noise.
    Writes a tidy CSV `engine,N,run,seconds` and (plot=True) a PNG of run# vs time — the
    up/down noise graph, so you can tell real signal from Colab jitter.
    Download (Colab):  from google.colab import files; files.download(out)."""
    names = engines or (CASES[case] if case else list(ENGINES))
    rows = []
    for N in Ns:
        xj = jnp.asarray(np.random.randn(N, K).astype(np.float32))
        for n in names:
            try:
                fn, x = _build(ENGINES[n], xj)
                jax.block_until_ready(fn(x))  # warmup / compile (not recorded)
            except Exception as e:
                print(f"  skip {n} N={N}: {repr(e)[:55]}. Error: {e}")
                continue
            for r in range(reps):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(x))
                rows.append((n, N, r, time.perf_counter() - t0))

    with open(out, "w") as fh:
        fh.write("engine,N,run,seconds\n")
        for n, N, r, s in rows:
            fh.write(f"{n},{N},{r},{s:.9f}\n")
    print(f"wrote {len(rows)} timings ({reps}/engine) -> {out}")

    png = None
    if plot and rows:
        try:
            import collections

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            series = collections.defaultdict(list)
            for n, N, r, s in rows:
                series[(n, N)].append((r, s * 1e3))
            fig, ax = plt.subplots(figsize=(9, 4.6))
            for (n, N), pts in series.items():
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=".", ms=4, lw=1,
                        label=f"{n} N={N}")
            ax.set_xlabel("run #")
            ax.set_ylabel("time (ms)")
            ax.set_title(f"per-run wall-clock (noise)   backend={BACKEND}  K={K}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
            png = out.rsplit(".", 1)[0] + ".png"
            fig.tight_layout()
            fig.savefig(png, dpi=130)
            print(f"wrote noise plot          -> {png}")
        except Exception as e:
            print(f"plot skipped: {repr(e)[:60]}. Error: {e}")

    dl = f'from google.colab import files; files.download("{out}")'
    if png:
        dl += f'; files.download("{png}")'
    print(f"download:  {dl}")
    return out


if __name__ == "__main__":
    # correctness + real time (median + cv% noise)
    Ns = (256, 1024, 4096, 8192, 16384)
    compare(case="rfft", Ns=Ns)
    compare(case="bf16", Ns=Ns)

    # every run's wall-clock -> CSV + noise plot (download to inspect jitter)
    timings(case="rfft", Ns=Ns, reps=15, out="/tmp/timings.csv")

    # real measured device profiles — one xprof capture per (engine, N)
    runs = profile(case="rfft", Ns=Ns)
    for r in runs:
        explain(r)  # plain-terms read of where the device time went
