"""Multiprocessing worker for gen_pcb.py's PathFinder negotiation loop.

Runs the negotiated-congestion A* (astar_neg) over plain-python state
extracted from the pcbnew-side Grid.  This module must NOT import pcbnew:
it is re-imported by spawned worker processes on Windows.

Static state (grid occupancy, via keepouts, per-edge metadata) arrives
once per worker via Pool(initializer=init_worker, initargs=(static,)).
Per-iteration usage/history snapshots live in shared-memory int32 arrays
written by the parent between batches; workers re-snapshot them into flat
python lists once per iteration (cheap) for fast scalar indexing in the
A* inner loop.

Usage-array encoding per cell: -1 = no net, -2 = multiple nets, else the
net id occupying the cell.  A cell clashes with net `nid` iff
u != -1 and (u == -2 or u != nid) -- exactly the dict-based
`any(k != name for k in d)` test in the parent.

The snapshot lists are tail-padded with sentinel values (0 for hist,
-1 for usage) so that the unchecked flat-index arithmetic inherited from
the dict-based pen() (which silently returned defaults for out-of-grid
indices) can never raise or wrap onto live cells.
"""

import atexit
import heapq
import math
import time
from multiprocessing import shared_memory

import numpy as np

GRID = 0.5  # overwritten from static payload in init_worker

NX = 0
NY = 0
BLOCKED = None      # (bytes, bytes) per layer
OWNER = None        # ({cellidx: nid}, {cellidx: nid})
VIA_FORBID = None   # {(ix, iy): nid}
EDGES = None        # {eid: (nid, sx, sy, gx, gy, width, extra)}
DISC = None         # {width: [(dx, dy), ...]} penalty-sampling disc
HARD = None         # {width: [(dx, dy), ...]} off-center hard-swath disc

_SHMS = []
_USE_NP = None      # per-layer int32 views into shared memory
_HIST_NP = None
_PAD = 0
_use_l = None       # per-iteration flat-list snapshots
_hist_l = None
_snap_it = None


def disc_offsets(width):
    half = width / 2 + (0.15 if width <= 0.31 else 0.36)
    r = int(math.ceil(half / GRID))
    offs = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if (dx * GRID) ** 2 + (dy * GRID) ** 2 <= half * half:
                offs.append((dx, dy))
    return offs


def init_worker(static):
    global GRID, NX, NY, BLOCKED, OWNER, VIA_FORBID, EDGES, DISC
    global _USE_NP, _HIST_NP, _PAD
    GRID = static["grid"]
    NX = static["nx"]
    NY = static["ny"]
    BLOCKED = static["blocked"]
    OWNER = static["owner"]
    VIA_FORBID = static["via_forbid"]
    EDGES = static["edges"]
    global HARD
    DISC = {}
    HARD = {}
    for meta in EDGES.values():
        w = meta[5]
        if w not in DISC:
            DISC[w] = disc_offsets(w)
            HARD[w] = [o for o in DISC[w] if o != (0, 0)]
    _PAD = 2 * NY + 8
    ncell = NX * NY
    use_np, hist_np = [], []
    for name in static["use_shm"]:
        s = shared_memory.SharedMemory(name=name)
        _SHMS.append(s)
        use_np.append(np.frombuffer(s.buf, dtype=np.int32, count=ncell))
    for name in static["hist_shm"]:
        s = shared_memory.SharedMemory(name=name)
        _SHMS.append(s)
        hist_np.append(np.frombuffer(s.buf, dtype=np.int32, count=ncell))
    _USE_NP = use_np
    _HIST_NP = hist_np
    atexit.register(_release_shm)


def _release_shm():
    """Drop numpy views before SharedMemory.__del__ runs at interpreter
    shutdown; otherwise close() raises BufferError (exported pointers)."""
    global _USE_NP, _HIST_NP, _use_l, _hist_l
    _USE_NP = _HIST_NP = _use_l = _hist_l = None
    for s in _SHMS:
        try:
            s.close()
        except Exception:
            pass


def route_batch(task):
    """task = (it, p_now, [eid, ...]) ->
    [(eid, path-or-None, snap_seconds, astar_seconds), ...]"""
    global _snap_it, _use_l, _hist_l
    it, p_now, eids = task
    tsnap = 0.0
    if _snap_it != it:
        t0 = time.perf_counter()
        _use_l = [a.tolist() + [-1] * _PAD for a in _USE_NP]
        _hist_l = [a.tolist() + [0] * _PAD for a in _HIST_NP]
        _snap_it = it
        tsnap = time.perf_counter() - t0
    out = []
    for eid in eids:
        t0 = time.perf_counter()
        path = _route(eid, p_now)
        out.append((eid, path, tsnap, time.perf_counter() - t0))
        tsnap = 0.0
    return out


def _route(eid, p_now):
    nid, sx, sy, gx, gy, width, extra = EDGES[eid]
    offs = DISC[width]
    hoffs = HARD[width]
    nx, ny = NX, NY
    blocked, owner, vf = BLOCKED, OWNER, VIA_FORBID
    use, hist = _use_l, _hist_l
    heappop, heappush = heapq.heappop, heapq.heappush

    startn = (0, sx, sy)
    goals = {(0, gx, gy), (1, gx, gy)}
    openq = [(0, startn)]
    came = {startn: None}
    cost = {startn: 0}
    pops = 0
    while openq and pops < 600000:
        _, node = heappop(openq)
        pops += 1
        if node in goals:
            path = []
            while node:
                path.append(node)
                node = came[node]
            return path[::-1]
        l, ix, iy = node
        bl = blocked[l]
        owl = owner[l]
        ul = use[l]
        hl = hist[l]
        cnode = cost[node]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nix, niy = ix + dx, iy + dy
            # is_free(l, nix, niy, nid, extra), inlined
            if not (0 <= nix < nx and 0 <= niy < ny):
                continue
            i = nix * ny + niy
            if bl[i] and (l, nix, niy) not in extra \
                    and owl.get(i) != nid:
                continue
            if hoffs:
                # wide tracks: whole swath must clear hard obstacles,
                # not just the centerline
                ok = True
                for hdx, hdy in hoffs:
                    hx, hy = nix + hdx, niy + hdy
                    if not (0 <= hx < nx and 0 <= hy < ny):
                        ok = False
                        break
                    j = hx * ny + hy
                    if bl[j] and (l, hx, hy) not in extra \
                            and owl.get(j) != nid:
                        ok = False
                        break
                if not ok:
                    continue
            nn = (l, nix, niy)
            # pen(l, nix, niy, nid, p_now, offs), inlined
            c = 0
            clash = False
            for odx, ody in offs:
                j = (nix + odx) * ny + niy + ody
                c += hl[j]
                if not clash:
                    u = ul[j]
                    if u != -1 and (u == -2 or u != nid):
                        clash = True
            if clash:
                c += p_now
            nc = cnode + 1 + c
            if nc < cost.get(nn, 1 << 30):
                cost[nn] = nc
                came[nn] = node
                heappush(openq, (nc + abs(nix - gx) + abs(niy - gy), nn))
        ol = 1 - l
        i = ix * ny + iy
        if (not blocked[ol][i] or (ol, ix, iy) in extra
                or owner[ol].get(i) == nid):
            # via barrel + clearance reaches ~0.5mm: the four orthogonal
            # cells on BOTH layers must be clear too (off-grid stubs/ties
            # half a cell away were a DRC short source)
            vok = True
            for ll in (l, ol):
                bll = blocked[ll]
                owll = owner[ll]
                for hdx, hdy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    hx, hy = ix + hdx, iy + hdy
                    if not (0 <= hx < nx and 0 <= hy < ny):
                        vok = False
                        break
                    j = hx * ny + hy
                    if bll[j] and (ll, hx, hy) not in extra \
                            and owll.get(j) != nid:
                        vok = False
                        break
                if not vok:
                    break
            if vok and hoffs:
                blo = blocked[ol]
                owo = owner[ol]
                for hdx, hdy in hoffs:
                    hx, hy = ix + hdx, iy + hdy
                    if not (0 <= hx < nx and 0 <= hy < ny):
                        vok = False
                        break
                    j = hx * ny + hy
                    if blo[j] and (ol, hx, hy) not in extra \
                            and owo.get(j) != nid:
                        vok = False
                        break
            z = vf.get((ix, iy))
            if vok and (z is None or z == nid):
                nn = (ol, ix, iy)
                c = 0
                clash = False
                ulo = use[ol]
                hlo = hist[ol]
                for odx, ody in offs:
                    j = (ix + odx) * ny + iy + ody
                    c += hlo[j]
                    if not clash:
                        u = ulo[j]
                        if u != -1 and (u == -2 or u != nid):
                            clash = True
                if clash:
                    c += p_now
                nc = cnode + 10 + c
                if nc < cost.get(nn, 1 << 30):
                    cost[nn] = nc
                    came[nn] = node
                    heappush(openq, (nc + abs(ix - gx) + abs(iy - gy), nn))
    return None
