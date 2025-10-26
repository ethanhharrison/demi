#!/usr/bin/env python3
"""
Simple N-point moving average filter for streaming sensor data.

Works well on Raspberry Pi (no third-party deps). Designed for:
  - incremental/streaming use via MovingAverage.update(x)
  - one-shot/batch use via moving_average(seq, N)

Behavior:
  - During "warm-up" (fewer than N samples), the filter returns the
    average of the samples seen so far (common for real-time smoothing).
  - Set require_full_window=True to return None until the window fills.
"""
import time
import threading
import math
import random
import statistics as stats
from collections import deque
from typing import Deque, Iterable, Iterator, Optional, Dict, Any

import requests
from fastapi import FastAPI
from pydantic import BaseModel


class MovingAverage:
    def __init__(self, N: int, *, require_full_window: bool = False) -> None:
        if N <= 0:
            raise ValueError("N must be a positive integer")
        self.N: int = N
        self.q: Deque[float] = deque()
        self.s: float = 0.0
        self.require_full_window: bool = require_full_window

    def reset(self) -> None:
        self.q.clear()
        self.s = 0.0

    def update(self, x: float) -> Optional[float]:
        self.q.append(x)
        self.s += x
        if len(self.q) > self.N:
            self.s -= self.q.popleft()
        if self.require_full_window and len(self.q) < self.N:
            return None
        return self.s / len(self.q)

def moving_average(seq: Iterable[float], N: int, *, require_full_window: bool = False) -> Iterator[Optional[float]]:
    m = MovingAverage(N, require_full_window=require_full_window)
    for v in seq:
        yield m.update(float(v))


elastic_url: str = "https://f8c759dad36b4678b91080ee8b61abf2.us-central1.gcp.cloud.es.io:443"
elastic_index: str = "telemetry-signal"
elastic_api_key: str = "RkU3aUg1b0J3VmVKblFPMlpJZW86eHVEOFhJaTVGUGxPZWFUQzlCaUtlQQ=="

elastic_headers = {
    "Content-Type": "application/json",
    "Authorization": f"ApiKey {elastic_api_key}",
}

def emit_to_elastic(doc: Dict[str, Any]) -> None:
    r = requests.post(
        f"{elastic_url}/{elastic_index}/_doc",
        json=doc,
        headers=elastic_headers,
        timeout=2,
    )
    r.raise_for_status()


class FilterOrchestrator:
    def __init__(self) -> None:
        self.ma5 = MovingAverage(5)
        self.ma15 = MovingAverage(15)
        self.ma45 = MovingAverage(45)
        self.mode: str = "auto"
        self.recent: Deque[float] = deque(maxlen=60)
        self.w_short: float = 0.6
        self.w_mid: float = 0.3
        self.w_long: float = 0.1

    def set_mode(self, mode: str) -> None:
        if mode not in {"auto", "ma5", "ma15", "ma45", "tri"}:
            raise ValueError("mode must be one of: auto|ma5|ma15|ma45|tri")
        self.mode = mode

    def set_tri_weights(self, short: float, mid: float, long: float) -> None:
        s = max(1e-9, short + mid + long)
        self.w_short = short / s
        self.w_mid = mid / s
        self.w_long = long / s

    def metrics(self, x: float) -> Dict[str, float]:
        self.recent.append(x)
        if len(self.recent) < 5:
            return {"mean": x, "std": 0.0, "mad": 0.0, "spike_ratio": 0.0}
        mean = sum(self.recent) / len(self.recent)
        std = (sum((v - mean) ** 2 for v in self.recent) / len(self.recent)) ** 0.5 or 1e-9
        mad = stats.median(abs(v - mean) for v in self.recent) or 1e-9
        spikes = sum(1 for v in self.recent if abs(v - mean) > 3.0 * mad)
        return {"mean": mean, "std": std, "mad": mad, "spike_ratio": spikes / len(self.recent)}

    def tri_mix(self, y5: float, y15: float, y45: float, std: float) -> float:
        u = max(0.0, min(std / 0.2, 1.0))
        ws = self.w_short * (1.0 - u)
        wm = self.w_mid * (0.7 + 0.3 * u)
        wl = self.w_long * (0.5 + 0.5 * u)
        s = max(1e-9, ws + wm + wl)
        return (ws * y5 + wm * y15 + wl * y45) / s

    def step(self, x: float) -> Dict[str, Any]:
        y5 = self.ma5.update(x)
        y15 = self.ma15.update(x)
        y45 = self.ma45.update(x)
        m = self.metrics(x)
        std = m["std"]
        if self.mode == "ma5":
            y, used = y5, "ma5"
        elif self.mode == "ma15":
            y, used = y15, "ma15"
        elif self.mode == "ma45":
            y, used = y45, "ma45"
        elif self.mode == "tri":
            y, used = self.tri_mix(y5, y15, y45, std), "tri"
        else:
            if m["spike_ratio"] > 0.20:
                y, used = y5, "ma5"
            elif std / max(m["mad"], 1e-9) > 6.0:
                y, used = y45, "ma45"
            elif std < 0.02:
                y, used = y5, "ma5"
            else:
                y, used = self.tri_mix(y5, y15, y45, std), "tri"
        return {"y": float(y), "y5": float(y5), "y15": float(y15), "y45": float(y45), "mode": used, **m}


orch = FilterOrchestrator()

def filter_step(x: float, ingest: bool = True) -> Dict[str, Any]:
    out = orch.step(float(x))
    if ingest:
        emit_to_elastic({"ts": time.time(), "raw": float(x), **out})
    return out

filterstep = filter_step


app = FastAPI()

class SetMode(BaseModel):
    mode: str

class SetTriWeights(BaseModel):
    short: float
    mid: float
    long: float

@app.post("/set_filter_mode")
def set_filter_mode(req: SetMode) -> Dict[str, Any]:
    orch.set_mode(req.mode)
    return {"ok": True, "mode": orch.mode}

@app.post("/set_tri_weights")
def set_tri_weights(req: SetTriWeights) -> Dict[str, Any]:
    orch.set_tri_weights(req.short, req.mid, req.long)
    return {"ok": True, "weights": {"short": orch.w_short, "mid": orch.w_mid, "long": orch.w_long}}

@app.get("/get_filter_cfg")
def get_filter_cfg() -> Dict[str, Any]:
    return {"mode": orch.mode, "weights": {"short": orch.w_short, "mid": orch.w_mid, "long": orch.w_long}}

autostartactions = False
autostartdemo = False

def startactionsserver():
    def run():
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8088, log_level="warning")
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th

def demostream():
    while True:
        t = time.time()
        base = 0.8 + 0.2 * math.sin(2 * math.pi * 1.2 * t)
        noise = random.gauss(0, 0.03)
        if random.random() < 0.02:
            noise += random.choice([-1.0, 1.0]) * 0.3
        yield base + noise

def rundemoloop():
    for v in demostream():
        filter_step(v, True)

if autostartactions:
    startactionsserver()
if autostartdemo:
    rundemoloop()

stream_thread = None
stream_stop = threading.Event()
stream_rate_sec = 0.05

def stream_loop():
    for v in demostream():
        if stream_stop.is_set():
            break
        filter_step(v, True)
        time.sleep(stream_rate_sec)

def stream_start():
    global stream_thread
    if stream_thread and stream_thread.is_alive():
        return False
    stream_stop.clear()
    stream_thread = threading.Thread(target=stream_loop, daemon=True)
    stream_thread.start()
    return True

def stream_stop_fn():
    stream_stop.set()
    return True

@app.get("/stream/status")
def stream_status() -> Dict[str, Any]:
    running = stream_thread is not None and stream_thread.is_alive()
    return {"running": running}

@app.post("/stream/start")
def api_stream_start() -> Dict[str, Any]:
    ok = stream_start()
    running = stream_thread is not None and stream_thread.is_alive()
    return {"ok": ok or running, "running": running}

@app.post("/stream/stop")
def api_stream_stop() -> Dict[str, Any]:
    stream_stop_fn()
    return {"ok": True}

def startactionsserver():
    def run():
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8088, log_level="info")
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th

autostartactions = True
autostartdemo = True

def _run():
    import pathlib, sys
    print(f"run: {pathlib.Path(__file__).resolve()}", flush=True)
    startactionsserver()
    ok = stream_start()
    print(f"filter server http://127.0.0.1:8088  stream={'ON' if ok else 'ALREADY'}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass

_run()
