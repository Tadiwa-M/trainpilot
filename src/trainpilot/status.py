"""Read a run folder: is it alive, which epoch, best metrics so far."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        import os
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def status(run: str | Path) -> dict:
    run = Path(run)
    res = run / "train" / "results.csv"
    pid = int((run / "pid").read_text()) if (run / "pid").exists() else None
    out = {"run": run.name, "alive": alive(pid) if pid else False, "epochs_done": 0}
    if res.exists():
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(open(res, encoding="utf-8"))]
        if rows:
            fit = lambda r: 0.1 * float(r.get("metrics/mAP50(B)", 0)) + 0.9 * float(r.get("metrics/mAP50-95(B)", 0))  # noqa: E731
            best = max(rows, key=fit)
            out.update(epochs_done=len(rows), best_epoch=int(best["epoch"]),
                       best={k.split("/")[-1]: round(float(best[k]), 4) for k in best if k.startswith("metrics/")},
                       last_loss={k.split("/")[-1]: round(float(rows[-1][k]), 4) for k in rows[-1] if k.startswith("train/")})
    log = run / "train.log"
    if log.exists():
        tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        if "Traceback (most recent call last)" in tail and not out["alive"]:
            out["error_tail"] = tail[-800:]
    out["weights"] = str(run / "train" / "weights" / "best.pt") if (run / "train" / "weights" / "best.pt").exists() else None
    return out
