"""Launch a run: dataset gate -> GPU check -> render script from a fixed template -> start detached.

Guardrails:
  * dataset-gate must PASS (any FAIL blocks the run; its report is kept in the run folder);
  * on a GPU device, at least spec.min_free_vram_gb must be free (shared machines);
  * the training script is rendered from TEMPLATE with validated values only — no model-written code
    is ever executed;
  * the run folder records spec, gate report, script, log and pid, so every run is auditable.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .spec import RunSpec, save, validate

TEMPLATE = '''"""Rendered by trainpilot from run.yaml — do not edit; change the spec and re-launch."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO({base_weights!r})
    model.train(data={data_yaml!r}, project={project!r}, name="train", exist_ok=True,
                imgsz={imgsz}, epochs={epochs}, patience={patience}, batch={batch}, lr0={lr0},
                optimizer={optimizer!r}, fliplr={fliplr}, flipud={flipud}, degrees={degrees},
                mosaic={mosaic}, scale={scale}, device={device!r}, seed={seed}, workers=4,
                plots=True, verbose=True{time_arg})
'''


def free_vram_gb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        total, used = (float(x) for x in out.split(","))
        return (total - used) / 1024
    except Exception:  # noqa: BLE001 — no NVIDIA GPU / driver
        return None


def gate(spec: RunSpec, out: Path) -> tuple[str, list[dict]]:
    """Run dataset-gate in-process; returns (verdict, compact results)."""
    from dsgate import checks, report
    from dsgate.core import load
    out.mkdir(parents=True, exist_ok=True)
    ds = load(spec.dataset, spec.gate_manifest)
    opts = {"out_dir": str(out), "allow_missing": spec.gate_allow_missing}
    res = []
    for name, fn in checks.CHECKS.items():
        try:
            res.append(fn(ds, opts))
        except Exception as exc:  # noqa: BLE001
            res.append(checks.Result(name, "FAIL", f"check crashed: {type(exc).__name__}: {exc}"))
    v = report.verdict(res)
    report.write(res, out, str(Path(spec.dataset).resolve()), v)
    return v, [{"check": r.check, "status": r.status, "summary": r.summary} for r in res]


def launch(spec: RunSpec, runs_dir: Path, dry_run: bool = False) -> dict:
    notes = validate(spec)
    if any(n.startswith("REFUSED") for n in notes):
        return {"ok": False, "reason": "; ".join(n for n in notes if n.startswith("REFUSED")), "notes": notes}
    run = runs_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{spec.name}"
    run.mkdir(parents=True, exist_ok=False)
    save(spec, run / "run.yaml")
    verdict, gate_res = gate(spec, run / "gate")
    (run / "gate.json").write_text(json.dumps({"verdict": verdict, "results": gate_res}, indent=1))
    if verdict != "PASSED":
        fails = [g for g in gate_res if g["status"] == "FAIL"]
        return {"ok": False, "run": str(run), "reason": "dataset-gate BLOCKED the run",
                "failed_checks": fails, "notes": notes, "gate_report": str(run / "gate" / "report.html")}
    if spec.device != "cpu":
        free = free_vram_gb()
        if free is not None and free < spec.min_free_vram_gb:
            return {"ok": False, "run": str(run), "notes": notes,
                    "reason": f"only {free:.1f} GB GPU memory free (< {spec.min_free_vram_gb} GB) — someone else is "
                              f"using the GPU; try later or use device 'cpu'"}
    time_arg = f", time={spec.time_hours}" if spec.time_hours else ""
    # the run gets its own data.yaml with an absolute root: relative 'path:' entries resolve against the
    # training process's working folder (the run folder), and the copy records exactly what was trained on
    import yaml
    cfg = yaml.safe_load((Path(spec.dataset) / "data.yaml").read_text(encoding="utf-8"))
    cfg["path"] = str(Path(spec.dataset).resolve()).replace("\\", "/")
    for k in ("train", "val", "test"):
        if k in cfg and isinstance(cfg[k], str) and Path(cfg[k]).is_absolute() is False:
            cfg[k] = cfg[k].replace("\\", "/")
    (run / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    script = TEMPLATE.format(base_weights=spec.base_weights, data_yaml=str(run / "data.yaml"),
                             project=str(run), time_arg=time_arg,
                             **{k: v for k, v in asdict(spec).items()
                                if k in ("imgsz", "epochs", "patience", "batch", "lr0", "optimizer", "fliplr",
                                         "flipud", "degrees", "mosaic", "scale", "device", "seed")})
    (run / "train.py").write_text(script, encoding="utf-8")
    if dry_run:
        return {"ok": True, "run": str(run), "dry_run": True, "notes": notes}
    py = spec.train_python or sys.executable
    log = open(run / "train.log", "w", encoding="utf-8")
    flags = 0x00000008 | 0x00000200 if sys.platform == "win32" else 0   # DETACHED_PROCESS | NEW_PROCESS_GROUP
    p = subprocess.Popen([py, str(run / "train.py")], cwd=run, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=flags, close_fds=True)
    (run / "pid").write_text(str(p.pid))
    return {"ok": True, "run": str(run), "pid": p.pid, "notes": notes}
