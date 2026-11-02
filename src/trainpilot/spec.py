"""Run spec: everything a training run needs, with hard limits enforced in code.

The LLM planner proposes values; validate() clamps or rejects them. Nothing outside these ranges
can reach the training script, whatever the model writes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

LIMITS = {                       # field: (min, max)
    "imgsz": (320, 1600), "epochs": (1, 300), "patience": (3, 50), "batch": (1, 64),
    "lr0": (1e-5, 0.05), "fliplr": (0.0, 1.0), "flipud": (0.0, 1.0), "degrees": (0.0, 30.0),
    "mosaic": (0.0, 1.0), "scale": (0.0, 0.9), "time_hours": (0.05, 24.0), "min_free_vram_gb": (1.0, 24.0),
}
INT_FIELDS = {"imgsz", "epochs", "patience", "batch"}
OPTIMIZERS = {"auto", "SGD", "AdamW", "Adam"}


@dataclass
class RunSpec:
    name: str
    dataset: str                         # folder with data.yaml (YOLO layout)
    base_weights: str                    # .pt to warm-start from (or e.g. yolo11s.pt)
    imgsz: int = 640
    epochs: int = 50
    patience: int = 10
    batch: int = 8
    lr0: float = 0.001
    optimizer: str = "AdamW"
    fliplr: float = 0.5
    flipud: float = 0.0
    degrees: float = 0.0
    mosaic: float = 1.0
    scale: float = 0.5
    device: str = "0"                    # "0" = first GPU, "cpu"
    time_hours: float | None = None      # optional wall-clock cap
    seed: int = 0
    gate_manifest: str | None = None     # manifest CSV for dataset-gate's time/GPS checks
    gate_allow_missing: list[str] = field(default_factory=list)
    min_free_vram_gb: float = 4.0        # don't start if the GPU has less free (shared machines)
    train_python: str | None = None      # python with ultralytics+torch (defaults to the current one)
    notes: str = ""                      # why this run exists (the planner's reasoning)


def validate(spec: RunSpec) -> list[str]:
    """Clamp numeric fields into LIMITS, fix the obvious, return a list of what was changed/refused."""
    notes = []
    for k, (lo, hi) in LIMITS.items():
        v = getattr(spec, k)
        if v is None:
            continue
        cast = int if k in INT_FIELDS else float
        nv = cast(min(max(float(v), lo), hi))
        if nv != v:
            notes.append(f"{k}: {v} -> {nv} (allowed {lo}..{hi})")
        setattr(spec, k, nv)
    if spec.imgsz % 32:
        nv = int(round(spec.imgsz / 32) * 32)
        notes.append(f"imgsz: {spec.imgsz} -> {nv} (multiple of 32)")
        spec.imgsz = nv
    if spec.optimizer not in OPTIMIZERS:
        notes.append(f"optimizer: {spec.optimizer!r} -> 'AdamW' (allowed {sorted(OPTIMIZERS)})")
        spec.optimizer = "AdamW"
    if spec.device not in ("cpu",) and not spec.device.replace(",", "").isdigit():
        notes.append(f"device: {spec.device!r} -> '0'")
        spec.device = "0"
    if not (Path(spec.dataset) / "data.yaml").is_file():
        notes.append(f"REFUSED: {spec.dataset} has no data.yaml")
    if not spec.base_weights.endswith(".pt"):
        notes.append(f"REFUSED: base_weights must be a .pt file, got {spec.base_weights!r}")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in spec.name)[:60] or "run"
    if safe != spec.name:
        notes.append(f"name: {spec.name!r} -> {safe!r}")
        spec.name = safe
    return notes


def from_dict(d: dict) -> RunSpec:
    known = {f.name for f in fields(RunSpec)}
    return RunSpec(**{k: v for k, v in d.items() if k in known})


def save(spec: RunSpec, path: Path) -> None:
    path.write_text(yaml.safe_dump(asdict(spec), sort_keys=False), encoding="utf-8")


def load(path: str | Path) -> RunSpec:
    return from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
