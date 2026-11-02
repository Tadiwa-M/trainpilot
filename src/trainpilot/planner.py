"""LLM planner (local Ollama): proposes a RunSpec and reviews finished runs. It never launches anything.

plan()   dataset summary + goal + hardware + past runs  ->  a validated RunSpec saved for a person to approve
review() a finished run's metrics + spec                 ->  plain-English summary + next-experiment proposal
The model fills JSON; code clamps every value (spec.validate) and falls back to safe defaults.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict
from pathlib import Path

import yaml

from .launch import free_vram_gb
from .spec import LIMITS, OPTIMIZERS, RunSpec, from_dict, validate

OLLAMA = "http://localhost:11434/api/chat"
TUNABLE = ["imgsz", "epochs", "patience", "batch", "lr0", "optimizer", "fliplr", "flipud", "degrees", "mosaic",
           "scale", "notes"]


def _chat(llm: str, system: str, user: str, json_mode: bool = True) -> str:
    body = {"model": llm, "stream": False, "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"]


def dataset_summary(dataset: str) -> dict:
    from dsgate.core import load
    ds = load(dataset)
    per_cls = ds.boxes[ds.boxes.split == "train"].cls.value_counts().to_dict()
    return {"classes": ds.names, "images_per_split": ds.tiles.split.value_counts().to_dict(),
            "train_boxes_per_class": {ds.names[int(k)]: int(v) for k, v in per_cls.items() if 0 <= int(k) < len(ds.names)}}


def plan(dataset: str, base_weights: str, goal: str, name: str, device: str = "0", llm: str = "qwen2.5:3b",
         past: list[dict] | None = None) -> tuple[RunSpec, list[str]]:
    vram = free_vram_gb() if device != "cpu" else None
    system = ("You plan ONE training run for an Ultralytics YOLO object detector. Reply with a JSON object with "
              f"exactly these keys: {TUNABLE}. Allowed ranges: "
              + ", ".join(f"{k} {lo}..{hi}" for k, (lo, hi) in LIMITS.items() if k in TUNABLE)
              + f"; optimizer one of {sorted(OPTIMIZERS)}. Fine-tuning from good weights needs a low lr0 (~0.001) and "
              "few epochs; aerial imagery has no canonical orientation, so flips both ways help. Keep batch small "
              "enough for the free GPU memory. Put your one-paragraph reasoning in 'notes'.")
    user = json.dumps({"goal": goal, "base_weights": base_weights, "device": device,
                       "free_gpu_memory_gb": None if vram is None else round(vram, 1),
                       "dataset": dataset_summary(dataset), "past_runs": past or []})
    spec = RunSpec(name=name, dataset=dataset, base_weights=base_weights, device=device)
    notes = []
    try:
        proposal = json.loads(_chat(llm, system, user))
        for k in TUNABLE:
            if k in proposal:
                setattr(spec, k, proposal[k])
    except Exception as exc:  # noqa: BLE001 — model down / bad JSON: keep the safe defaults
        notes.append(f"planner fell back to defaults ({type(exc).__name__}: {exc})")
    try:
        notes += validate(spec)
    except (TypeError, ValueError) as exc:                      # model put text where a number belongs
        notes.append(f"invalid proposal ({exc}); using defaults")
        spec = RunSpec(name=name, dataset=dataset, base_weights=base_weights, device=device)
        notes += validate(spec)
    return spec, notes


def review(run_status: dict, spec: RunSpec, llm: str = "qwen2.5:3b") -> dict:
    system = ("You review a finished YOLO training run for a researcher. Reply with JSON: {\"summary\": str, "
              "\"next_run\": {" + ", ".join(f'"{k}": ...' for k in TUNABLE) + "}}. The summary is 3-5 plain sentences "
              "using only numbers given to you. next_run changes at most two settings from the current spec and "
              "says why in its notes.")
    user = json.dumps({"spec": {k: getattr(spec, k) for k in TUNABLE}, "status": run_status})
    try:
        out = json.loads(_chat(llm, system, user))
    except Exception as exc:  # noqa: BLE001
        return {"summary": f"(no review: {exc})", "next_spec": None, "notes": []}
    # facts first, from code; the model's prose only if it is actually prose (small models return junk)
    b = run_status.get("best", {})
    facts = (f"{run_status.get('epochs_done', 0)} epochs trained, best at epoch {run_status.get('best_epoch')}: "
             f"precision {b.get('precision(B)')}, recall {b.get('recall(B)')}, mAP50 {b.get('mAP50(B)')}, "
             f"mAP50-95 {b.get('mAP50-95(B)')} (on the dataset's own val split).")
    llm_text = out.get("summary")
    out["summary"] = facts + (f"\nModel's comment: {llm_text}" if isinstance(llm_text, str) and len(llm_text.split()) >= 6 else "")
    nxt = from_dict({**asdict(spec), **{k: v for k, v in (out.get("next_run") or {}).items() if k in TUNABLE}})
    nxt.name = spec.name + "_next"
    try:
        notes = validate(nxt)
    except (TypeError, ValueError) as exc:
        return {"summary": out.get("summary", ""), "next_spec": None, "notes": [f"invalid next_run: {exc}"]}
    return {"summary": out.get("summary", ""), "next_spec": nxt, "notes": notes}


def save_plan(spec: RunSpec, notes: list[str], path: Path) -> None:
    doc = asdict(spec)
    doc["_validation_notes"] = notes
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
