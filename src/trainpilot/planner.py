"""LLM planner (local Ollama): proposes a RunSpec and reviews finished runs. It never launches anything.

plan()   dataset summary + goal + hardware + past runs  ->  a validated RunSpec saved for a person to approve
review() a finished run's metrics + spec                 ->  plain-English summary + next-experiment proposal
The model fills JSON; code clamps every value (spec.validate) and falls back to safe defaults. Your rules file's
planner_notes go into the prompt as preferences (they can't widen any limit). Answers are streamed and capped;
on_event gets each stage and the text as the model writes it, so a person can follow along.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

import yaml

from .launch import free_vram_gb
from .spec import LIMITS, OPTIMIZERS, RunSpec, from_dict, validate

OLLAMA = "http://localhost:11434/api/chat"
TUNABLE = ["imgsz", "epochs", "patience", "batch", "lr0", "optimizer", "fliplr", "flipud", "degrees", "mosaic",
           "scale", "notes"]


def _chat(llm: str, system: str, user: str, json_mode: bool = True, emit=None) -> str:
    body = {"model": llm, "stream": True, "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 900},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    text, n, t0, last = "", 0, time.time(), 0.0
    with urllib.request.urlopen(req, timeout=120) as r:                  # timeout per read: a stall fails fast
        for line in r:
            if not line.strip():
                continue
            chunk = json.loads(line)
            text += (chunk.get("message") or {}).get("content") or ""
            n += 1
            if emit and (time.time() - last > 0.4 or chunk.get("done")):
                emit("thinking", tokens=n, seconds=round(time.time() - t0), text=text[-400:])
                last = time.time()
            if chunk.get("done"):
                break
    return text


def _prefs() -> list[str]:
    try:
        from dsgate.rules import load
        return list(load().get("planner_notes") or [])
    except Exception:  # noqa: BLE001 — no rules file / broken file: plan without preferences
        return []


def dataset_summary(dataset: str) -> dict:
    from dsgate.core import load
    ds = load(dataset)
    per_cls = ds.boxes[ds.boxes.split == "train"].cls.value_counts().to_dict()
    return {"classes": ds.names, "images_per_split": ds.tiles.split.value_counts().to_dict(),
            "train_boxes_per_class": {ds.names[int(k)]: int(v) for k, v in per_cls.items() if 0 <= int(k) < len(ds.names)}}


PLAN_STAGES = ["Read data", "Model proposes", "Code checks limits", "Ready for you"]


def plan(dataset: str, base_weights: str, goal: str, name: str, device: str = "0", llm: str = "qwen2.5:3b",
         past: list[dict] | None = None, on_event=None) -> tuple[RunSpec, list[str]]:
    def emit(kind, **kw):
        if on_event:
            on_event({"kind": kind, **kw})
    emit("stage", stage="Read data", text="summarising the dataset and checking free GPU memory")
    vram = free_vram_gb() if device != "cpu" else None
    system = ("You plan ONE training run for an Ultralytics YOLO object detector. Reply with a JSON object with "
              f"exactly these keys: {TUNABLE}. Allowed ranges: "
              + ", ".join(f"{k} {lo}..{hi}" for k, (lo, hi) in LIMITS.items() if k in TUNABLE)
              + f"; optimizer one of {sorted(OPTIMIZERS)}. Fine-tuning from good weights needs a low lr0 (~0.001) and "
              "few epochs; aerial imagery has no canonical orientation, so flips both ways help. Keep batch small "
              "enough for the free GPU memory. Follow researcher_preferences. Put your one-paragraph reasoning in "
              "'notes'.")
    summ = dataset_summary(dataset)
    emit("result", text=f"{sum(summ['images_per_split'].values())} images, {len(summ['classes'])} classes; free GPU "
                        f"{'n/a' if vram is None else f'{vram:.1f} GB'}; {len(past or [])} past run(s) as context")
    prefs = _prefs()
    if prefs:
        emit("result", text=f"{len(prefs)} preferences from your rules file go to the model")
    user = json.dumps({"goal": goal, "base_weights": base_weights, "device": device,
                       "free_gpu_memory_gb": None if vram is None else round(vram, 1),
                       "researcher_preferences": prefs, "dataset": summ, "past_runs": past or []})
    spec = RunSpec(name=name, dataset=dataset, base_weights=base_weights, device=device)
    notes = []
    emit("stage", stage="Model proposes", text=f"{llm} is choosing settings")
    try:
        proposal = json.loads(_chat(llm, system, user, emit=emit))
        for k in TUNABLE:
            if k in proposal:
                setattr(spec, k, proposal[k])
        emit("proposal", data=proposal)                                  # raw, before code clamps it
        if isinstance(proposal.get("notes"), str):
            emit("reason", text=proposal["notes"][:600])
    except Exception as exc:  # noqa: BLE001 — model down / bad JSON: keep the safe defaults
        notes.append(f"planner fell back to defaults ({type(exc).__name__}: {exc})")
        emit("warn", text=notes[-1])
    emit("stage", stage="Code checks limits", text="clamping every value to its allowed range")
    try:
        notes += validate(spec)
    except (TypeError, ValueError) as exc:                      # model put text where a number belongs
        notes.append(f"invalid proposal ({exc}); using defaults")
        spec = RunSpec(name=name, dataset=dataset, base_weights=base_weights, device=device)
        notes += validate(spec)
    for n in notes:
        emit("warn", text=f"code corrected: {n}")
    emit("stage", stage="Ready for you", text="review the settings, edit anything, then approve")
    return spec, notes


def review(run_status: dict, spec: RunSpec, llm: str = "qwen2.5:3b", on_event=None) -> dict:
    system = ("You review a finished YOLO training run for a researcher. Reply with JSON: {\"summary\": str, "
              "\"next_run\": {" + ", ".join(f'"{k}": ...' for k in TUNABLE) + "}}. The summary is 3-5 plain sentences "
              "using only numbers given to you. next_run changes at most two settings from the current spec and "
              "says why in its notes. Respect researcher_preferences.")
    user = json.dumps({"spec": {k: getattr(spec, k) for k in TUNABLE}, "status": run_status,
                       "researcher_preferences": _prefs()})
    try:
        out = json.loads(_chat(llm, system, user,
                               emit=(lambda k, **kw: on_event({"kind": k, **kw})) if on_event else None))
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
    # one experiment at a time: a proposal that changes more than two settings can't say what helped
    changed = [k for k in TUNABLE if k != "notes" and getattr(nxt, k) != getattr(spec, k)]
    if len(changed) > 2:
        return {"summary": out["summary"] + f"\nNo next run proposed: the model changed {len(changed)} settings "
                                            f"({', '.join(changed)}); at most two are allowed so the result is readable.",
                "next_spec": None, "notes": [f"rejected: {len(changed)} settings changed"]}
    return {"summary": out.get("summary", ""), "next_spec": nxt, "notes": notes}


def save_plan(spec: RunSpec, notes: list[str], path: Path) -> None:
    doc = asdict(spec)
    doc["_validation_notes"] = notes
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
