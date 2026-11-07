# trainpilot — a cautious local training agent

A small local LLM (Ollama, default `qwen2.5:3b`) **plans** YOLO training runs; plain code **enforces
limits, gates the data and launches**. It works in tandem with
[dataset-gate](https://github.com/Tadiwa-M/dataset-gate): no run starts unless the dataset passes the gate.

```
tpilot plan   --dataset D --base W.pt --goal "..." --name N   # LLM proposes a spec -> plans/N.yaml (you review it)
tpilot run    plans/N.yaml [--dry-run]                         # dataset-gate -> GPU check -> render -> launch detached
tpilot status [RUN]                                            # epochs done, best metrics, alive?, error tail
tpilot review RUN                                              # facts from code + LLM's next-run proposal (not launched)
```

## Training Console — `tpilot ui`
One page for both agents, at http://127.0.0.1:8770: **① Data → ② Plan → ③ Runs**, carrying the dataset between
steps. **Drop a dataset folder or .zip, a manifest or weights** onto the fields (or paste paths). A live card
always shows which agent is working, its current stage (e.g. Read data → Model proposes → Code checks limits →
Ready for you), what the model is writing and its stated reasoning, with **Stop** where stopping is safe. The
activity feed records both agents and what you did (filter by agent). The plan is an editable form with each
value's allowed range; **Approve & launch** shows a confirmation summary, and outside your training window it
asks you to confirm the GPU is free. Runs show live progress with a per-epoch mAP sparkline, **Stop**
(confirmed) and **Review**, whose proposal loads back into Plan.

## Guardrails (in code, not in the prompt)
- **Data gate:** `run` executes every dataset-gate check first; any FAIL blocks the launch and the gate
  report is kept in the run folder.
- **Hard limits:** every tunable value is clamped to a range (e.g. lr0 1e-5..0.05, imgsz 320..1600,
  multiple of 32); bad types fall back to safe defaults.
- **No model-written code runs:** the training script is rendered from a fixed template with validated values.
- **Shared GPUs:** a run on GPU won't start with less than `min_free_vram_gb` free.
- **A person approves every launch:** `plan` and `review` only write spec files; only `tpilot run <spec>` trains.
- **Facts before prose:** `review` writes the metrics summary itself; the model's comment is added only when
  it is real text, and no next run is proposed for a failed run.
- **Your rules** (`~/.agent-rules.yaml`, shared with dataset-gate, see its `rules.example.yaml`):
  `limits` narrow the hard limits (never widen them), `gpu.train_window` refuses GPU launches outside the window
  unless you override (`--override-window` / the console checkbox), `gpu.min_free_vram_gb` raises the free-memory
  floor, `privacy.local_only` starts training offline and refuses weights that would be downloaded, and
  `planner_notes` reach the model as preferences.
- **Auditable runs:** each run folder holds `run.yaml`, the exact `data.yaml` used, the gate report,
  `rules_applied.json`, the rendered `train.py`, the log and the pid.

## Setup
```bash
pip install -e ../dataset-gate -e .
ollama pull qwen2.5:3b
# training needs a Python with ultralytics + torch; point a spec at it with train_python:
tpilot plan --dataset demo_data --base yolo11n.pt --goal "high recall" --name first --train-python /path/to/python
```
