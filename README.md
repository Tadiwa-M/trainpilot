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
- **Auditable runs:** each run folder holds `run.yaml`, the exact `data.yaml` used, the gate report,
  the rendered `train.py`, the log and the pid.

## Setup
```bash
pip install -e ../dataset-gate -e .
ollama pull qwen2.5:3b
# training needs a Python with ultralytics + torch; point a spec at it with train_python:
tpilot plan --dataset demo_data --base yolo11n.pt --goal "high recall" --name first --train-python /path/to/python
```
