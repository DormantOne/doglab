# 🦿 DOG LAB v7 3D — “Iron Wolf” (CPG Enhanced)

A single-file experiment in **3D quadruped locomotion** where a **Central Pattern Generator (CPG)** provides a rhythmic gait prior, and a **multi-timescale “liquid reservoir” actor-critic** learns **residual corrections** on top of that prior. Training is driven by an **evolutionary outer loop** that searches hyperparameters + reward weights, while the UI renders the sim in **Three.js** and shows live telemetry.

> **Primary metric:** distance traveled (`mean_distance`)  
> **UI:** Flask server (local) + Three.js dashboard (in-browser)

## Screenshot

(https://dormantone.github.io/doglab/)

## Why this exists

Learning quadruped locomotion from scratch is hard and often brittle. This version makes it easier by:

- Using a **CPG** to supply a **reasonable trotting baseline**
- Having the network learn **residuals**, not raw joint targets
- Adding **coupled torso–leg dynamics** so motion looks/feels more physical
- Rewarding **coordination** (diagonal sync, symmetry, smoothness) and penalizing **instability** and **lateral drift**
- Running a **natural-gradient-ish evolution loop** with per-gene adaptive sigma + diversity injection

## Features

### Control + learning
- **CPG Prior** (trot gait, diagonal pairs synced)
- **Residual policy**: network outputs small adjustments on CPG targets
- **Multi-timescale reservoir (“liquid” net)** feeding an **A2C-style** actor-critic head
- **Evolutionary outer loop**
  - per-gene adaptive mutation (`sigma`) driven by sensitivity
  - BLX-α crossover
  - periodic diversity injection + stagnation boost

### Physics
- 3D torso + 4 legs (12 joints) with:
  - ground contact via **mirror-floor shadow anchors**
  - friction-limited tangential forces
  - soft joint limits (damping near boundaries)
  - reaction torques from leg motors back onto torso

### UI
- Live Three.js render (orbit camera)
- “Hyperspace” population projection view
- Neural activation view (demo mode)
- Per-gene sigma/sensitivity view
- Telemetry plots (reward, distance, height)





## Requirements

- **Python**: 3.9+ (3.10/3.11 recommended)
- **OS**: macOS / Linux / Windows (works anywhere Python + Torch works)
- **Compute**: CPU is fine (default). Faster cores = faster generations.

**Python packages**
- `flask`
- `numpy`
- `torch`

The UI pulls Three.js from a CDN (no npm build step).

## Quick start

### 1) Install deps

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -U pip
pip install flask numpy torch
If torch install is annoying on your platform, use the official PyTorch install selector for your OS/Python and then come back.

2) Run
python dog_lab_v7_3d.py

Then open:
http://127.0.0.1:5005
