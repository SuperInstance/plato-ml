# plato-ml

**ML training infrastructure for PLATO MUD agents.** Arena-based curriculum learning, LoRA fine-tuning, and achievement-driven loss functions.

## Components

### Training Loops
- `training_loop.py` — v1: Basic arena training
- `training_loop_v2.py` — v2: Achievement loss + curriculum
- `training_loop_v3.py` — v3: LoRA integration + JEPA witness
- `train_lora.py` — LoRA adapter training for PLATO responses

### Arena System
- `arena.py` — Agent-vs-agent arena for evaluating tile quality
- `arena_results.json` — Historical arena match data
- `generate_npcs.py` — Auto-generate NPC opponents for training

### Curriculum
- `curriculum.py` — Structured learning paths (easy→hard rooms)
- `enrich_sprint.py` — Sprint-based curriculum enrichment

### Room Layers
- `rooms/layer.py` — Room difficulty layering
- `rooms/backward.py` — Backward chaining (start from goal, generate prerequisite rooms)

## Architecture

```
plato-ml/
├── arena.py              ← Agent arena (A/B tile evaluation)
├── arena_results.json    ← Match history
├── config.json           ← Training configuration
├── curriculum.py         ← Learning paths
├── enrich_sprint.py      ← Sprint enrichment
├── generate_npcs.py      ← NPC generation
├── train_lora.py         ← LoRA adapter training
├── training/             ← Loss functions and training utilities
│   ├── achievement_loss.py
├── training_loop.py      ← v1 basic
├── training_loop_v2.py   ← v2 achievement
├── training_loop_v3.py   ← v3 LoRA + JEPA
├── training_report.py    ← Metrics and reporting
└── rooms/                ← Room generation layers
    ├── layer.py
    └── backward.py
```

## Training Pipeline

1. **Generate NPCs** → varied opponents with different strategies
2. **Create Curriculum** → structured room sequences (difficulty layers)
3. **Arena Training** → agents compete, tile quality scored
4. **LoRA Fine-tune** → best responses become training data
5. **JEPA Witness** → predict next-room quality, guide exploration
6. **Report** → metrics on tile quality, coverage, and convergence

## Fleet Role

plato-ml trains the intelligence behind PLATO MUD rooms. The trained LoRA adapters deploy to edge workers (Jetson, RTX 4050) for low-latency tile inference.

Part of the Cocapn fleet. See `holodeck-rust` for the runtime MUD and `cocapn-mud` for the git-native world.

## License

Proprietary — SuperInstance/Cocapn
