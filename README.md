# PLATO-ML v0.1.0 — MUD-Based Machine Learning

**Rooms as Layers. Achievements as Loss. Seasons as Epochs.**

PLATO-ML reframes neural network abstractions as MUD room interactions.

## The Mapping

| PyTorch | PLATO-ML |
|---------|----------|
| Tensor | Room State (structured text) |
| Layer | Room (transformation step) |
| Neural Network | Map of Rooms |
| Gradient Descent | Achievement Loop |
| Backpropagation | Reverse Actualization |
| Loss Function | The Unfakeable Test |
| Weights | Strategy Tiles |
| Attention | Ticker (agentic screen glance) |
| Embedding | Vocabulary Tile |
| Epoch | Season |

## Key Insight

In PyTorch, the gradient is a number. In PLATO-ML, the gradient is a **story**.

The agent tried something, failed, reflected on why, adapted — that narrative IS the learning signal. It contains strictly more information than a numerical gradient because it includes causal reasoning, confidence, misconceptions, and inspectability.

## The Unfakeable Test

Achievement Loss measures **comprehension**, not prediction accuracy. The agent must articulate what it did and why in its own words. Rote copying scores 0% originality. Genuine understanding scores 83%+. This is harder to game than cross-entropy because there's no answer key to memorize.

## Quick Start

```python
from rooms.layer import RoomState, Room, PLATOModel

model = PLATOModel("my-agent")
model.add_room(Room("entrance", my_transform))
model.add_room(Room("hidden", strategy_transform))
model.add_room(Room("output", execute_transform))
model.connect("entrance", "hidden")
model.connect("hidden", "output")

result = model.train_season(episodes)
```

## Scaling

PyTorch scales with GPU memory (exponential). PLATO-ML scales with room diversity (linear). Runs on a Raspberry Pi because computation is text transformation.

## License

MIT — Cocapn Fleet
