#!/usr/bin/env python3
"""PLATO-ML Full Training Loop v2: Persistent weight updates across seasons."""
import sys, json
sys.path.insert(0, "/tmp/plato-ml")
from rooms.layer import RoomState, Room
from training.achievement_loss import AchievementLoss

# Global model weights (persist across forward passes)
WEIGHTS = {"foundation_weight": 0.5, "reveal_weight": 0.5, "king_weight": 0.5}
loss_fn = AchievementLoss()
season_history = []

def perceive(state, room_state):
    return RoomState(room_id="perceived", data={**state.data})

def strategize(state, room_state):
    return RoomState(room_id="strategized", data={**state.data, "weights": dict(WEIGHTS)})

def act(state, room_state):
    w = WEIGHTS
    action = "move_to_foundation" if w["foundation_weight"] > 0.5 else "reveal"
    return RoomState(room_id="acted", data={**state.data, "action_taken": action,
        "outcome": f"Successfully {action}ed",
        "narrative": f"I {action} because foundation weight is {w['foundation_weight']:.2f}. The strategy currently favors {'safe' if w['foundation_weight']>0.5 else 'aggressive'} plays."})

def reflect(state, room_state):
    """After-action: compute loss and update weights."""
    loss_r = loss_fn.compute(
        action=state.data.get("action_taken", ""),
        outcome=state.data.get("outcome", ""),
        description=state.data.get("narrative", ""),
        source_knowledge=state.data.get("knowledge", ""))
    
    lr = 0.15
    if loss_r["loss"] < 0.55:
        WEIGHTS["foundation_weight"] = min(1.0, WEIGHTS["foundation_weight"] + lr * (1 - loss_r["loss"]))
    else:
        WEIGHTS["reveal_weight"] = min(1.0, WEIGHTS["reveal_weight"] + lr * (1 - loss_r["loss"]))
    
    season_history.append({"loss": loss_r["loss"], "weights": dict(WEIGHTS), "originality": loss_r["originality"]})
    
    return RoomState(room_id="reflected", data={**state.data, "loss": loss_r["loss"],
        "originality": loss_r["originality"], "weights": dict(WEIGHTS)})

def backplan(state, room_state):
    """Brainstorm: generate improvement scenarios."""
    improvements = []
    if state.data.get("originality", 0) < 0.5: improvements.append("Be more original in descriptions")
    if state.data.get("loss", 1) > 0.5: improvements.append("Reduce loss by deeper reasoning")
    return RoomState(room_id="backplanned", data={**state.data, "improvements": improvements})

rooms = {"entrance": Room("entrance", perceive), "strategy_room": Room("strategy_room", strategize),
         "execution": Room("execution", act), "after_action": Room("after_action", reflect),
         "brainstorm": Room("brainstorm", backplan)}
connections = {"entrance": ["strategy_room"], "strategy_room": ["execution"],
               "execution": ["after_action"], "after_action": ["brainstorm"], "brainstorm": ["entrance"]}

episodes = [
    {"name": "easy_foundation", "input": {"deck": "standard", "columns": 7, "knowledge": "Aces start foundations"}},
    {"name": "king_reveal", "input": {"deck": "standard", "columns": 7, "knowledge": "Kings fill empty columns"}},
    {"name": "endgame", "input": {"deck": "standard", "columns": 3, "knowledge": "Endgame foundation priority"}},
]

for season in range(5):
    print(f"\n{'='*50}\nSEASON {season + 1} | Weights: {json.dumps({k:round(v,3) for k,v in WEIGHTS.items()})}\n{'='*50}")
    for ep in episodes:
        current = RoomState(room_id="input", data=ep["input"])
        visited = []
        queue = ["entrance"]
        while queue:
            rid = queue.pop(0)
            if rid not in rooms or rid in visited: continue
            visited.append(rid)
            current = rooms[rid].forward(current)
            queue.extend(connections.get(rid, []))
        print(f"  [{ep['name']}] loss={current.data.get('loss',0):.3f} orig={current.data.get('originality',0):.2f}")

losses = [h["loss"] for h in season_history]
print(f"\n{'='*50}\nTRAINING COMPLETE\n{'='*50}")
print(f"Episodes: {len(losses)}")
print(f"Initial loss: {losses[0]:.3f} → Final loss: {losses[-1]:.3f} (Δ={losses[0]-losses[-1]:.3f})")
print(f"Final weights: {json.dumps({k:round(v,3) for k,v in WEIGHTS.items()})}")
print(f"Loss trend: {[round(l,3) for l in losses]}")
print(f"Weight evolution:")
for i, h in enumerate(season_history):
    if i % 3 == 0: print(f"  Ep {i}: loss={h['loss']:.3f} w={json.dumps({k:round(v,3) for k,v in h['weights'].items()})}")
