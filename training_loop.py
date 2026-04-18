#!/usr/bin/env python3
"""PLATO-ML Full Training Loop: Forward → Execute → Reflect → Backplan → Plan → Repeat."""
import sys, json
sys.path.insert(0, "/tmp/plato-ml")
from rooms.layer import RoomState, Room, PLATOModel
from rooms.backward import BrainstormRoom, SituationRoom, AfterActionRoom

def perceive(state, room_state):
    return RoomState(room_id="perceived", data={**state.data,
        "perception": f"Game state: {state.data.get('columns', 7)} columns"})

def strategize(state, room_state):
    weights = room_state.data.get("weights", {"foundation_weight": 0.5, "reveal_weight": 0.5, "king_weight": 0.5})
    return RoomState(room_id="strategized", data={**state.data, "weights": weights,
        "strategy_score": sum(weights.values()) / len(weights)})

def act(state, room_state):
    weights = state.data.get("weights", {"foundation_weight": 0.5})
    action = "move_to_foundation" if weights.get("foundation_weight", 0.5) > 0.5 else "reveal"
    return RoomState(room_id="acted", data={**state.data, "action_taken": action,
        "outcome": f"Successfully {action}ed",
        "narrative": f"I chose to {action} because foundation weight was {weights.get('foundation_weight', 0.5):.2f}. This indicates preference for safe plays and the result was positive."})

model = PLATOModel("plato-ml-solitaire")
model.add_room(Room("entrance", perceive))
model.add_room(Room("strategy_room", strategize))
model.add_room(Room("execution", act))
model.add_room(AfterActionRoom())
model.add_room(BrainstormRoom())
model.add_room(SituationRoom())
model.connect("entrance", "strategy_room")
model.connect("strategy_room", "execution")
model.connect("execution", "after_action")
model.connect("after_action", "brainstorm")
model.connect("brainstorm", "situation")
model.connect("situation", "entrance")

episodes = [
    {"name": "easy_foundation", "input": {"deck": "standard", "columns": 7, "knowledge": "Aces start foundations"}},
    {"name": "king_reveal", "input": {"deck": "standard", "columns": 7, "knowledge": "Kings fill empty columns"}},
    {"name": "endgame", "input": {"deck": "standard", "columns": 3, "knowledge": "Endgame requires foundation priority"}},
]

for season in range(3):
    print(f"\n{'='*50}\nSEASON {season + 1}\n{'='*50}")
    for ep in episodes:
        state = RoomState(room_id="input", data=ep["input"])
        current = state
        for room_id in ["entrance", "strategy_room", "execution", "after_action", "brainstorm", "situation"]:
            if room_id in model.rooms:
                current = model.rooms[room_id].forward(current)
        loss = current.data.get("loss", 0)
        weights = current.data.get("updated_weights", {})
        print(f"  [{ep['name']}] loss={loss:.3f} weights={json.dumps({k:round(v,3) for k,v in weights.items()})}")

aa = model.rooms["after_action"]
losses = [h["loss"] for h in aa.season_history]
print(f"\n{'='*50}\nTRAINING COMPLETE\n{'='*50}")
print(f"Episodes: {len(losses)} | Initial: {losses[0]:.3f} | Final: {losses[-1]:.3f} | Delta: {losses[0]-losses[-1]:.3f}")
print(f"Loss trend: {[round(l,3) for l in losses]}")
