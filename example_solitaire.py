#!/usr/bin/env python3
"""Example: Solitaire game as a PLATO-ML model."""
import sys
sys.path.insert(0, "/tmp/plato-ml")
from rooms.layer import RoomState, Room, PLATOModel

# Room 1: Deal (input layer)
def deal_transform(input_state, room_state):
    """Deal cards from input specification."""
    deck_spec = input_state.data.get("deck", "standard")
    num_columns = input_state.data.get("columns", 7)
    # Simplified: just track the structure
    return RoomState(room_id="dealt", data={
        "columns": num_columns,
        "deck_type": deck_spec,
        "cards_dealt": num_columns * (num_columns + 1) // 2,
        "status": "dealt"
    })

# Room 2: Strategy (hidden layer — transforms via weights/strategy tiles)
def strategy_transform(input_state, room_state):
    """Apply strategy tiles (weights) to evaluate the game state."""
    strategy = room_state.data.get("active_strategy", "zen")
    strategies = {
        "turtle": {"foundation_weight": 0.9, "reveal_weight": 0.6, "king_weight": 0.3},
        "blitz": {"foundation_weight": 0.3, "reveal_weight": 0.9, "king_weight": 0.9},
        "zen": {"foundation_weight": 0.6, "reveal_weight": 0.7, "king_weight": 0.6},
    }
    weights = strategies.get(strategy, strategies["zen"])
    
    # Score the current position
    foundation_w = weights["foundation_weight"]
    reveal_w = weights["reveal_weight"]
    
    return RoomState(room_id="evaluated", data={
        **input_state.data,
        "strategy": strategy,
        "weights": weights,
        "position_score": foundation_w * 0.5 + reveal_w * 0.3,
        "recommended_action": "move_to_foundation" if foundation_w > 0.7 else "reveal"
    })

# Room 3: Execute (output layer)
def execute_transform(input_state, room_state):
    """Execute the recommended action."""
    action = input_state.data.get("recommended_action", "wait")
    return RoomState(room_id="executed", data={
        **input_state.data,
        "action_taken": action,
        "moves": room_state.data.get("moves", 0) + 1,
        "result": "success" if action != "wait" else "no_move"
    })

# Build the model (map of rooms = neural network)
model = PLATOModel("solitaire-agent")
model.add_room(Room("entrance", deal_transform))
model.add_room(Room("strategy_room", strategy_transform))
model.add_room(Room("execution", execute_transform))
model.connect("entrance", "strategy_room")
model.connect("strategy_room", "execution")

# Train with episodes (seasons)
episodes = [
    {"name": "easy_deal", "input": {"deck": "standard", "columns": 7}, "expected": "deal evaluate execute foundation"},
    {"name": "turtle_mode", "input": {"deck": "standard", "columns": 7, "strategy": "turtle"}, "expected": "foundation safe move"},
    {"name": "blitz_mode", "input": {"deck": "standard", "columns": 7, "strategy": "blitz"}, "expected": "reveal aggressive king"},
]

result = model.train_season(episodes)
print(f"Season 1: {result["episodes"]} episodes, achievement score: {result["mean_achievement"]:.2f}")

# Show the narrative gradient
for room_id, room in model.rooms.items():
    print(f"\n{room.describe()}")
    for g in room.state.gradient()[:3]:
        print(f"  gradient: {g['delta']}")
