#!/usr/bin/env python3
"""PLATO-ML v4: Training loop with curriculum scheduling, metrics tracking, and state persistence.

Enhancements over v3:
- Curriculum scheduling (easy rooms first, progressively harder)
- Metrics tracking (loss, originality, win rate over time)
- Save/load training state for resumable training
- Varied room scenarios generated programmatically
"""
import sys, json, os, time, urllib.request
sys.path.insert(0, "/tmp/plato-ml")
from rooms.layer import RoomState, Room
from training.achievement_loss import AchievementLoss

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq(system, prompt, model="llama-3.1-8b-instant"):
    body = json.dumps({"model": model, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ], "max_tokens": 100, "temperature": 0.85}).encode()
    req = urllib.request.Request(GROQ_URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {GROQ_KEY}",
        "User-Agent": "curl/7.88"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[fallback] Action taken based on weights."

WEIGHTS = {"foundation_weight": 0.5, "reveal_weight": 0.5, "king_weight": 0.5}
loss_fn = AchievementLoss()
history = []

def perceive(state, room_state):
    return RoomState(room_id="perceived", data={**state.data})

def strategize(state, room_state):
    return RoomState(room_id="strategized", data={**state.data, "weights": dict(WEIGHTS)})

def act(state, room_state):
    w = WEIGHTS
    action = "move_to_foundation" if w["foundation_weight"] > 0.5 else "reveal_hidden_card"
    narrative = groq(
        "You are a solitaire-playing agent. Explain your move in 1-2 sentences.",
        f"Weights: foundation={w['foundation_weight']:.2f}, reveal={w['reveal_weight']:.2f}. Action: {action}."
    )
    return RoomState(room_id="acted", data={**state.data, "action_taken": action,
        "outcome": f"Successfully {action}ed", "narrative": narrative})

def reflect(state, room_state):
    loss_r = loss_fn.compute(
        action=state.data.get("action_taken", ""),
        outcome=state.data.get("outcome", ""),
        description=state.data.get("narrative", ""),
        source_knowledge=state.data.get("knowledge", ""))
    lr = 0.12
    if loss_r["loss"] < 0.55:
        WEIGHTS["foundation_weight"] = min(1.0, WEIGHTS["foundation_weight"] + lr * (1 - loss_r["loss"]))
    else:
        WEIGHTS["reveal_weight"] = min(1.0, WEIGHTS["reveal_weight"] + lr * (1 - loss_r["loss"]))
    history.append({"loss": loss_r["loss"], "originality": loss_r["originality"],
                     "weights": dict(WEIGHTS), "narrative": state.data.get("narrative", "")[:80]})
    return RoomState(room_id="reflected", data={**state.data, "loss": loss_r["loss"], "originality": loss_r["originality"]})

rooms = {"entrance": Room("entrance", perceive), "strategy_room": Room("strategy_room", strategize),
         "execution": Room("execution", act), "after_action": Room("after_action", reflect)}

# Curriculum: easy → hard
CURRICULUM = [
    {"name": "foundation", "difficulty": 1, "input": {"deck": "standard", "columns": 7, "knowledge": "Aces start foundations. Build up by suit."}},
    {"name": "reveal", "difficulty": 1, "input": {"deck": "standard", "columns": 7, "knowledge": "Face-down cards must be revealed before playing."}},
    {"name": "king_empty", "difficulty": 2, "input": {"deck": "standard", "columns": 7, "knowledge": "Only kings can fill empty columns."}},
    {"name": "multi_move", "difficulty": 2, "input": {"deck": "standard", "columns": 7, "knowledge": "Moving stacks reveals hidden cards. Prioritize face-down cards."}},
    {"name": "endgame", "difficulty": 3, "input": {"deck": "standard", "columns": 7, "knowledge": "Track all face-down cards. Plan moves 3-4 steps ahead."}},
    {"name": "competitive", "difficulty": 3, "input": {"deck": "standard", "columns": 7, "knowledge": "Arena play: balance speed vs accuracy. Suboptimal moves can bait opponents."}},
]

def generate_scenarios(count=10):
    """Generate varied training scenarios programmatically."""
    knowledge_pool = [
        "Aces are always safe to play to foundations.",
        "Empty columns are valuable — don't waste them on non-kings.",
        "Red kings go on black queens, black kings on red queens.",
        "If two moves are equal, prefer the one that reveals a face-down card.",
        "Track which cards are in foundations to avoid redundant moves.",
        "Late game: focus on completing suits rather than building columns.",
        "In competitive play, sometimes blocking opponents beats advancing.",
        "Cycling through the deck costs moves — minimize deck passes.",
    ]
    scenarios = []
    for i in range(count):
        scenarios.append({
            "name": f"generated_{i}",
            "difficulty": min(3, 1 + i // 3),
            "input": {"deck": "standard", "columns": 7, "knowledge": knowledge_pool[i % len(knowledge_pool)]}
        })
    return scenarios

def save_state(filepath="/tmp/plato-ml/training_state.json"):
    """Save training state for resumption."""
    state = {"weights": dict(WEIGHTS), "history": history, "timestamp": time.time()}
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2)

def load_state(filepath="/tmp/plato-ml/training_state.json"):
    """Load training state from file."""
    global WEIGHTS, history
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath) as f:
            state = json.load(f)
        WEIGHTS.update(state.get("weights", WEIGHTS))
        history.extend(state.get("history", []))
        return True
    except:
        return False

def run_episode(ep):
    """Run a single training episode through all rooms."""
    current = RoomState(room_id="input", data=ep["input"])
    for rid in ["entrance", "strategy_room", "execution", "after_action"]:
        if rid in rooms:
            current = rooms[rid].forward(current)
    return current

def print_metrics():
    """Print training metrics summary."""
    if not history:
        return
    losses = [h["loss"] for h in history]
    origs = [h["originality"] for h in history]
    wins = sum(1 for h in history if h["loss"] < 0.5)
    n = len(history)
    print(f"\n  Metrics ({n} episodes):")
    print(f"    Avg loss: {sum(losses)/n:.3f} | Avg originality: {sum(origs)/n:.3f}")
    print(f"    Win rate: {wins}/{n} ({100*wins//n}%)")
    print(f"    Weights: {json.dumps({k:round(v,3) for k,v in WEIGHTS.items()})}")

if __name__ == "__main__":
    print("PLATO-ML v4: Curriculum Training with Metrics & State Persistence")
    print("=" * 60)

    if load_state():
        print("  Resuming from checkpoint...")

    for season in range(3):
        max_diff = 1 + season
        eligible = [ep for ep in CURRICULUM if ep["difficulty"] <= max_diff]
        print(f"\nSeason {season+1} (difficulty ≤ {max_diff}) | {len(eligible)} episodes")
        for ep in eligible:
            run_episode(ep)
            latest = history[-1] if history else {}
            print(f"  [{ep['name']}] loss={latest.get('loss',0):.3f} orig={latest.get('originality',0):.2f}")
        print_metrics()
        save_state()

    print(f"\n{'=' * 60}")
    print(f"FINAL: {len(history)} episodes across {season+1} seasons")
    print_metrics()
    save_state()
