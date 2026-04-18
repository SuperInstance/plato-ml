#!/usr/bin/env python3
"""PLATO-ML v3: Training loop with improving narratives via Groq API."""
import sys, json, urllib.request
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
    # Use Groq to generate a natural language narrative
    narrative = groq(
        "You are a solitaire-playing agent. Explain your move in 1-2 sentences. Be specific about your strategy weights.",
        f"My weights are: foundation={w['foundation_weight']:.2f}, reveal={w['reveal_weight']:.2f}, king={WEIGHTS['king_weight']:.2f}. I'm going to {action}. Knowledge: {state.data.get('knowledge','general solitaire')}. Why?"
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
connections = {"entrance": ["strategy_room"], "strategy_room": ["execution"], "execution": ["after_action"]}

episodes = [
    {"name": "foundation", "input": {"deck": "standard", "columns": 7, "knowledge": "Aces start foundations. Build up by suit."}},
    {"name": "reveal", "input": {"deck": "standard", "columns": 7, "knowledge": "Face-down cards must be revealed before playing."}},
    {"name": "king_empty", "input": {"deck": "standard", "columns": 7, "knowledge": "Only kings can fill empty columns."}},
]

print("PLATO-ML v3: LLM-Powered Narrative Gradient Descent")
print("="*60)
for season in range(3):
    print(f"\nSeason {season+1} | Weights: {json.dumps({k:round(v,2) for k,v in WEIGHTS.items()})}")
    for ep in episodes:
        current = RoomState(room_id="input", data=ep["input"])
        for rid in ["entrance", "strategy_room", "execution", "after_action"]:
            if rid in rooms:
                current = rooms[rid].forward(current)
        print(f"  [{ep['name']}] loss={current.data.get('loss',0):.3f} orig={current.data.get('originality',0):.2f}")

print(f"\n{'='*60}")
print(f"FINAL: {len(history)} episodes")
for i, h in enumerate(history):
    print(f"  Ep{i}: loss={h['loss']:.3f} orig={h['originality']:.2f} | {h['narrative'][:60]}...")
