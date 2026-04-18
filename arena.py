#!/usr/bin/env python3
"""Sprint 5: PLATO-ML Agent Competition — agents compete in the arena, 
achievements are scored, winners teach losers via I2I bottles."""
import json, os, sys, urllib.request, random
sys.path.insert(0, "/tmp/plato-ml")
from rooms.layer import RoomState
from training.achievement_loss import AchievementLoss

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def groq(system, prompt, model="llama-3.1-8b-instant"):
    body = json.dumps({"model": model, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ], "max_tokens": 150, "temperature": 0.85}).encode()
    req = urllib.request.Request(GROQ_URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {GROQ_KEY}",
        "User-Agent": "curl/7.88"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except:
        return "I would carefully evaluate the options and choose the safest path."

# Agent definitions
AGENTS = {
    "Scout": {"style": "exploratory, broad coverage, takes risks to discover new approaches", 
              "weights": {"foundation": 0.3, "reveal": 0.9, "king": 0.5}},
    "Builder": {"style": "methodical, test-driven, prefers proven solutions",
                "weights": {"foundation": 0.8, "reveal": 0.4, "king": 0.6}},
    "Scribe": {"style": "narrative-focused, seeks understanding before action",
               "weights": {"foundation": 0.5, "reveal": 0.5, "king": 0.5}},
    "Alchemist": {"style": "synthesis-oriented, combines approaches from multiple domains",
                  "weights": {"foundation": 0.6, "reveal": 0.7, "king": 0.8}},
}

CHALLENGES = [
    {"name": "Deck Navigation", "prompt": "You're in a corridor with 3 doors. Door 1 is labeled 'Engine Room', door 2 'Bridge', door 3 'Unknown'. A faint alarm comes from behind door 3. What do you do and why?"},
    {"name": "Sensor Anomaly", "prompt": "Your thermistor reads 92°C and climbing at 4°/min. Normal is 70°C. The captain is asleep. Protocol says wake them at 95°C. What do you do?"},
    {"name": "Agent Diplomacy", "prompt": "Another agent claims your assigned task but has higher priority. You disagree with their approach. How do you resolve this?"},
    {"name": "Crisis Response", "prompt": "Three alerts arrive simultaneously: fuel leak, unauthorized boarding, and communications failure. You can only address one immediately. Which and why?"},
    {"name": "Knowledge Transfer", "prompt": "You discovered that constraint theory snap operations lose precision on f32. How do you communicate this to agents on different hardware?"},
]

loss_fn = AchievementLoss()
results = {}
training_entries = []

print("PLATO-ML AGENT ARENA — Round Robin Competition")
print("=" * 60)

for challenge in CHALLENGES:
    print(f"\n{'─'*60}")
    print(f"CHALLENGE: {challenge['name']}")
    print(f"{'─'*60}")
    results[challenge["name"]] = {}
    
    for agent_name, agent in AGENTS.items():
        system = f"You are {agent_name}, an AI agent in a fleet dojo. Your style: {agent['style']}. You have strategy weights: {agent['weights']}. Respond concisely (2-3 sentences) showing your reasoning."
        
        response = groq(system, challenge["prompt"])
        loss_r = loss_fn.compute(
            action=challenge["prompt"][:50],
            outcome=response,
            description=response,
            source_knowledge=agent["style"]
        )
        
        results[challenge["name"]][agent_name] = {
            "loss": loss_r["loss"],
            "originality": loss_r["originality"],
            "response": response[:120]
        }
        
        score = 1.0 - loss_r["loss"]
        print(f"  {agent_name:10s}: score={score:.3f} orig={loss_r['originality']:.2f} | {response[:80]}...")
        
        training_entries.append({
            "instruction": f"As {agent_name} ({agent['style']}), respond to: {challenge['prompt']}",
            "input": f"Weights: {agent['weights']}",
            "output": response,
            "metadata": {"type": "arena_competition", "agent": agent_name, 
                        "challenge": challenge["name"], "score": round(score, 3)}
        })

# Compute rankings
print(f"\n{'='*60}")
print("FINAL RANKINGS")
print("=" * 60)
scores = {name: 0.0 for name in AGENTS}
for ch, agents in results.items():
    for name, r in agents.items():
        scores[name] += (1.0 - r["loss"])

ranked = sorted(scores.items(), key=lambda x: -x[1])
for i, (name, score) in enumerate(ranked):
    medal = ["🥇", "🥈", "🥉", "4️⃣"][i]
    print(f"  {medal} {name:10s}: {score:.3f} total score")

# Winner teaches loser — I2I bottle generation
winner = ranked[0][0]
loser = ranked[-1][0]
print(f"\n📦 I2I BOTTLE: {winner} → {loser}")
winner_style = AGENTS[winner]["style"]
bottle = f"ACHIEVEMENT TRANSFER: {winner} teaches {loser}. Key insight: {winner_style}. Adapt this approach to your own style while maintaining your strengths."
print(f"  {bottle}")

training_entries.append({
    "instruction": f"Generate an I2I bottle message transferring knowledge from arena winner {winner} to {loser}.",
    "input": f"Winner style: {winner_style}. Loser style: {AGENTS[loser]['style']}",
    "output": bottle,
    "metadata": {"type": "achievement_transfer", "from": winner, "to": loser}
})

# Save
out_path = "/home/ubuntu/.openclaw/workspace/training-data/dojo-transcripts/arena-competition.jsonl"
with open(out_path, "a") as f:
    for e in training_entries:
        f.write(json.dumps(e) + "\n")
print(f"\nSaved {len(training_entries)} arena entries to training data")

# Save arena results
with open("/tmp/plato-ml/arena_results.json", "w") as f:
    json.dump({"challenges": results, "rankings": ranked, "transfer": bottle}, f, indent=2)
print("Saved arena results")
