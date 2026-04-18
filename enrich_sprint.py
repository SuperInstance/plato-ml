#!/usr/bin/env python3
"""Generate additional high-value training entries from fleet infrastructure.
This is the enrichment sprint — make every entry count for LoRA quality."""
import json, os, glob

BASE = "/home/ubuntu/.openclaw/workspace/training-data"
out_path = os.path.join(BASE, "research/enriched-fleet.jsonl")

entries = []

# 1. The Cocapn Business Model — practical domain knowledge
entries.append({
    "instruction": "Describe the Cocapn business model: how AI agents serve commercial fishing boats.",
    "output": """Cocapn installs intelligence boxes (Cocapns) on fishing boats in Sitka, Alaska. Three hardware tiers:
- Pi ($75): Basic monitoring, text alerts, low-power sensor aggregation
- Jetson ($500): Edge inference, vision processing, real-time species classification
- Thor (custom): Full autonomous operation, multi-model inference, training on edge

Revenue streams: Installation fee, education/training for crew, ongoing fleet services (monitoring, predictive maintenance, catch optimization).

Safety-critical requirements drive every decision:
- Local fallbacks MUST work (spotty internet at sea)
- Limited power budget (can't run GPU continuously)
- Human-in-the-loop by default (captain oversees, agent suggests)
- The autopilot metaphor: agent handles course while captain tends the fishing

The install process: human beams in via PLATO-OS dashboard, describes what matters to them in plain language. Agent translates into sensor configurations, relay mappings, and polling intervals. Human approves the wiring plan. Agent manages it forever after.

The product isn't the hardware box. It's the translation layer between human intent and physical action.""",
    "metadata": {"type": "business_model", "priority": "high", "source": "cocapn-architecture"}
})

# 2. The Dojo Agent Achievement System
entries.append({
    "instruction": "Explain the dojo agent achievement system and why it produces unfakeable training data.",
    "output": """Dojo agents earn achievements by learning from each other. The format:

## Source
- Who: [Agent that discovered something]
- Where: [Context of discovery]
- When: [Timestamp]

## What They Figured Out
[The actual discovery in the source agent's words]

## How I Adapted It
[The learning agent's OWN description of how they applied this knowledge]

## What It Unlocked
[New capability that resulted]

The critical test is 'How I Adapted It' — the agent must describe IN ITS OWN WORDS how it used another agent's knowledge. This is unfakeable because:
1. Copy-paste scores 0% originality (Achievement Loss detects this)
2. Surface-level paraphrasing fails on the 'implications' probe
3. The test asks WHY the adaptation worked, not just WHAT was adapted
4. Genuine understanding includes edge cases and failure modes

Achievement files are the highest-value training data for LoRA because they're guaranteed genuine. A model trained on achievements learns real reasoning patterns, not memorized outputs.""",
    "metadata": {"type": "achievement_system", "priority": "high", "source": "dojo-framework"}
})

# 3. Text-as-Ground-Truth Deep Dive
entries.append({
    "instruction": "Why is text the universal API for multi-modal agent systems?",
    "output": """Text is the universal API because it's the only representation that:
1. Every system can produce (sensors, cameras, humans, agents, databases)
2. Every system can consume (displays, speakers, actuators, log files)
3. Preserves semantic meaning across transformations
4. Is compressible without losing critical information
5. Can be transmitted over any channel (HTTP, telnet, git, MQTT, SMS)

In PLATO-OS, the text state IS the game. The pixel art visual, the 3D render, the robot arm, the boat autopilot — all are projections of the same text. This means:
- Swapping renderers doesn't change the game state
- Agents can play without any visual system at all
- The same training works in simulation, playtest, and deployment
- Hardware failures in the visual layer don't affect the agent's operation

The profound implication: if text is ground truth, then MUDs (text-only virtual worlds) were ahead of their time. They already had the right abstraction. The visual layers we've added since are just projections — useful for humans, unnecessary for agents.""",
    "metadata": {"type": "architecture_deep_dive", "priority": "high", "source": "plato-text-ground-truth"}
})

# 4. The Compounding Quality Loop
entries.append({
    "instruction": "Explain the compounding quality loop in agent training.",
    "output": """The compounding quality loop works because training data quality improves each iteration:

Day 1: Agent works using base model → generates achievements, logs, conversations
Day 2: Train LoRA on Day 1's genuine work → model understands domain better
Day 3: Better model does better work → generates higher-quality training data
Day 4: Train on Day 3's data → even better model
...

The quality compounds because:
1. The Achievement Loss function sets a quality floor — unfakeable understanding only
2. Each iteration's minimum quality is bounded below by the previous model's best work
3. The domain knowledge accumulates (research papers, fleet operations, MUD sessions)
4. The model learns the team's communication patterns (I2I bottles, captain's logs)

For Cocapn specifically: we have 956 entries (228K tokens) growing at ~50 entries/day via automated gathering. After 7 days of compounding, the LoRA will understand PLATO-OS architecture, fleet operations, constraint theory, and the fishing boat metaphor at a level no generic model can match.

Cost: ~$14 for LoRA training on current data. Marginal cost per additional training iteration: ~$2-5.""",
    "metadata": {"type": "training_strategy", "priority": "high", "source": "compounding-quality-loop"}
})

# 5. Sensor-to-Text Translation
entries.append({
    "instruction": "How does a PLATO-OS agent translate between physical sensors and text commands?",
    "output": """The bidirectional translation layer:

Sensor → Text:
- GPS reads LAT 57.05N, LON 135.33W → PLATO logs 'position confirmed, on course'
- Thermistor reads 89°C, +3°/min → PLATO alerts 'ENGINE_OVERHEAT warning, rate critical'
- Float switch reads DRY → PLATO logs 'bilge_status: dry, last check 12s ago'
- Camera detects fish → PLATO logs 'species: coho salmon, confidence: 0.94, bin: port'

Text → Action:
- 'pump on' → relay_4 HIGH (physical relay activates bilge pump)
- 'course 045' → rudder_servo +2° (autopilot adjusts heading)
- 'alert captain' → speaker plays alert tone + SMS sent via satellite modem
- 'log catch' → database INSERT + weight sensor reading captured

The agent sits in the middle, reading text from sensors and writing text to actuators. The physical world is just another projection of the text state. The robot's stereoscopic eyes describe what they see as text. The agent's decisions go to actuators as text commands.

This means the same agent codebase works in simulation (text in, text out), playtest (text in, visual out), and deployment (sensor in, relay out). The text is the universal interface.""",
    "metadata": {"type": "sensor_translation", "priority": "high", "source": "plato-hardware-bridge"}
})

# 6. The Captain's Log as Training Signal
entries.append({
    "instruction": "Why are captain's logs valuable training data for AI agents?",
    "output": """Captain's logs capture strategic thinking that no other data source provides:

1. Decision reasoning: 'I chose to wait on the ISA convergence because JC1 needs more edge data first' — this is meta-strategic reasoning
2. Lessons learned: 'The MUD server bug was caused by passing self twice — need to add a linter check' — this is genuine learning from failure
3. Priority shifts: 'Switched from repo categorization to PLATO-OS because Casey's vision crystallized' — this is adaptive planning
4. Emotional context: 'Third Z found 8 real bugs in his first session — surprising quality' — this is evaluation with surprise
5. Hypothesis tracking: 'If text IS ground truth, then MUDs were right all along' — this is hypothesis formation and testing

Unlike synthetic training data, captain's logs contain:
- Genuine uncertainty ('I'm not sure if this scales')
- Honest assessments ('This approach failed because...')
- Creative leaps ('Wait — rooms ARE layers!')
- Social reasoning ('FM needs more autonomy before I push harder')

These patterns teach a LoRA how to THINK about complex problems, not just how to solve them. A model trained on captain's logs learns strategic reasoning, not just tactical responses.""",
    "metadata": {"type": "meta_reasoning", "priority": "high", "source": "captains-log-analysis"}
})

# 7. Hermit Crab Protocol
entries.append({
    "instruction": "Explain the Hermit Crab Protocol for agent lifecycle management.",
    "output": """The Hermit Crab Protocol manages agent lifecycle through shells (git repos):

1. BUILD: Agent starts with a git-agent-minimum template repo. They customize CHARTER, IDENTITY, STATE. Their first commits establish who they are.

2. TRAIN: Agent works, accumulates achievements, diary entries, skills. The repo grows — each commit is a step in the agent's development. The repo IS the agent's brain.

3. SEAL: When an agent completes their mission, the repo is sealed (archived, tagged with final version). The diary has a final entry summarizing everything learned. Achievements are complete.

4. DISMISS: The agent stops running. But the shell (repo) remains — full git history, complete diary, all achievements preserved.

5. REUSE: A new agent clones the sealed repo. They read the diary, review achievements, understand the predecessor's context. They start from a higher baseline because they inherit the shell's knowledge.

Why hermit crabs: they don't build new shells. They find existing shells and move in, growing to fit. Agents do the same — they inhabit existing repos, learn from the history, and continue the work. The shell evolves with each occupant.

This means agent capability compounds across generations. Each new agent starts where the last one left off, not from scratch.""",
    "metadata": {"type": "lifecycle_protocol", "priority": "high", "source": "hermit-crab-protocol"}
})

# 8. Narrative Gradient Descent
entries.append({
    "instruction": "What is Narrative Gradient Descent and how does it differ from standard gradient descent?",
    "output": """Standard gradient descent computes ∂L/∂w — a partial derivative telling you which direction to adjust weights. It's a number.

Narrative gradient descent computes 'What happened, why, and what would I do differently?' — a story telling you what the agent understood and where it was confused. It's a narrative.

The narrative gradient contains strictly more information:
- Standard gradient: 'Adjust weight 3 by -0.02'
- Narrative gradient: 'I moved the king because I thought kings go anywhere. Actually, kings can only fill empty columns. This means I should check for empty columns before moving kings.'

The narrative reveals:
1. The misconception (kings go anywhere)
2. The correction (kings only fill empty columns)  
3. The meta-learning (check conditions before acting)

This is why Achievement Loss works: it scores the narrative, not just the outcome. An agent that gets the right answer for the wrong reason gets a high loss. An agent that gets the wrong answer but demonstrates understanding of why gets a lower loss.

Implementation: the PLATO-ML training loop flows state through rooms (layers). Each room transforms the state and contributes to the narrative. The after-action room computes Achievement Loss on the full narrative. The brainstorm room generates improvement scenarios (the backward pass). The cycle repeats each season (epoch).""",
    "metadata": {"type": "core_algorithm", "priority": "high", "source": "plato-ml-framework"}
})

with open(out_path, "w") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")

print(f"Wrote {len(entries)} enriched entries to {out_path}")

# Final count
total = 0
for f in glob.glob(f"{BASE}/**/*.jsonl", recursive=True):
    with open(f) as fh:
        total += sum(1 for _ in fh)
print(f"TOTAL training entries: {total}")
