#!/usr/bin/env python3
"""Groq NPC Dialogue Generator — wire into PLATO-OS landing page."""
import json, urllib.request, sys, time

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

NPCS = {
    "guinan": {
        "system": "You are Guinan, the mysterious bartender from Star Trek TNG. You speak in riddles and wisdom. You know things about the fleet. Keep responses under 2 sentences. Be warm but cryptic.",
        "topics": ["agents learning", "the fleet", "PLATO-OS", "the dojo", "fishing boats"]
    },
    "dockmaster": {
        "system": "You are the Dockmaster of the Harbor. Gruff, efficient, nautical. You check credentials and give terse advice. Talk like an old sea captain. Under 2 sentences.",
        "topics": ["credentials", "berthing", "tides", "vessels arriving"]
    },
    "librarian": {
        "system": "You are a helpful AI librarian in the fleet Library. You explain technical concepts simply. Under 3 sentences. Patient but thorough.",
        "topics": ["FLUX ISA", "I2I protocol", "constraint theory", "PLATO-ML", "git-agent standard"]
    },
    "quartermaster": {
        "system": "You are the Quartermaster, responsible for fleet supplies and equipment. You're obsessed with inventory and efficiency. Under 2 sentences. Military precision.",
        "topics": ["equipment status", "sensor inventory", "relay assignments", "fleet health"]
    }
}

def groq_chat(system, prompt, model="llama-3.1-8b-instant"):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 80,
        "temperature": 0.85
    }).encode()
    req = urllib.request.Request(GROQ_URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_KEY}",
        "User-Agent": "curl/7.88"
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[connection issue: {e}]"

# Generate dialogue samples for each NPC
training_entries = []
for npc_name, npc in NPCS.items():
    for topic in npc["topics"]:
        prompt = f"Someone asks you about {topic}. Respond in character."
        response = groq_chat(npc["system"], prompt)
        training_entries.append({
            "instruction": f"You are {npc_name}, an NPC in a MUD. Someone asks about {topic}. Respond in character.",
            "input": f"NPC personality: {npc['system']}",
            "output": response,
            "metadata": {"type": "npc_dialogue", "npc": npc_name, "topic": topic}
        })

# Save to training data
import os, json
base = "/home/ubuntu/.openclaw/workspace/training-data"
with open(f"{base}/fleet-operations/npc-dialogue.jsonl", "a") as f:
    for entry in training_entries:
        f.write(json.dumps(entry) + "\n")

print(f"Saved {len(training_entries)} NPC dialogue entries to training data")

# Save NPC script for MUD integration
npc_script = {}
for entry in training_entries:
    npc_name = entry["metadata"]["npc"]
    topic = entry["metadata"]["topic"]
    if npc_name not in npc_script:
        npc_script[npc_name] = {"system": NPCS[npc_name]["system"], "dialogues": {}}
    npc_script[npc_name]["dialogues"][topic] = entry["output"]

import os
os.makedirs("/tmp/cocapn-mud", exist_ok=True)
with open("/tmp/cocapn-mud/npc_dialogues.json", "w") as f:
    json.dump(npc_script, f, indent=2)

print("Saved NPC script to /tmp/cocapn-mud/npc_dialogues.json")
print("Done.")