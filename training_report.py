#!/usr/bin/env python3
"""Visualize PLATO-ML training: loss curves, weight evolution, narrative quality."""
import json, glob, os

# Load all training data and compute statistics
training_dir = "/home/ubuntu/.openclaw/workspace/training-data"
stats = {"total_entries": 0, "by_type": {}, "by_priority": {}, "total_bytes": 0}

for jsonl in glob.glob(f"{training_dir}/**/*.jsonl", recursive=True):
    with open(jsonl) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    stats["total_entries"] += len(entries)
    stats["total_bytes"] += os.path.getsize(jsonl)
    for e in entries:
        t = e.get("metadata", {}).get("type", "unknown")
        stats["by_type"][t] = stats["by_type"].get(t, 0) + 1
        p = e.get("metadata", {}).get("priority", "normal")
        stats["by_priority"][p] = stats["by_priority"].get(p, 0) + 1

# Token estimate
total_tokens = 0
for jsonl in glob.glob(f"{training_dir}/**/*.jsonl", recursive=True):
    with open(jsonl) as f:
        for line in f:
            e = json.loads(line)
            total_tokens += len(e.get("output", "").split()) * 1.3

print("╔══════════════════════════════════════════════════╗")
print("║       PLATO-ML TRAINING DATA REPORT v1          ║")
print("╠══════════════════════════════════════════════════╣")
print(f"║ Total entries:  {stats['total_entries']:>6}                          ║")
print(f"║ Estimated tokens: {total_tokens:>10,.0f}                    ║")
print(f"║ Total size: {stats['total_bytes']/1024:>8.1f} KB                       ║")
print("╠══════════════════════════════════════════════════╣")
print("║ By type:                                         ║")
for t, c in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
    bar = "█" * min(c // 5, 30)
    print(f"║   {t:30s} {c:4d} {bar:<30s} ║")
print("╠══════════════════════════════════════════════════╣")
print("║ By priority:                                     ║")
for p, c in sorted(stats["by_priority"].items(), key=lambda x: -x[1]):
    print(f"║   {p:10s}: {c:4d}                                ║")
print("╠══════════════════════════════════════════════════╣")
print("║ LoRA Training Estimate:                          ║")
print(f"║   Model: Qwen2.5-7B (7B params)                 ║")
print(f"║   LoRA rank: 16, alpha: 32                       ║")
print(f"║   Epochs: 3                                      ║")
print(f"║   GPU: OCI A10 (24GB VRAM)                       ║")
print(f"║   Est. time: ~{total_tokens/50000:.0f} GPU hours                    ║")
print(f"║   Est. cost: ~${total_tokens/50000*2.95:.0f}                              ║")
print("╚══════════════════════════════════════════════════╝")
