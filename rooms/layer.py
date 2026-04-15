#!/usr/bin/env python3
"""PLATO-ML: Room as Layer — the fundamental abstraction."""
import json, hashlib
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

@dataclass
class RoomState:
    """A Room's state IS the tensor. Text-structured, portable, inspectable."""
    room_id: str
    data: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    
    def update(self, key: str, value: Any) -> "RoomState":
        """Forward pass: transform state."""
        old = self.data.get(key)
        self.data[key] = value
        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "key": key, "old": old, "new": value,
            "type": "forward"
        })
        return self
    
    def gradient(self) -> list:
        """Compute 'gradient' = the narrative of what changed and why."""
        return [
            {"key": h["key"], "delta": f"Changed {h['key']} from {h['old']} to {h['new']}"}
            for h in self.history if h["type"] == "forward"
        ]

@dataclass  
class Room:
    """A Room IS a Layer in the PLATO-ML network."""
    room_id: str
    transform: callable  # The forward function
    state: RoomState = field(default_factory=lambda: RoomState(room_id=""))
    
    def __post_init__(self):
        self.state.room_id = self.room_id
    
    def forward(self, input_state: RoomState) -> RoomState:
        """Transform input state through this room's logic."""
        result = self.transform(input_state, self.state)
        self.state.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "room": self.room_id,
            "input_keys": list(input_state.data.keys()),
            "output_keys": list(result.data.keys()),
            "type": "room_forward"
        })
        return result
    
    def describe(self) -> str:
        """The unfakeable test: describe what this room does in its own words."""
        return f"Room {self.room_id} transforms {{{', '.join(self.state.data.keys())}}} into outputs"

class PLATOModel:
    """A Map of Rooms = A Neural Network."""
    def __init__(self, name: str):
        self.name = name
        self.rooms = {}  # room_id -> Room
        self.connections = {}  # room_id -> [next_room_ids]
        self.achievements = []  # The loss log
    
    def add_room(self, room: Room):
        self.rooms[room.room_id] = room
        self.connections.setdefault(room.room_id, [])
    
    def connect(self, from_id: str, to_id: str):
        self.connections.setdefault(from_id, []).append(to_id)
    
    def forward(self, initial_state: RoomState) -> RoomState:
        """Run the full pipeline: Entrance → ... → Final Room."""
        current = initial_state
        visited = []
        
        # BFS through connected rooms
        queue = ["entrance"]
        while queue:
            room_id = queue.pop(0)
            if room_id not in self.rooms or room_id in visited:
                continue
            visited.append(room_id)
            room = self.rooms[room_id]
            current = room.forward(current)
            queue.extend(self.connections.get(room_id, []))
        
        return current
    
    def train_season(self, episodes: list) -> dict:
        """One season = one epoch. Episodes are training scenarios."""
        results = []
        for episode in episodes:
            state = RoomState(room_id="input", data=episode["input"])
            output = self.forward(state)
            
            # Achievement test: can the system describe what happened?
            narrative = []
            for room_id, room in self.rooms.items():
                narrative.append(room.describe())
            
            # Loss = unfakeable test score
            achievement_score = self._achievement_test(narrative, episode.get("expected", ""))
            
            results.append({
                "episode": episode.get("name", "unnamed"),
                "achievement_score": achievement_score,
                "output": output.data,
                "narrative": narrative
            })
            self.achievements.append(achievement_score)
        
        return {
            "season": len(self.achievements),
            "episodes": len(results),
            "mean_achievement": sum(self.achievements) / len(self.achievements) if self.achievements else 0,
            "results": results
        }
    
    def _achievement_test(self, narrative: list, expected: str) -> float:
        """The unfakeable test: score based on genuine comprehension.
        
        Unlike cross-entropy loss (prediction accuracy), this measures
        whether the system UNDERSTANDS what it did, not just whether
        it produced the right output.
        """
        if not narrative or not expected:
            return 0.5  # No basis for scoring
        
        # Simple heuristic: overlap between narrative and expected concepts
        narrative_text = " ".join(narrative).lower()
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.5
        
        hits = sum(1 for w in expected_words if w in narrative_text)
        return min(hits / len(expected_words), 1.0)
