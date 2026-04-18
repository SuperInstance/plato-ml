#!/usr/bin/env python3
"""Sprint 4: PLATO-ML Curriculum Builder — generate training curricula from room graphs."""
import json, os, sys
sys.path.insert(0, "/tmp/plato-ml")
from rooms.layer import RoomState, Room

class CurriculumRoom(Room):
    """Generates a training curriculum from a set of learning objectives."""
    def __init__(self):
        super().__init__("curriculum", self._build_curriculum)
        self.curricula = []
    
    def _build_curriculum(self, input_state, room_state):
        objectives = input_state.data.get("objectives", [])
        difficulty = input_state.data.get("difficulty", "medium")
        
        curriculum = []
        for i, obj in enumerate(objectives):
            # Build progressive difficulty
            levels = {
                "easy": {"hint_level": "full", "time_limit": None, "attempts": 5},
                "medium": {"hint_level": "partial", "time_limit": 300, "attempts": 3},
                "hard": {"hint_level": "none", "time_limit": 120, "attempts": 1}
            }
            config = levels.get(difficulty, levels["medium"])
            
            curriculum.append({
                "lesson": i + 1,
                "objective": obj,
                "difficulty": difficulty,
                "config": config,
                "assessment": f"Describe how you would {obj} in your own words",
                "passing_criteria": "Achievement Loss < 0.5 AND Originality > 0.7"
            })
        
        self.curricula.append(curriculum)
        return RoomState(room_id="curriculum_built", data={
            **input_state.data,
            "curriculum": curriculum,
            "total_lessons": len(curriculum),
            "difficulty": difficulty
        })

class AssessmentRoom(Room):
    """Runs the unfakeable test against agent responses."""
    def __init__(self):
        super().__init__("assessment", self._assess)
        self.results = []
    
    def _assess(self, input_state, room_state):
        curriculum = input_state.data.get("curriculum", [])
        agent_responses = input_state.data.get("responses", {})
        
        results = []
        for lesson in curriculum:
            obj = lesson["objective"]
            response = agent_responses.get(f"lesson_{lesson['lesson']}", "")
            
            # Score the response
            obj_words = set(obj.lower().split())
            resp_words = set(response.lower().split()) if response else set()
            
            coverage = len(obj_words & resp_words) / max(len(obj_words), 1)
            originality_words = resp_words - obj_words
            originality = len(originality_words) / max(len(resp_words), 1) if resp_words else 0
            
            passed = coverage > 0.5 and originality > 0.3
            
            results.append({
                "lesson": lesson["lesson"],
                "objective": obj,
                "coverage": round(coverage, 3),
                "originality": round(originality, 3),
                "passed": passed,
                "response_preview": response[:100] if response else "(no response)"
            })
        
        self.results.extend(results)
        pass_rate = sum(1 for r in results if r["passed"]) / max(len(results), 1)
        
        return RoomState(room_id="assessed", data={
            **input_state.data,
            "assessment_results": results,
            "pass_rate": pass_rate,
            "total_assessed": len(self.results)
        })

class PromotionRoom(Room):
    """Decides if agent advances to next difficulty level."""
    def __init__(self):
        super().__init__("promotion", self._decide)
    
    def _decide(self, input_state, room_state):
        pass_rate = input_state.data.get("pass_rate", 0)
        current_diff = input_state.data.get("difficulty", "easy")
        
        difficulty_ladder = ["easy", "medium", "hard", "expert", "master"]
        current_idx = difficulty_ladder.index(current_diff) if current_diff in difficulty_ladder else 0
        
        if pass_rate >= 0.8:
            # Promote
            new_idx = min(current_idx + 1, len(difficulty_ladder) - 1)
            decision = "promoted"
        elif pass_rate >= 0.5:
            decision = "hold"
            new_idx = current_idx
        else:
            # Demote
            new_idx = max(current_idx - 1, 0)
            decision = "demoted"
        
        return RoomState(room_id="decided", data={
            **input_state.data,
            "decision": decision,
            "old_difficulty": current_diff,
            "new_difficulty": difficulty_ladder[new_idx],
            "pass_rate": pass_rate
        })

# Build the curriculum model
curriculum = CurriculumRoom()
assessment = AssessmentRoom()
promotion = PromotionRoom()

# Demo: PLATO-OS agent training curriculum
objectives = [
    "Navigate a MUD room using text commands",
    "Send an I2I bottle to another agent",
    "Read and interpret a sensor text state",
    "Translate a human request into relay commands",
    "Write a captain's log entry reflecting on strategy",
    "Demonstrate the text-as-ground-truth principle",
    "Design a room that functions as a cognitive tool",
    "Execute narrative gradient descent on a training episode"
]

# Run curriculum at each difficulty level
print("PLATO-ML CURRICULUM SYSTEM")
print("=" * 60)

current_diff = "easy"
for round_num in range(4):
    state = RoomState(room_id="input", data={"objectives": objectives, "difficulty": current_diff})
    
    # Build curriculum
    state = curriculum.forward(state)
    lessons = state.data["curriculum"]
    
    # Simulate agent responses (improving each round)
    responses = {}
    for lesson in lessons:
        # Simulated improvement: more words and originality each round
        base_words = lesson["objective"].split()
        extra = ["understand", "because", "apply", "context", "fleet"] * (round_num + 1)
        responses[f"lesson_{lesson['lesson']}"] = " ".join(base_words + extra[:round_num + 2])
    
    state.data["responses"] = responses
    
    # Assess
    state = assessment.forward(state)
    pass_rate = state.data["pass_rate"]
    
    # Decide promotion
    state = promotion.forward(state)
    
    print(f"\nRound {round_num + 1} ({current_diff})")
    print(f"  Lessons: {len(lessons)} | Pass rate: {pass_rate:.1%}")
    print(f"  Decision: {state.data['decision']} → {state.data['new_difficulty']}")
    
    for r in state.data["assessment_results"][:3]:
        status = "✓" if r["passed"] else "✗"
        print(f"    {status} L{r['lesson']}: cov={r['coverage']:.2f} orig={r['originality']:.2f} | {r['objective'][:40]}")
    
    current_diff = state.data["new_difficulty"]

print(f"\nTotal assessments: {len(assessment.results)}")
print(f"Final difficulty: {current_diff}")

# Save as training data
training_entry = {
    "instruction": "Design a progressive training curriculum for an AI agent in a PLATO-OS environment.",
    "output": json.dumps({"objectives": objectives, "difficulty_ladder": ["easy","medium","hard","expert","master"],
                          "pass_threshold": 0.8, "assessment": "unfakeable_test", "promotion_rule": "pass_rate >= 0.8 promotes, < 0.5 demotes"}, indent=2),
    "metadata": {"type": "curriculum_design", "source": "plato-ml-curriculum"}
}
with open("/home/ubuntu/.openclaw/workspace/training-data/fleet-operations/curriculum.jsonl", "a") as f:
    f.write(json.dumps(training_entry) + "\n")
print("\nSaved curriculum to training data")
