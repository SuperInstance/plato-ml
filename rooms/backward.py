#!/usr/bin/env python3
"""PLATO-ML Backward Pass: Reverse Actualization and Reflection Rooms."""
import sys; sys.path.insert(0, "/tmp/plato-ml")
from rooms.layer import RoomState, Room
from training.achievement_loss import AchievementLoss

class BrainstormRoom(Room):
    """Reverse Actualization: outcome → inputs. IS backpropagation."""
    def __init__(self):
        super().__init__("brainstorm", self._backplan)
        self.loss_fn = AchievementLoss()
    
    def _backplan(self, input_state, room_state):
        outcome = input_state.data.get("outcome", "")
        action = input_state.data.get("action_taken", "")
        description = input_state.data.get("narrative", "")
        loss_result = self.loss_fn.compute(action=action, outcome=outcome, description=description,
                                            source_knowledge=input_state.data.get("knowledge", ""))
        improvements = []
        if loss_result["originality"] < 0.5: improvements.append("Increase originality")
        if loss_result["action_coverage"] < 0.5: improvements.append("Describe actions more clearly")
        if loss_result["outcome_coverage"] < 0.5: improvements.append("Connect actions to outcomes")
        return RoomState(room_id="backplanned", data={**input_state.data, "loss": loss_result["loss"],
            "narrative_gradient": loss_result["narrative_gradient"], "improvements": improvements})

class SituationRoom(Room):
    """Pre-Planning: Generate training scenarios from improvements."""
    def __init__(self):
        super().__init__("situation", self._plan)
        self.scenarios = []
    
    def _plan(self, input_state, room_state):
        scenarios = [{"training_scenario": f"Practice: {imp}"} for imp in input_state.data.get("improvements", [])]
        self.scenarios.extend(scenarios)
        return RoomState(room_id="scenarios_ready", data={**input_state.data, "scenarios": scenarios})

class AfterActionRoom(Room):
    """Post-Mortem: Update strategy weights based on loss."""
    def __init__(self):
        super().__init__("after_action", self._reflect)
        self.season_history = []
    
    def _reflect(self, input_state, room_state):
        loss = input_state.data.get("loss", 1.0)
        weights = room_state.data.get("weights", {"foundation_weight": 0.5, "reveal_weight": 0.5, "king_weight": 0.5})
        lr = 0.1
        if loss < 0.6:
            weights["foundation_weight"] = min(1.0, weights["foundation_weight"] + lr * (1 - loss))
        else:
            weights["reveal_weight"] = min(1.0, weights["reveal_weight"] + lr * (1 - loss))
        self.season_history.append({"loss": loss, "weights": dict(weights)})
        return RoomState(room_id="reflected", data={**input_state.data, "updated_weights": weights,
            "season": len(self.season_history), "loss_trend": [h["loss"] for h in self.season_history[-5:]]})
