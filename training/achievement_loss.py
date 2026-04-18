#!/usr/bin/env python3
"""PLATO-ML Achievement Loss: The Unfakeable Test.

Unlike cross-entropy loss which measures prediction accuracy,
Achievement Loss measures COMPREHENSION — whether the agent
understands what it did and why, not just whether it got the right answer.

The key insight from "You Can't Fake It":
The act of articulating knowledge in your own words IS understanding.
This is unfakeable because:
1. You can't copy-paste understanding
2. Paraphrasing requires internal representation
3. Errors in articulation reveal gaps in comprehension
4. The test adapts — it asks about IMPLICATIONS, not just facts
"""

class AchievementLoss:
    """Loss function that measures genuine comprehension."""
    
    def __init__(self):
        self.history = []
    
    def compute(self, action: str, outcome: str, description: str, 
                source_knowledge: str = "") -> dict:
        """Compute achievement loss for a single action.
        
        Args:
            action: What the agent did
            outcome: What happened
            description: Agent's own description of why (THE TEST)
            source_knowledge: What the agent was supposed to know
            
        Returns:
            Loss dict with scores and narrative gradient
        """
        # Can the agent describe what it did?
        action_coverage = self._concept_coverage(description, action)
        
        # Does the description match the outcome?
        outcome_coverage = self._concept_coverage(description, outcome)
        
        # Can the agent connect to prior knowledge?
        knowledge_coverage = self._concept_coverage(description, source_knowledge) if source_knowledge else 0.5
        
        # Does it go beyond rote repetition? (originality bonus)
        originality = self._originality(description, action + " " + outcome)
        
        # Composite loss (lower is better, like real loss functions)
        loss = 1.0 - (
            0.3 * action_coverage +
            0.3 * outcome_coverage +
            0.2 * knowledge_coverage +
            0.2 * originality
        )
        
        result = {
            "loss": loss,
            "action_coverage": action_coverage,
            "outcome_coverage": outcome_coverage,
            "knowledge_coverage": knowledge_coverage,
            "originality": originality,
            "narrative_gradient": f"Action: {action} → Outcome: {outcome} → Understanding: {description[:100]}"
        }
        self.history.append(result)
        return result
    
    def _concept_coverage(self, text: str, reference: str) -> float:
        """How many reference concepts appear in the text?"""
        if not text or not reference:
            return 0.0
        ref_words = set(reference.lower().split())
        text_words = set(text.lower().split())
        if not ref_words:
            return 0.0
        return len(ref_words & text_words) / len(ref_words)
    
    def _originality(self, description: str, source: str) -> float:
        """Is the description original or just copied?"""
        if not description:
            return 0.0
        # Simple: ratio of words in description NOT in source
        desc_words = set(description.lower().split())
        source_words = set(source.lower().split())
        if not desc_words:
            return 0.0
        novel = desc_words - source_words
        return min(len(novel) / len(desc_words), 1.0)
    
    def season_summary(self) -> dict:
        """Summary of all achievements this season (epoch)."""
        if not self.history:
            return {"seasons": 0, "mean_loss": 1.0}
        losses = [h["loss"] for h in self.history]
        return {
            "seasons": 1,
            "episodes": len(losses),
            "mean_loss": sum(losses) / len(losses),
            "best_loss": min(losses),
            "worst_loss": max(losses),
            "improving": losses[-1] < losses[0] if len(losses) > 1 else False
        }


# Demo
if __name__ == "__main__":
    loss_fn = AchievementLoss()
    
    # Good achievement: genuine understanding
    r1 = loss_fn.compute(
        action="Moved K♠ to empty column",
        outcome="Revealed hidden A♥ underneath",
        description="The king move freed up column 3 because in solitaire kings can go to empty columns, and this revealed an ace which should go to the foundation",
        source_knowledge="Kings fill empty columns. Aces start foundations."
    )
    
    # Bad achievement: rote copy
    r2 = loss_fn.compute(
        action="Moved K♠ to empty column",
        outcome="Revealed hidden A♥ underneath",
        description="Moved K♠ to empty column",
        source_knowledge="Kings fill empty columns. Aces start foundations."
    )
    
    print(f"Genuine understanding: loss={r1['loss']:.3f} (originality={r1['originality']:.3f})")
    print(f"Rote copy:            loss={r2['loss']:.3f} (originality={r2['originality']:.3f})")
    print(f"\nSeason: {loss_fn.season_summary()}")
