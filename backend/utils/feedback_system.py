"""
Feedback system for monitoring model performance in production
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List

class FeedbackSystem:
    """System to collect and monitor feedback on model predictions"""
    
    def __init__(self, feedback_file: str = "backend/data/feedback.json"):
        self.feedback_file = feedback_file
        self.ensure_feedback_file()
    
    def ensure_feedback_file(self):
        """Create feedback file if it doesn't exist"""
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                json.dump([], f)
    
    def add_feedback(self, text: str, prediction: str, confidence: float, 
                    user_feedback: str = None, is_correct: bool = None):
        """Add feedback entry"""
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "prediction": prediction,
            "confidence": confidence,
            "user_feedback": user_feedback,
            "is_correct": is_correct
        }
        
        # Load existing feedback
        with open(self.feedback_file, 'r') as f:
            feedback_data = json.load(f)
        
        # Add new entry
        feedback_data.append(feedback_entry)
        
        # Save updated feedback
        with open(self.feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        with open(self.feedback_file, 'r') as f:
            feedback_data = json.load(f)
        
        if not feedback_data:
            return {"total_feedback": 0}
        
        total_feedback = len(feedback_data)
        correct_predictions = sum(1 for entry in feedback_data 
                                if entry.get('is_correct') == True)
        incorrect_predictions = sum(1 for entry in feedback_data 
                                  if entry.get('is_correct') == False)
        
        accuracy = (correct_predictions / total_feedback * 100) if total_feedback > 0 else 0
        
        return {
            "total_feedback": total_feedback,
            "correct_predictions": correct_predictions,
            "incorrect_predictions": incorrect_predictions,
            "accuracy": accuracy,
            "recent_feedback": feedback_data[-5:]  # Last 5 entries
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics from feedback"""
        stats = self.get_feedback_stats()
        
        if stats["total_feedback"] == 0:
            return {"message": "No feedback available yet"}
        
        return {
            "feedback_accuracy": stats["accuracy"],
            "total_predictions": stats["total_feedback"],
            "correct_predictions": stats["correct_predictions"],
            "incorrect_predictions": stats["incorrect_predictions"],
            "recent_activity": len(stats["recent_feedback"])
        }
