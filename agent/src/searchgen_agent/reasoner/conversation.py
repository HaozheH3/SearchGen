"""
Conversation Manager for Multi-Turn Interactions
Tracks conversation history and context with detailed stage tracking
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path


class ConversationManager:
    """Manages conversation history and context"""

    def __init__(self):
        """Initialize conversation manager"""
        self.messages: List[Dict] = []
        self.metadata: Dict = {
            "created_at": datetime.now().isoformat(),
            "total_messages": 0
        }
        # Track stage start times for duration calculation
        self._stage_start_times: Dict[str, float] = {}

    def start_stage(self, stage_name: str) -> None:
        """
        Mark the start of a stage for timing purposes

        Args:
            stage_name: Name of the stage
        """
        self._stage_start_times[stage_name] = time.time()

    def add_stage(
        self,
        stage_name: str,
        stage_input: Any,
        stage_output: Any,
        stage_metadata: Optional[Dict] = None,
        duration_seconds: Optional[float] = None
    ) -> None:
        """
        Add a complete pipeline stage with input and output

        Args:
            stage_name: Name of the stage (e.g., "prompt_analysis", "image_search")
            stage_input: Raw input to the stage
            stage_output: Raw output from the stage
            stage_metadata: Optional metadata about the stage
            duration_seconds: Optional duration in seconds (if not provided, will try to calculate from start_stage)
        """
        # Calculate duration if not provided
        if duration_seconds is None and stage_name in self._stage_start_times:
            duration_seconds = time.time() - self._stage_start_times[stage_name]
            # Clean up the start time
            del self._stage_start_times[stage_name]

        message = {
            "role": "stage",
            "stage_name": stage_name,
            "input": stage_input,
            "output": stage_output,
            "timestamp": datetime.now().isoformat(),
            "metadata": stage_metadata or {}
        }

        # Add duration if available
        if duration_seconds is not None:
            message["duration_seconds"] = round(duration_seconds, 3)

        self.messages.append(message)
        self.metadata["total_messages"] = len(self.messages)

    def get_messages(self) -> List[Dict]:
        """Get all messages"""
        return self.messages

    def save_to_file(self, output_path: Path) -> None:
        """
        Save conversation to JSON file

        Args:
            output_path: Path to save conversation
        """
        conversation_data = {
            "metadata": self.metadata,
            "stages": self.messages
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, input_path: Path) -> None:
        """
        Load conversation from JSON file

        Args:
            input_path: Path to load conversation from
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)

        self.metadata = conversation_data.get("metadata", {})
        self.messages = conversation_data.get("stages", [])

    def get_summary(self) -> Dict:
        """
        Get conversation summary

        Returns:
            Dict with summary statistics
        """
        stage_types = {}
        for msg in self.messages:
            stage_name = msg.get("stage_name", "unknown")
            stage_types[stage_name] = stage_types.get(stage_name, 0) + 1

        return {
            "total_stages": len(self.messages),
            "stage_types": stage_types,
            "created_at": self.metadata.get("created_at"),
            "duration": self._calculate_duration()
        }

    def _calculate_duration(self) -> Optional[float]:
        """Calculate conversation duration in seconds"""
        if len(self.messages) < 2:
            return None

        try:
            start_time = datetime.fromisoformat(self.messages[0]["timestamp"])
            end_time = datetime.fromisoformat(self.messages[-1]["timestamp"])
            duration = (end_time - start_time).total_seconds()
            return duration
        except Exception:
            return None


if __name__ == "__main__":
    # Test conversation manager
    conv = ConversationManager()

    conv.add_stage(
        "prompt_analysis",
        {"user_prompt": "Test prompt"},
        {"needs_search": True}
    )

    print("Conversation Summary:")
    print(json.dumps(conv.get_summary(), indent=2))

    print("\nSaving to file...")
    test_path = Path("/tmp/test_conversation.json")
    conv.save_to_file(test_path)
    print(f"Saved to {test_path}")
