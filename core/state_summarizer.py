# MFM/core/state_summarizer.py

import numpy as np
from typing import List, Dict, Any, Optional

from MFM.config import StateSummarizerConfig
from MFM.utils.logging_utils import setup_logger

class StateSummarizer:
    """
    A minimal state summarizer, customized as per user requirements.
    It only extracts key distance information required for task diagnostics.
    """

    def __init__(self, config: StateSummarizerConfig):
        """
        Initializes the minimal state summarizer.

        Args:
            config (StateSummarizerConfig): The configuration for the state summarizer.
        """
        self.config = config
        self.logger = setup_logger("StateSummarizer_Minimal")
        self.logger.info("Minimal State Summarizer initialized.")

    def _get_distance(self, state: Dict[str, Any], object_name: str) -> str:
        """
        A safe helper function to extract and calculate distance, returning a formatted string.
        
        Args:
            state (Dict[str, Any]): A single state dictionary from the collector.
            object_name (str): The name of the object to calculate distance to.

        Returns:
            str: The formatted distance string (e.g., "0.0512") or "N/A" if not found.
        """
        try:
            # The path is adapted based on the provided code structure.
            observations = state.get('observation_non_visual', {})
            rel_pos_vector = observations.get(f"{object_name}_to_robot0_eef_pos")
            
            if rel_pos_vector is not None:
                # Calculate and return the formatted distance.
                return f"{np.linalg.norm(rel_pos_vector):.4f}"
            return "N/A"
        except (KeyError, TypeError):
            return "N/A"

    def summarize_state_history(self, state_history: List[Dict[str, Any]]) -> str:
        """
        Generates a summary containing only the distances to the bowl and plate, as requested.
        
        Args:
            state_history (List[Dict[str, Any]]): The list of recent states provided by StateCollector.

        Returns:
            str: A concise, machine-readable text where each line represents a step.
                 Example:
                 "Step 81: bowl_dist=0.0512, plate_dist=0.3456
                  Step 82: bowl_dist=0.0488, plate_dist=0.3312"
        """
        if not state_history:
            return "No state history available."

        report_lines = []
        for state in state_history:
            step = state.get('step', 'N/A')
            
            # Extract distance to the bowl.
            bowl_dist_str = self._get_distance(state, "akita_black_bowl_1")
            
            # Extract distance to the plate.
            plate_dist_str = self._get_distance(state, "plate_1")

            # Generate the report line for this step.
            line = (
                f"Step {step}: "
                f"bowl_dist={bowl_dist_str}, "
                f"plate_dist={plate_dist_str}"
            )
            report_lines.append(line)

        final_report = "\n".join(report_lines)
        self.logger.debug(f"Generated minimal state report:\n{final_report}")
        return final_report