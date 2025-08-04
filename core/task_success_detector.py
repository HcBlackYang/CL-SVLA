# MFM/core/task_success_detector.py

import logging
import re
import numpy as np
from typing import Tuple, Optional, List, Dict

from MFM.config import SuccessDetectionConfig
from MFM.utils.logging_utils import setup_logger

class TaskSuccessDetector:
    """
    Task Status Diagnoser (Simplified version - only detects grasp failure).
    - Retains only the grasp failure detection logic.
    - Removes placement failure detection.
    """

    def __init__(self, config: SuccessDetectionConfig):
        """
        Initializes the simplified task status detector.

        Args:
            config (SuccessDetectionConfig): The configuration for success detection.
        """
        self.config = config
        self.logger = setup_logger("TaskSuccessDetector_GraspOnly")
        
        # Parameters for grasp failure detection
        self.window_size = 10
        
        # Activation condition for grasp failure: robot needs to be close to the bowl first.
        self.bowl_activation_distance = 0.2
        
        # "Significant increase" threshold for grasp failure: distance increase above this value
        # is considered a failure.
        self.grasp_failure_increase_threshold = 0.01

        self.logger.info("Simplified Task Status Diagnoser initialized - Only detects grasp failure.")

    def _parse_report(self, state_report: str) -> Dict[str, List[float]]:
        """
        Quickly parses the list of bowl distances from the minimal state report.
        
        Args:
            state_report (str): The state summary string.

        Returns:
            Dict[str, List[float]]: A dictionary containing the list of bowl distances.
        """
        data = { "bowl_dist": [] }
        lines = state_report.strip().split('\n')
        for line in lines:
            bowl_match = re.search(r"bowl_dist=([\d\.]+|N/A)", line)
            
            if bowl_match and bowl_match.group(1) != "N/A":
                data["bowl_dist"].append(float(bowl_match.group(1)))
        
        return data

    def detect_status(self, state_report: str) -> Tuple[bool, Optional[str]]:
        """
        Detects only grasp failure based on the following rules:
        1. Activation Condition: 10 steps ago, the robot needed to be close to the bowl (distance < 0.2m).
        2. Failure Condition: The current distance is significantly greater than 10 steps ago (> 0.01m).

        Args:
            state_report (str): The state summary string to analyze.

        Returns:
            Tuple[bool, Optional[str]]: (is_success, failure_type). is_success is False if a failure is detected.
        """
        parsed_data = self._parse_report(state_report)
        
        if len(parsed_data["bowl_dist"]) < self.window_size:
            self.logger.debug(f"Data has < {self.window_size} steps, skipping detection.")
            return True, None # Not enough data, assume success for now.

        bowl_dists = parsed_data["bowl_dist"]

        # --- Sole Rule: Detect "Failed to Grasp Bowl" ---
        
        first_bowl_dist_in_window = bowl_dists[-self.window_size]
        current_bowl_dist = bowl_dists[-1]
        
        # Check activation condition: was it close to the bowl 10 steps ago?
        if first_bowl_dist_in_window < self.bowl_activation_distance:
            self.logger.debug(
                f"Grasp failure detection activated (bowl dist 10 steps ago was {first_bowl_dist_in_window:.4f}m < {self.bowl_activation_distance}m)"
            )
            
            # Check failure condition: did the distance increase significantly?
            distance_increase = current_bowl_dist - first_bowl_dist_in_window
            
            if distance_increase > self.grasp_failure_increase_threshold:
                self.logger.warning(
                    f"🚨 Grasp Failure Detected: Bowl distance significantly increased by {distance_increase:.4f}m in 10 steps "
                    f"(threshold is {self.grasp_failure_increase_threshold}m). "
                    f"Distance went from {first_bowl_dist_in_window:.4f}m to {current_bowl_dist:.4f}m"
                )
                return False, "GraspFailure"
            else:
                self.logger.debug(
                    f"Distance change is normal: {distance_increase:.4f}m (threshold {self.grasp_failure_increase_threshold}m)"
                )
        else:
            self.logger.debug(
                f"Grasp failure detection not activated (bowl dist 10 steps ago {first_bowl_dist_in_window:.4f}m >= {self.bowl_activation_distance}m)"
            )

        # No failure detected.
        return True, None