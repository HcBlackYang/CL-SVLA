# -*- coding: utf-8 -*-
# MFM/core/mfm_core.py

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from MFM.config import MFMConfig
from MFM.core.state_collector import StateCollector
from MFM.core.state_summarizer import StateSummarizer
from MFM.core.task_success_detector import TaskSuccessDetector
from MFM.core.cot_reasoning_engine import CoTReasoningEngine
from MFM.utils.logging_utils import setup_logger

@dataclass
class ExecutionContext:
    """
    Data class to hold the context of the current task execution.
    """
    task_prompt: str
    step_count: int = 0
    failure_count: int = 0
    last_feedback: Optional[str] = None
    last_state_summary: Optional[str] = None
    last_failure_type: Optional[str] = None

class MetacognitiveFeedbackModule:
    """
    Core controller for the Metacognitive Feedback Module (MFM).
    This version is refactored based on user-defined rules.
    """
    
    def __init__(self, mfm_config: MFMConfig):
        """
        Initializes the MFM core controller.
        
        Args:
            mfm_config (MFMConfig): The configuration object for the MFM.
        """
        self.mfm_config = mfm_config
        self.logger = setup_logger("MFM_Core", mfm_config.log_level)
        
        # Initialize sub-modules
        self.state_collector = StateCollector(self.mfm_config.monitoring)
        self.state_summarizer = StateSummarizer(self.mfm_config.summarizer)
        self.task_detector = TaskSuccessDetector(self.mfm_config.success_detection)
        self.cot_engine = CoTReasoningEngine(self.mfm_config.cot)
        
        # Runtime state
        self.execution_context: Optional[ExecutionContext] = None
        self.is_monitoring = False
        self.intervention_triggered = False
        
        self.logger.info("MFM Core Module (Rule-Refactored Version) initialized.")
    
    def start_monitoring(self, task_prompt: str):
        """
        Starts monitoring a new task execution.
        
        Args:
            task_prompt (str): The initial prompt for the task.
        """
        self.execution_context = ExecutionContext(task_prompt=task_prompt)
        self.is_monitoring = True
        self.state_collector.reset()
        self.intervention_triggered = False
        self.logger.info(f"Started monitoring task: {task_prompt}. Intervention flag reset.")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stops monitoring and returns a summary of the execution.
        
        Returns:
            Dict[str, Any]: A summary dictionary of the monitored task.
        """
        if not self.execution_context: return {}
        summary = {
            "task_prompt": self.execution_context.task_prompt,
            "total_steps": self.execution_context.step_count,
            "failure_count": self.execution_context.failure_count,
            "last_feedback": self.execution_context.last_feedback
        }
        self.is_monitoring = False
        self.logger.info(f"Task monitoring stopped: {summary}")
        return summary

    def process_execution_step(
        self, 
        observation: Dict[str, Any], action: np.ndarray, reward: float, 
        done: bool, info: Dict[str, Any], env: Optional[Any] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Processes a single execution step of the agent.
        (Return type bug has been fixed in this version).

        Args:
            observation (Dict[str, Any]): The observation from the environment.
            action (np.ndarray): The action taken by the agent.
            reward (float): The reward received.
            done (bool): Whether the episode has ended.
            info (Dict[str, Any]): Additional info from the environment.
            env (Optional[Any]): The environment instance, for detailed state collection.

        Returns:
            Tuple[bool, Optional[str]]: A tuple containing:
                - bool: True if an intervention is required, False otherwise.
                - Optional[str]: The corrective feedback if intervention is needed, else None.
        """
        if not self.is_monitoring or not self.execution_context:
            return False, None
        
        self.execution_context.step_count += 1
        self.state_collector.collect_step_data(observation, action, reward, done, info, env)
        
        # If an intervention has already been triggered, do nothing until reset.
        if self.intervention_triggered:
            return False, None
        
        # Analyze the state only at a specified frequency.
        if self.execution_context.step_count % self.mfm_config.monitoring.frequency == 0:
            return self._analyze_current_state()
        
        # <<< [CORE BUG FIX] >>>: Provide an explicit return value for non-analysis steps.
        return False, None
    
    def _analyze_current_state(self) -> Tuple[bool, Optional[str]]:
        """
        Analyzes the current state history to decide if intervention is needed.
        
        Returns:
            Tuple[bool, Optional[str]]: (should_intervene, feedback_string).
        """
        if not self.execution_context: 
            return False, None

        try:
            # 1. Get state history from the collector.
            history_window = self.task_detector.window_size # Use window size defined in the detector.
            state_history = self.state_collector.get_recent_states(history_window)
            
            if len(state_history) < history_window:
                self.logger.debug(f"State history has < {history_window} steps, cannot analyze.")
                return False, None

            # 2. Generate a minimal state summary.
            state_summary = self.state_summarizer.summarize_state_history(state_history)
            self.execution_context.last_state_summary = state_summary # Record the summary.
            self.logger.info(f"--- MFM Analysis (Step {self.execution_context.step_count}) ---\n{state_summary}")

            # 3. Use the summary for failure detection.
            is_success, failure_type = self.task_detector.detect_status(state_summary)

            # 4. If a failure is detected, generate feedback.
            if not is_success and failure_type:
                self.execution_context.last_failure_type = failure_type # Record the failure type.
                self.logger.warning(f"Detector Result: Failure Detected! Type: {failure_type}")
                feedback = self._generate_corrective_feedback(failure_type, state_summary)
                
                if feedback:
                    self.execution_context.failure_count += 1
                    self.execution_context.last_feedback = feedback
                    self.intervention_triggered = True
                    self.logger.warning("MFM intervention triggered!")
                    return True, feedback
            
            return False, None
        
        except Exception as e:
            self.logger.error(f"Error during state analysis: {e}", exc_info=True)
            return False, None

    def _generate_corrective_feedback(self, failure_type: str, state_summary: str) -> str:
        """
        Generates corrective feedback using the CoT engine. Now receives the state summary.
        
        Args:
            failure_type (str): The type of failure detected.
            state_summary (str): The summary of the recent state history.

        Returns:
            str: The generated corrective feedback string.
        """
        if not self.execution_context: return ""
        try:
            collected_data_for_export = self.state_collector.export_collected_data(include_detailed=True)
            latest_image = self.state_collector.get_latest_visual()

            if latest_image is None:
                self.logger.error("Could not get the latest image for CoT analysis.")
                return f"A {failure_type} was detected. Try again."

            # <<< [CORE DATA FLOW FIX] >>>: Pass the state summary to the CoT engine.
            reasoning_result, feedback = self.cot_engine.analyze_failure(
                state_collector_data=collected_data_for_export, 
                latest_image=latest_image,
                failure_type=failure_type, 
                task_prompt=self.execution_context.task_prompt,
                state_summary=state_summary # Pass the summary here!
            )
            self.logger.debug(f"CoT Reasoning Details: {reasoning_result}")
            return feedback
        except Exception as e:
            self.logger.error(f"Failed to generate feedback: {e}", exc_info=True)
            return f"An issue was detected. Try again."