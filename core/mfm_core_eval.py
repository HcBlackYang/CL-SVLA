# MFM/core/mfm_core_eval.py
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import gc
import torch
from MFM.config import MFMConfig
from MFM.core.state_collector import StateCollector
from MFM.core.state_summarizer import StateSummarizer
from MFM.core.task_success_detector import TaskSuccessDetector
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

class MetacognitiveFeedbackModuleEval:
    """
    Core controller for the MFM (modified to accept an external CoT analysis function).
    This version is designed for evaluation and uses dependency injection.
    """
    
    def __init__(self, mfm_config: MFMConfig, cot_analyzer_func: Optional[Callable] = None):
        """
        Initializes the evaluation version of the MFM core controller.

        Args:
            mfm_config (MFMConfig): The configuration object for the MFM.
            cot_analyzer_func (Optional[Callable]): An external function for CoT analysis.
                                                   This allows decoupling the CoT engine from the core MFM logic.
        """
        self.mfm_config = mfm_config
        self.logger = setup_logger("MFM_Core_Eval", mfm_config.log_level)
        
        # Initialize sub-modules
        self.state_collector = StateCollector(self.mfm_config.monitoring)
        self.state_summarizer = StateSummarizer(self.mfm_config.summarizer)
        self.task_detector = TaskSuccessDetector(self.mfm_config.success_detection)
        
        # [MODIFIED] Instead of creating a CoTReasoningEngine, it accepts a function.
        if cot_analyzer_func and callable(cot_analyzer_func):
            self.cot_analyzer = cot_analyzer_func
            self.logger.info("MFM is configured to use an externally provided CoT analysis function.")
        else:
            self.cot_analyzer = None
            self.logger.warning("No CoT analysis function provided. MFM will be unable to generate corrective feedback!")

        # Runtime state
        self.execution_context: Optional[ExecutionContext] = None
        self.is_monitoring = False
        self.intervention_triggered = False
        self.logger.info("MFM Core Module (Dependency Injection Version) initialized.")
    
    def start_monitoring(self, task_prompt: str):
        """
        Starts monitoring a new task execution.
        """
        self.execution_context = ExecutionContext(task_prompt=task_prompt)
        self.is_monitoring = True
        self.state_collector.reset()
        self.intervention_triggered = False
        self.logger.info(f"Started monitoring task: {task_prompt}. Intervention flag reset.")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stops monitoring and returns a summary of the execution.
        """
        if not self.execution_context: return {}
        summary = {
            "task_prompt": self.execution_context.task_prompt,
            "total_steps": self.execution_context.step_count,
            "failure_count": self.execution_context.failure_count
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
        """
        if not self.is_monitoring or not self.execution_context:
            return False, None
        
        self.execution_context.step_count += 1
        self.state_collector.collect_step_data(observation, action, reward, done, info, env)
        
        if self.intervention_triggered:
            return False, None
        
        if self.execution_context.step_count % self.mfm_config.monitoring.frequency == 0:
            return self._analyze_current_state()
        
        return False, None
    
    def _analyze_current_state(self) -> Tuple[bool, Optional[str]]:
        """
        Analyzes the current state history to decide if intervention is needed.
        """
        if not self.execution_context: return False, None
        try:
            history_window = self.task_detector.window_size
            state_history = self.state_collector.get_recent_states(history_window)
            if len(state_history) < history_window: return False, None

            state_summary = self.state_summarizer.summarize_state_history(state_history)
            self.execution_context.last_state_summary = state_summary
            is_success, failure_type = self.task_detector.detect_status(state_summary)

            if not is_success and failure_type:
                self.execution_context.last_failure_type = failure_type
                
                feedback = self._generate_corrective_feedback(failure_type)
                
                if feedback:
                    self.execution_context.failure_count += 1
                    self.execution_context.last_feedback = feedback
                    self.intervention_triggered = True
                    return True, feedback
            
            return False, None
        except Exception as e:
            self.logger.error(f"Error during state analysis: {e}", exc_info=True)
            return False, None

    def _generate_corrective_feedback(self, failure_type: str) -> str:
        """
        [MODIFIED] Generates corrective feedback by calling the external analyzer function.
        
        Args:
            failure_type (str): The type of failure detected.

        Returns:
            str: The generated corrective feedback.
        """
        if not self.execution_context or not self.cot_analyzer: return ""
        try:
            # <<< [CORE FIX] >>>
            # Clear GPU cache right here, before calling the heavyweight CoT analysis!
            # Although we can't delete variables from the main loop, gc and empty_cache are global.
            self.logger.info("Clearing CUDA cache before CoT analysis (inside MFM)...")
            gc.collect()
            torch.cuda.empty_cache()

            # Now, we should have more VRAM available to run the CoT analysis.
            feedback = self.cot_analyzer(
                state_collector=self.state_collector,
                task_prompt=self.execution_context.task_prompt,
                failure_type=failure_type
            )
            return feedback
        except Exception as e:
            # Modify exception handling here to make OOM errors more apparent.
            if "out of memory" in str(e).lower():
                self.logger.critical(f"FATAL: CUDA out of memory during CoT analysis, even after clearing cache. This indicates a fundamental memory issue.", exc_info=True)
                # We can choose to re-raise the exception to stop the program.
                # raise e
                # Or return a more explicit error message.
                return "FATAL: CUDA out of memory during analysis."
            
            self.logger.error(f"Error calling external CoT analysis function: {e}", exc_info=True)
            return f"An issue was detected. Try again."