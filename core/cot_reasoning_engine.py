# -*- coding: utf-8 -*-
# MFM/core/cot_reasoning_engine.py

import json
import logging
from typing import Dict, Any, Tuple
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import numpy as np

from MFM.config.mfm_config import CoTConfig

class CoTReasoningEngine:
    """
    Uses Qwen2.5-VL for Chain-of-Thought (CoT) failure analysis and directly generates recovery instructions.
    """
    
    def __init__(self, config: CoTConfig):
        """
        Initializes the CoT Reasoning Engine.

        Args:
            config (CoTConfig): The configuration for the CoT engine.
        """
        self.config = config
        self.logger = logging.getLogger("CoTReasoningEngine")
        self.template = self._load_template()
        
        # --- Model Loading (unchanged) ---
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This CoT engine requires a GPU.")
        
        self.device = "cuda"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )
        # NOTE: model_path is hardcoded. Consider making this configurable.
        model_path = "/root/autodl-tmp/openvla/weights/MFM"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, quantization_config=quantization_config,
            torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.logger.info("Qwen2.5-VL model loaded.")

    def _load_template(self) -> Dict[str, str]:
        """
        Loads the CoT analysis prompt template from a JSON file.
        
        Returns:
            Dict[str, str]: The prompt template dictionary.
        """
        try:
            with open(self.config.template_path, 'r', encoding='utf-8') as f:
                return json.load(f)['default_template']
        except Exception as e:
            self.logger.error(f"Failed to load CoT template: {e}")
            # Return a robust fallback template.
            return {
                "system_prompt": "You are a robot failure analyst...",
                "user_prompt": "A robot failed the task: '{task_prompt}'.\n\n**Failure Signature:** {failure_type}\n\n**State Summary:**\n{state_summary}\n\n**Analysis Data:**\n- End-Effector Position: {eef_pos}\n- Gripper Status: {gripper_status}\n\nAnalyze and provide a 'COMMAND:'."
            }

    def analyze_failure(
        self, 
        state_collector_data: Dict[str, Any], 
        latest_image: np.ndarray, # [MODIFIED] Receives the image directly
        failure_type: str, 
        task_prompt: str,
        state_summary: str
    ) -> Tuple[str, str]:
        """
        Analyzes the failure reason using Qwen2.5-VL and generates a feedback command.

        Args:
            state_collector_data (Dict[str, Any]): Data exported from the StateCollector.
            latest_image (np.ndarray): The latest visual observation (as a numpy array).
            failure_type (str): The detected type of failure (e.g., 'GraspFailure').
            task_prompt (str): The original task prompt.
            state_summary (str): A summary of the recent state history.

        Returns:
            Tuple[str, str]: A tuple containing (reasoning_text, recovery_command).
        """
        visual_history = state_collector_data.get('basic_states', [])
        if not visual_history:
            self.logger.error("Cannot get visual history data for analysis!")
            return "Error: No visual data available.", f"try to '{task_prompt}' again"
            
        latest_image_pil = Image.fromarray(latest_image)

        # Prepare context, which now includes the state summary.
        context = self._prepare_context(state_collector_data, failure_type, task_prompt)
        
        user_content = self.template['user_prompt'].format(
            task_prompt=context["task_prompt"],
            failure_type=context["failure_type"],
            state_summary=state_summary,  # [NEW] Pass the summary into the template
            eef_pos=context["eef_pos"],
            gripper_status=context["gripper_status"]
        )
        
        messages = [
            {"role": "system", "content": self.template['system_prompt']},
            {"role": "user", "content": [{"type": "image", "image": latest_image_pil}, {"type": "text", "text": user_content}]}
        ]

        self.logger.info("Sending unified analysis and command generation request to Qwen2.5-VL...")
        self.logger.debug(f"CoT Prompt: {user_content}") # Log the prompt for debugging.

        try:
            # --- Model Inference and Response Parsing (unchanged) ---
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_input], images=[latest_image_pil], return_tensors="pt").to(self.device)
            with torch.no_grad():
                generate_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_tokens, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=self.processor.tokenizer.eos_token_id)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generate_ids)]
            full_response = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            
            reasoning_text, recovery_command = "", ""
            if "COMMAND:" in full_response:
                parts = full_response.split("COMMAND:", 1)
                reasoning_text = parts[0].strip()
                recovery_command = parts[1].strip().split('\n')[0].strip()
            else:
                reasoning_text = full_response
                recovery_command = f"try to '{task_prompt}' again carefully"

            if not recovery_command:
                 recovery_command = f"try to '{task_prompt}' again carefully"

            self.logger.info(f"CoT Analysis Result: {reasoning_text}")
            self.logger.info(f"Generated Recovery Command: {recovery_command}")
            
            return reasoning_text, recovery_command
            
        except Exception as e:
            self.logger.error(f"Error during Qwen2.5-VL analysis: {e}", exc_info=True)
            return f"Error during analysis: {str(e)}", f"try to '{task_prompt}' again"

    def _prepare_context(self, collected_data: Dict[str, Any], failure_type: str, task_prompt: str) -> Dict[str, Any]:
        """
        Prepares the context dictionary for the prompt template.
        (This method remains unchanged, still extracts the latest single state point).

        Args:
            collected_data (Dict[str, Any]): Data from the state collector.
            failure_type (str): The type of failure.
            task_prompt (str): The original task prompt.

        Returns:
            Dict[str, Any]: A dictionary with formatted context information.
        """
        recent_states = collected_data.get('detailed_states', []) or collected_data.get('basic_states', [])
        current_state = recent_states[-1] if recent_states else {}
        robot_state = current_state.get('robot_state', {})
        eef_pos_raw = robot_state.get('eef_pos', 'N/A')
        
        if isinstance(eef_pos_raw, (list, np.ndarray)):
            eef_pos = f"[{eef_pos_raw[0]:.3f}, {eef_pos_raw[1]:.3f}, {eef_pos_raw[2]:.3f}]"
        else:
            eef_pos = str(eef_pos_raw)
            
        gripper_qpos_raw = robot_state.get('gripper_qpos', None)
        gripper_status = "Unknown"
        if gripper_qpos_raw is not None and isinstance(gripper_qpos_raw, (list, np.ndarray)) and len(gripper_qpos_raw) > 0:
            try:
                gripper_qpos_numeric = np.array(gripper_qpos_raw, dtype=np.float64)
                gripper_status = "Open" if np.mean(gripper_qpos_numeric) > 0.01 else "Closed"
            except (ValueError, TypeError):
                gripper_status = "Invalid Data"
        else:
            gripper_status = "Not Available"

        return {
            "task_prompt": task_prompt,
            "failure_type": failure_type,
            "eef_pos": eef_pos,
            "gripper_status": gripper_status,
        }