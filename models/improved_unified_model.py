# MFM/models/improved_unified_model.py
# This file defines the command-injecting VLA model for training.

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Dict
import os
import gc
from PIL import Image

class TrueEndToEndDualViewVLA(nn.Module):
    """
    Implements an end-to-end model where CoT analysis generates an "instruction vector"
    that is injected into a VLA to guide its actions.
    
    Architecture: CoT (Qwen) -> Instruction Vector -> VLA (OpenVLA) -> Action
    """
    def __init__(self, vla_path: str, qwen_path: str, use_lora: bool = True, finetune_vla_lora: bool = False):
        super().__init__()
        
        self.use_lora = use_lora
        self.finetune_vla_lora = finetune_vla_lora
        self.model_dtype = torch.bfloat16
        self.trainable_dtype = torch.float32

        print("--- Command-Injecting Dual-View VLA ---")
        print(f"  - Architecture: CoT guides VLA via instruction vector injection")
        print(f"  - Qwen-VL LoRA: {'ENABLED' if self.use_lora else 'DISABLED'}")
        print(f"  - VLA LoRA: {'ENABLED' if self.finetune_vla_lora else 'DISABLED'}")
        print("==========================================")

        # 1. Load Qwen-VL (CoT Analyzer)
        print("Loading Qwen-VL for CoT analysis...")
        self.qwen_processor = AutoProcessor.from_pretrained(qwen_path, trust_remote_code=True)
        self.qwen_model = AutoModelForVision2Seq.from_pretrained(
            qwen_path, torch_dtype=self.model_dtype, low_cpu_mem_usage=True, trust_remote_code=True)
        
        if use_lora:
            qwen_lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], bias="none")
            self.qwen_model = get_peft_model(self.qwen_model, qwen_lora_config)
            self.qwen_model.print_trainable_parameters()
        
        self.qwen_model.gradient_checkpointing_enable()
        self._remove_lm_head(self.qwen_model)

        # 2. Load OpenVLA (Action Executor)
        print("Loading OpenVLA for action execution...")
        self.vla_processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
        self.vla_model = AutoModelForVision2Seq.from_pretrained(
            vla_path, torch_dtype=self.model_dtype, attn_implementation="flash_attention_2", low_cpu_mem_usage=True, trust_remote_code=True)
        
        if self.finetune_vla_lora:
            vla_lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], bias="none")
            self.vla_model = get_peft_model(self.vla_model, vla_lora_config)
            self.vla_model.print_trainable_parameters()
        else: # Freeze VLA
            for param in self.vla_model.parameters():
                param.requires_grad = False
        
        self.vla_model.gradient_checkpointing_enable()

        # 3. Key Connector Layers
        qwen_hidden_size = self.qwen_model.config.text_config.hidden_size
        vla_hidden_size = self.vla_model.config.text_config.hidden_size
        
        # Project Qwen's output features into an "instruction vector"
        self.instruction_projector = nn.Sequential(
            nn.Linear(qwen_hidden_size, vla_hidden_size), nn.LayerNorm(vla_hidden_size), nn.GELU()
        ).to(self.trainable_dtype)
        
        # 4. VLA's action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(vla_hidden_size * 2, 512), # Fuses dual-view features
            nn.GELU(), nn.Linear(512, 7), nn.Tanh()
        ).to(self.trainable_dtype)

        self._ensure_trainable_params_are_fp32()
        print("✅ Command-injecting dual-view model initialized.")

    def _remove_lm_head(self, model):
        """Removes the language model head to save computation."""
        if hasattr(model, 'lm_head'): 
            model.lm_head = nn.Identity()

    def _ensure_trainable_params_are_fp32(self):
        """Ensures all trainable parameters are float32 for stable training."""
        for name, param in self.named_parameters():
            if param.requires_grad and param.dtype != self.trainable_dtype:
                param.data = param.data.to(self.trainable_dtype)

    def forward(self, batch: Dict, device: str) -> torch.Tensor:
        """
        Performs a full forward pass through the command-injecting model.

        Args:
            batch (Dict): A dictionary containing batch data (images, prompts, etc.).
            device (str): The device to run the computation on ('cuda' or 'cpu').

        Returns:
            torch.Tensor: The predicted 7D action tensor.
        """
        # 1. CoT Analysis to generate instruction vector
        qwen_texts, qwen_images = [], []
        for i in range(batch['total_batch_size']):
            messages = [{"role": "system", "content": "You are a helpful robot assistant."}]
            task_prompt, failure_context = batch['task_prompts'][i], batch['failure_contexts'][i]
            
            # Dual-view analysis prompt
            current_obs_text = f"Analyze the situation for task '{task_prompt}'. My third-person view and first-person view are provided. Expert analysis: {failure_context}"
            messages.append({"role": "user", "content": [{"type": "text", "text": current_obs_text}, {"type": "image"}, {"type": "image"}]})
            qwen_images.extend([batch['current_agentview_images'][i], batch['current_robot_eye_images'][i]])
            qwen_texts.append(self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            
        qwen_inputs = self.qwen_processor(text=qwen_texts, images=qwen_images, return_tensors="pt", padding=True).to(device)
        qwen_inputs['pixel_values'] = qwen_inputs['pixel_values'].to(self.model_dtype)
        
        # Get Qwen's final hidden state
        qwen_outputs = self.qwen_model(**qwen_inputs, output_hidden_states=True)
        cot_features = qwen_outputs.hidden_states[-1][:, -1, :]
        instruction_vectors = self.instruction_projector(cot_features)
        
        del qwen_outputs, qwen_inputs; gc.collect(); torch.cuda.empty_cache()

        # 2. VLA Execution with injected instruction
        vla_prompts = [f"In: {p}\nOut:" for p in batch['task_prompts']]
        
        # Process agent-view
        vla_agent_inputs = self.vla_processor(text=vla_prompts, images=batch['current_agentview_images'], return_tensors="pt", padding=True).to(device)
        vla_agent_inputs['pixel_values'] = vla_agent_inputs['pixel_values'].to(self.model_dtype)
        # Process eye-in-hand view
        vla_eye_inputs = self.vla_processor(text=vla_prompts, images=batch['current_robot_eye_images'], return_tensors="pt", padding=True).to(device)
        vla_eye_inputs['pixel_values'] = vla_eye_inputs['pixel_values'].to(self.model_dtype)

        agent_input_embeds = self.vla_model.get_input_embeddings()(vla_agent_inputs['input_ids'])
        eye_input_embeds = self.vla_model.get_input_embeddings()(vla_eye_inputs['input_ids'])

        # 🔥 Instruction Injection: Add the instruction vector to VLA's input embeddings
        agent_input_embeds[:, 0, :] += instruction_vectors
        eye_input_embeds[:, 0, :] += instruction_vectors

        # VLA forward pass
        agent_outputs = self.vla_model(inputs_embeds=agent_input_embeds, pixel_values=vla_agent_inputs['pixel_values'], output_hidden_states=True)
        agent_features = agent_outputs.hidden_states[-1][:, -1, :]
        eye_outputs = self.vla_model(inputs_embeds=eye_input_embeds, pixel_values=vla_eye_inputs['pixel_values'], output_hidden_states=True)
        eye_features = eye_outputs.hidden_states[-1][:, -1, :]

        # 3. Fuse dual-view features and predict action
        combined_features = torch.cat([agent_features, eye_features], dim=1)
        predicted_action = self.action_head(combined_features)
        
        return predicted_action

    def save_lora_weights(self, save_path: str):
        """Saves LoRA weights and connector layers."""
        os.makedirs(save_path, exist_ok=True)
        
        if self.use_lora: self.qwen_model.save_pretrained(os.path.join(save_path, "qwen_lora"))
        torch.save({'instruction_projector': self.instruction_projector.state_dict(), 'action_head': self.action_head.state_dict()}, os.path.join(save_path, "connector_and_head.pth"))
        if self.finetune_vla_lora: self.vla_model.save_pretrained(os.path.join(save_path, "vla_lora"))
            
        print(f"✅ Command-injecting model saved to {save_path}")