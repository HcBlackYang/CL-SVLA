# MFM/models/eval_unified_model.py

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Dict
import os
import gc
from PIL import Image

class EvalTrueEndToEndDualViewVLA(nn.Module):
    """
    An end-to-end model that implements CoT analysis to generate an "instruction vector",
    which is then injected into a VLA to guide its actions. This version is adapted
    for memory-efficient evaluation by providing decoupled forward methods.
    
    Architecture: CoT (Qwen) -> Instruction Vector -> VLA (OpenVLA) -> Action
    """
    def __init__(self, vla_path: str, qwen_path: str, use_lora: bool = True, finetune_vla_lora: bool = False):
        super().__init__()
        
        self.use_lora = use_lora
        self.finetune_vla_lora = finetune_vla_lora
        self.model_dtype = torch.bfloat16
        self.trainable_dtype = torch.float32

        print("--- Command-Injecting Dual-View VLA (Evaluation Version) ---")
        print(f"  - Architecture: CoT guides VLA via instruction vector injection")
        print(f"  - Qwen-VL LoRA: {'ENABLED' if self.use_lora else 'DISABLED'}")
        print(f"  - VLA LoRA: {'ENABLED' if self.finetune_vla_lora else 'DISABLED'}")
        print("==============================================================")

        # 1. Load Qwen-VL (for CoT analysis)
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

        # 2. Load OpenVLA (for action execution)
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
        
        # Project Qwen's output features into an "instruction vector" that VLA can understand
        self.instruction_projector = nn.Sequential(
            nn.Linear(qwen_hidden_size, vla_hidden_size), nn.LayerNorm(vla_hidden_size), nn.GELU()
        ).to(self.trainable_dtype)
        
        # 4. VLA's action prediction head (predicts action from VLA's final hidden state)
        self.action_head = nn.Sequential(
            nn.Linear(vla_hidden_size * 2, 512), # Fuse dual-view features
            nn.GELU(), nn.Linear(512, 7), nn.Tanh() # Output 7D action
        ).to(self.trainable_dtype)

        self._ensure_trainable_params_are_fp32()
        print("✅ Command-injecting dual-view model (Eval Version) initialized.")

    def _remove_lm_head(self, model):
        """Removes the language model head to avoid unnecessary computation."""
        if hasattr(model, 'lm_head'): 
            model.lm_head = nn.Identity()

    def _ensure_trainable_params_are_fp32(self):
        """Ensures all trainable parameters are in float32 for stable training."""
        for name, param in self.named_parameters():
            if param.requires_grad and param.dtype != self.trainable_dtype:
                param.data = param.data.to(self.trainable_dtype)

    # 🔥 Decoupled Method 1: CoT Analysis Only
    def forward_cot_only(self, batch: Dict, device: str) -> torch.Tensor:
        """
        🔥 Decoupled execution: Performs only the CoT analysis part, outputting an instruction vector.
        This allows the CoT model to run on the GPU while the VLA remains on the CPU.
        """
        qwen_texts, qwen_images = [], []
        for i in range(batch['total_batch_size']):
            messages = [{"role": "system", "content": "You are a helpful robot assistant."}]
            task_prompt, failure_context = batch['task_prompts'][i], batch['failure_contexts'][i]
            current_obs_text = f"Analyze the situation for task '{task_prompt}'. My third-person view and first-person view are provided. Expert analysis: {failure_context}"
            messages.append({"role": "user", "content": [{"type": "text", "text": current_obs_text}, {"type": "image"}, {"type": "image"}]})
            qwen_images.extend([batch['current_agentview_images'][i], batch['current_robot_eye_images'][i]])
            qwen_texts.append(self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            
        qwen_inputs = self.qwen_processor(text=qwen_texts, images=qwen_images, return_tensors="pt", padding=True).to(device)
        qwen_inputs['pixel_values'] = qwen_inputs['pixel_values'].to(self.model_dtype)
        
        qwen_outputs = self.qwen_model(**qwen_inputs, output_hidden_states=True)
        cot_features = qwen_outputs.hidden_states[-1][:, -1, :]
        instruction_vectors = self.instruction_projector(cot_features)
        
        del qwen_outputs, qwen_inputs; gc.collect(); torch.cuda.empty_cache()
        return instruction_vectors

    # 🔥 Decoupled Method 2: VLA Execution
    def forward_vla_with_instruction(self, batch: Dict, instruction_vectors: torch.Tensor, device: str) -> torch.Tensor:
        """
        🔥 Decoupled execution: Receives an instruction vector and performs the VLA part.
        This allows the VLA to run on the GPU while the CoT model remains on the CPU.
        """
        instruction_vectors = instruction_vectors.to(device)
        vla_prompts = [f"In: {p}\nOut:" for p in batch['task_prompts']]
        
        vla_agent_inputs = self.vla_processor(text=vla_prompts, images=batch['current_agentview_images'], return_tensors="pt", padding=True).to(device)
        vla_agent_inputs['pixel_values'] = vla_agent_inputs['pixel_values'].to(self.model_dtype)
        vla_eye_inputs = self.vla_processor(text=vla_prompts, images=batch['current_robot_eye_images'], return_tensors="pt", padding=True).to(device)
        vla_eye_inputs['pixel_values'] = vla_eye_inputs['pixel_values'].to(self.model_dtype)

        agent_input_embeds = self.vla_model.get_input_embeddings()(vla_agent_inputs['input_ids'])
        eye_input_embeds = self.vla_model.get_input_embeddings()(vla_eye_inputs['input_ids'])

        # 🔥 Instruction Injection: Add the instruction vector to VLA's input embeddings
        agent_input_embeds[:, 0, :] += instruction_vectors
        eye_input_embeds[:, 0, :] += instruction_vectors

        agent_outputs = self.vla_model(inputs_embeds=agent_input_embeds, pixel_values=vla_agent_inputs['pixel_values'], output_hidden_states=True)
        agent_features = agent_outputs.hidden_states[-1][:, -1, :]
        eye_outputs = self.vla_model(inputs_embeds=eye_input_embeds, pixel_values=vla_eye_inputs['pixel_values'], output_hidden_states=True)
        eye_features = eye_outputs.hidden_states[-1][:, -1, :]

        combined_features = torch.cat([agent_features, eye_features], dim=1)
        predicted_action = self.action_head(combined_features)
        
        del agent_outputs, eye_outputs, vla_agent_inputs, vla_eye_inputs; gc.collect(); torch.cuda.empty_cache()
        return predicted_action

    # 🔥 Enhanced forward method
    def forward(self, batch: Dict, device: str, decoupled: bool = False) -> torch.Tensor:
        """
        Enhanced forward method supporting decoupled execution for memory efficiency.
        """
        if decoupled:
            raise NotImplementedError("For decoupled mode, please call forward_cot_only and forward_vla_with_instruction sequentially, managing device placement externally.")
        else:
            return self._forward_end_to_end(batch, device)

    def _forward_end_to_end(self, batch: Dict, device: str) -> torch.Tensor:
        """Original end-to-end forward implementation, primarily for training compatibility."""
        # 1. CoT analysis to generate instruction vector
        qwen_texts, qwen_images = [], []
        for i in range(batch['total_batch_size']):
            messages = [{"role": "system", "content": "You are a helpful robot assistant."}]
            task_prompt, failure_context = batch['task_prompts'][i], batch['failure_contexts'][i]
            current_obs_text = f"Analyze the situation for task '{task_prompt}'. My third-person view and first-person view are provided. Expert analysis: {failure_context}"
            messages.append({"role": "user", "content": [{"type": "text", "text": current_obs_text}, {"type": "image"}, {"type": "image"}]})
            qwen_images.extend([batch['current_agentview_images'][i], batch['current_robot_eye_images'][i]])
            qwen_texts.append(self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            
        qwen_inputs = self.qwen_processor(text=qwen_texts, images=qwen_images, return_tensors="pt", padding=True).to(device)
        qwen_inputs['pixel_values'] = qwen_inputs['pixel_values'].to(self.model_dtype)
        
        qwen_outputs = self.qwen_model(**qwen_inputs, output_hidden_states=True)
        cot_features = qwen_outputs.hidden_states[-1][:, -1, :]
        instruction_vectors = self.instruction_projector(cot_features)
        
        del qwen_outputs, qwen_inputs; gc.collect(); torch.cuda.empty_cache()

        # 2. VLA execution with injected instruction
        vla_prompts = [f"In: {p}\nOut:" for p in batch['task_prompts']]
        vla_agent_inputs = self.vla_processor(text=vla_prompts, images=batch['current_agentview_images'], return_tensors="pt", padding=True).to(device)
        vla_agent_inputs['pixel_values'] = vla_agent_inputs['pixel_values'].to(self.model_dtype)
        vla_eye_inputs = self.vla_processor(text=vla_prompts, images=batch['current_robot_eye_images'], return_tensors="pt", padding=True).to(device)
        vla_eye_inputs['pixel_values'] = vla_eye_inputs['pixel_values'].to(self.model_dtype)

        agent_input_embeds = self.vla_model.get_input_embeddings()(vla_agent_inputs['input_ids'])
        eye_input_embeds = self.vla_model.get_input_embeddings()(vla_eye_inputs['input_ids'])

        agent_input_embeds[:, 0, :] += instruction_vectors
        eye_input_embeds[:, 0, :] += instruction_vectors

        agent_outputs = self.vla_model(inputs_embeds=agent_input_embeds, pixel_values=vla_agent_inputs['pixel_values'], output_hidden_states=True)
        agent_features = agent_outputs.hidden_states[-1][:, -1, :]
        eye_outputs = self.vla_model(inputs_embeds=eye_input_embeds, pixel_values=vla_eye_inputs['pixel_values'], output_hidden_states=True)
        eye_features = eye_outputs.hidden_states[-1][:, -1, :]

        combined_features = torch.cat([agent_features, eye_features], dim=1)
        predicted_action = self.action_head(combined_features)
        
        return predicted_action

    def save_lora_weights(self, save_path: str):
        """Saves the LoRA weights and connector layers."""
        os.makedirs(save_path, exist_ok=True)
        
        if self.use_lora: self.qwen_model.save_pretrained(os.path.join(save_path, "qwen_lora"))
        torch.save({'instruction_projector': self.instruction_projector.state_dict(), 'action_head': self.action_head.state_dict()}, os.path.join(save_path, "connector_and_head.pth"))
        if self.finetune_vla_lora: self.vla_model.save_pretrained(os.path.join(save_path, "vla_lora"))
            
        print(f"✅ Command-injecting model (Eval Version) saved to {save_path}")