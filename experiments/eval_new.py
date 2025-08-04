#!/usr/bin/env python3
"""
Modified evaluation script: Phase 1 (Pure VLA) + Phase 2 (End-to-end CoT+VLA)
🔥 Key Logic:
- Phase 1: Executes the task using the fine-tuned VLA. Upon failure detection,
           it stops immediately and saves the state.
- Phase 2: Uses the fine-tuned end-to-end CoT+VLA system.
  * Initially, CoT is on the GPU, and VLA is on the CPU.
  * CoT analyzes the situation and outputs an instruction vector for the VLA.
  * Then, CoT is moved to the CPU, and VLA is moved to the GPU for execution.
"""

import os
import sys
import torch
import numpy as np
import tqdm
import json
import pickle
import time
from pathlib import Path
from PIL import Image
import gc
import cv2

# --- [Path and Module Imports] ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OPENVLA_SRC_DIR = PROJECT_ROOT / "openvla" / "OpenVLA"
if not OPENVLA_SRC_DIR.is_dir():
    raise FileNotFoundError(f"OpenVLA source directory not found at: {OPENVLA_SRC_DIR}")
sys.path.insert(0, str(OPENVLA_SRC_DIR))
from MFM.config import Config as OpenVLAConfig
import MFM.utils.utils as vla_utils
from MFM.config import MFMConfig
from MFM.core import MetacognitiveFeedbackModuleEval
from MFM.utils import setup_logger
from libero.libero import benchmark

# --- [Core Model Import] ---
from MFM.models.eval_unified_model import EvalTrueEndToEndDualViewVLA
from peft import PeftModel

# --- [Helper Functions: Consistent with Original Code] ---
def save_dual_view_video_with_step_counter(
    agentview_images: list, eye_in_hand_images: list, video_path: Path, 
    fps: int = 20, task_prompt: str = ""
):
    """Saves a side-by-side dual-view video with a step counter."""
    if not agentview_images or not eye_in_hand_images: return
    num_frames = min(len(agentview_images), len(eye_in_hand_images))
    if num_frames == 0: return
    h, w, _ = agentview_images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w * 2, h))
    font, font_scale, font_color, line_type = cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    for i in range(num_frames):
        combined_frame = np.concatenate((agentview_images[i], eye_in_hand_images[i]), axis=1)
        step_text = f"Step: {i+1}"; text_size = cv2.getTextSize(step_text, font, font_scale, line_type)[0]
        text_x, text_y = combined_frame.shape[1] - text_size[0] - 10, text_size[1] + 10
        cv2.rectangle(combined_frame, (text_x-5, text_y-text_size[1]-5), (text_x+text_size[0]+5, text_y+5), (0,0,0), -1)
        cv2.putText(combined_frame, step_text, (text_x, text_y), font, font_scale, font_color, line_type)
        if task_prompt:
             prompt_text = (task_prompt[:40] + '...') if len(task_prompt) > 40 else task_prompt
             cv2.putText(combined_frame, prompt_text, (10, 20), font, 0.5, (0,0,0), 3); cv2.putText(combined_frame, prompt_text, (10, 20), font, 0.5, (255,255,255), 1)
        writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
    writer.release()

def convert_numpy_types(data):
    """Recursively converts numpy types to native Python types for JSON serialization."""
    if isinstance(data, dict): return {k: convert_numpy_types(v) for k, v in data.items()}
    if isinstance(data, list): return [convert_numpy_types(i) for i in data]
    if isinstance(data, (np.integer, np.int_)): return int(data)
    if isinstance(data, (np.floating, np.float_)): return float(data)
    if isinstance(data, np.ndarray): return data.tolist()
    return data

class DarkenedEnvironmentWrapper:
    """A wrapper to darken the environment's visual observations."""
    def __init__(self, env, darkness_factor=1, add_noise=False, noise_intensity=0):
        self.env, self.darkness_factor, self.add_noise, self.noise_intensity = env, darkness_factor, add_noise, noise_intensity
        self.original_darkness_factor, self.original_add_noise, self.original_noise_intensity = darkness_factor, add_noise, noise_intensity
        
    def restore_original_lighting(self):
        """Restores the original bright lighting conditions."""
        self.darkness_factor = 1.0
        self.add_noise = False
        
    def restore_darkened_lighting(self):
        """Restores the configured darkened lighting conditions."""
        self.darkness_factor = self.original_darkness_factor
        self.add_noise = self.original_add_noise
        
    def darken_image(self, image: np.ndarray) -> np.ndarray:
        """Applies darkness and optional noise to a single image."""
        if not (isinstance(image, np.ndarray) and len(image.shape) == 3): return image
        darkened = image.astype(np.float32) * self.darkness_factor
        if self.add_noise: darkened += np.random.normal(0, self.noise_intensity * 255, image.shape)
        return np.clip(darkened, 0, 255).astype(np.uint8)
    def process_observation(self, obs: dict) -> dict: return {k: self.darken_image(v) if 'image' in k else v for k, v in obs.items()} if obs else None
    def step(self, action): obs, r, d, i = self.env.step(action); return self.process_observation(obs), r, d, i
    def reset(self, *args, **kwargs): return self.process_observation(self.env.reset(*args, **kwargs))
    def set_init_state(self, *args, **kwargs): return self.process_observation(self.env.set_init_state(*args, **kwargs))
    def __getattr__(self, name): return getattr(self.env, name)

def process_vla_action(raw_action: np.ndarray) -> np.ndarray:
    """Post-processes the raw VLA action output."""
    action = raw_action.copy(); action[..., -1] = (2 * action[..., -1] - 1); action[..., -1] = np.sign(action[..., -1]) * -1.0; return action

def save_episode_state(env, task_id, ep_idx, step, init_state, save_dir):
    """Saves the current episode state (including simulation state) to a .pkl file."""
    state_data = {
        'task_id': task_id, 'episode_idx': ep_idx, 'step': step,
        'init_state': init_state, 'env_state': env.sim.get_state(), 'timestamp': time.time()
    }
    pkl_path = save_dir / f"task{task_id}_ep{ep_idx}_step{step}_state.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(state_data, f)
    return pkl_path

def safe_gpu_reset():
    """A safe function to reset GPU memory to avoid allocator crashes."""
    import gc
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'): torch.cuda.reset_peak_memory_stats()
            if hasattr(torch.cuda, 'reset_accumulated_memory_stats'): torch.cuda.reset_accumulated_memory_stats()
        return True
    except Exception as e:
        print(f"⚠️ GPU reset warning: {e}")
        return False

def load_episode_state(pkl_path, env):
    """Loads an episode state from a .pkl file and applies it to the environment."""
    with open(pkl_path, 'rb') as f:
        state_data = pickle.load(f)
    env.reset()
    env.sim.set_state(state_data['env_state'])
    env.sim.forward()
    return state_data

def execute_phase2_with_decoupled_e2e_model(env, unified_model, task_prompt, max_steps, logger, device, cpu_device):
    """
    🔥 Phase 2: Single-model alternating execution - only one large model is on the GPU at any time.
    (Image input issue has been fixed in this version).
    """
    logger.info(f"🎯 Starting Phase 2: Single-model alternating execution, Task: '{task_prompt}'")
    
    # Get initial observation state
    obs, _, done, info = env.step(np.zeros(7))
    if info.get('success', False):
        logger.info(f"🏆 Succeeded immediately after reloading!")
        return [], [], 0, done, info
    
    t, images_agent, images_eye, cot_analysis_result = 0, [], [], None
    
    def aggressive_memory_cleanup():
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ================================
    # 🧠 Phase 2.1: Pure CoT Analysis
    # ================================
    logger.info("🧠 Starting Phase 2.1: Pure CoT analysis...")
    try:
        logger.debug("📍 Loading only CoT model to GPU...")
        unified_model.qwen_model.to(device)
        unified_model.instruction_projector.to(device)
        aggressive_memory_cleanup()
        
        if torch.cuda.is_available(): logger.info(f"💾 CoT GPU Memory Footprint: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        
        current_obs = {
            'agentview_image': obs["agentview_image"][::-1, ::-1].copy(),
            'robot0_eye_in_hand_image': obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
        }
        
        with torch.no_grad():
            qwen_tokenizer = unified_model.qwen_processor.tokenizer
            
            # =================================================================
            # 💡 [KEY FIX]: Use `apply_chat_template` to build the correct multimodal input.
            # =================================================================
            logger.debug("🖼️ Building multimodal input using the correct chat template...")
            
            failure_context = "The initial VLA execution failed. Need step-by-step action guidance."
            user_text = f"Task: {task_prompt}\nContext: {failure_context}\n\nPlease provide clear, step-by-step action instructions for this task based on the following views.\nAgent view (third-person) and robot hand view (first-person) are provided.\nFormat your response as actionable steps.\nExample: \"The correct action would be to open the gripper, pick up the bowl, place it on the plate, then open the gripper, and then raise up.\"\n\nAction Instructions:"
            messages = [{"role": "system", "content": "You are a helpful robot assistant."}, {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image"}, {"type": "image"}]}]
            
            # 1. Generate text using the template
            prompt_text = unified_model.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 2. Pass text and images together to the processor
            agentview_img = Image.fromarray(current_obs['agentview_image'])
            robot_eye_img = Image.fromarray(current_obs['robot0_eye_in_hand_image'])
            qwen_inputs = unified_model.qwen_processor(text=[prompt_text], images=[agentview_img, robot_eye_img], return_tensors="pt").to(device)
            qwen_inputs['pixel_values'] = qwen_inputs['pixel_values'].to(unified_model.model_dtype)
            logger.debug("✅ Successfully built Qwen input.")
            # ====================== [END OF FIX] ========================
            
            # Generate analysis text
            logger.debug("🔍 Generating CoT action instructions...")
            try:
                qwen_outputs = unified_model.qwen_model.generate(**qwen_inputs, max_new_tokens=128, do_sample=False, pad_token_id=qwen_tokenizer.eos_token_id)
                response = qwen_tokenizer.decode(qwen_outputs[0], skip_special_tokens=True)
                # Qwen-VL output often includes the input prompt, so we strip it.
                cot_instruction = response.split("Action Instructions:")[-1].strip()

                if len(cot_instruction) < 20: # Fallback for very short/empty generation
                    logger.warning("⚠️ CoT generated instruction is too short, using default template.")
                    cot_instruction = f"The correct action would be to carefully {task_prompt.lower()}, step by step."
                
                logger.info(f"🎯 CoT Action Instruction: {cot_instruction}")
            except Exception as gen_e:
                logger.error(f"❌ Generation process failed: {gen_e}", exc_info=True)
                cot_instruction = f"The correct action would be to open the gripper, carefully {task_prompt.lower()}, then close the gripper and lift up."
                logger.warning(f"⚠️ Using default CoT instruction: {cot_instruction}")
            
            cot_analysis_result = {'action_instruction': cot_instruction, 'instruction_embedding': None, 'original_task': task_prompt, 'used_vision': True}
        
        logger.info("✅ CoT analysis complete.")
    except Exception as e:
        logger.error(f"❌ CoT analysis failed: {e}", exc_info=True)
        cot_analysis_result = None
    finally:
        logger.info("🧹 CoT analysis finished, clearing GPU...")
        try:
            unified_model.qwen_model.to(cpu_device); unified_model.instruction_projector.to(cpu_device)
            aggressive_memory_cleanup()
        except Exception as cleanup_e:
            logger.error(f"❌ CoT cleanup failed: {cleanup_e}")

    # ================================
    # 🤖 Phase 2.2: Pure VLA Execution
    # ================================
    if cot_analysis_result is not None:
        logger.info("🤖 Starting Phase 2.2: Pure VLA execution...")
        try:
            logger.debug("📍 Loading only VLA model to GPU...")
            unified_model.vla_model.to(device); unified_model.action_head.to(device)
            aggressive_memory_cleanup()
            if torch.cuda.is_available(): logger.info(f"💾 VLA GPU Memory Footprint: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            
            cot_instruction = cot_analysis_result['action_instruction']
            logger.info(f"🎯 Using CoT instruction: {cot_instruction}")
            
            while t < max_steps:
                try:
                    images_agent.append(obs["agentview_image"][::-1, ::-1].copy())
                    images_eye.append(obs["robot0_eye_in_hand_image"][::-1, ::-1].copy())
                    
                    with torch.no_grad():
                        img_for_vla = Image.fromarray(images_agent[-1]).resize((224, 224))
                        vla_prompt = f"In: {cot_instruction}\nOut:"
                        vla_inputs = unified_model.vla_processor(text=vla_prompt, images=img_for_vla, return_tensors="pt").to(device)
                        vla_inputs['pixel_values'] = vla_inputs['pixel_values'].to(unified_model.model_dtype)
                        raw_action = unified_model.vla_model.predict_action(**vla_inputs, unnorm_key="libero_spatial", do_sample=False).flatten()
                    
                    aggressive_memory_cleanup()
                    obs, _, done, info = env.step(process_vla_action(raw_action))
                    t += 1
                    
                    if info.get('success', False): logger.info(f"🏆 VLA execution succeeded at step {t}!"); break
                    elif done: logger.warning(f"⚠️ VLA execution ended at step {t} without success."); break
                except Exception as step_e:
                    logger.error(f"❌ VLA execution step {t} failed: {step_e}", exc_info=True)
                    break
            
            logger.info(f"✅ VLA execution phase complete after {t} steps.")
        except Exception as e:
            logger.error(f"❌ VLA execution phase failed: {e}", exc_info=True)
        finally:
            logger.info("🧹 VLA execution finished, clearing GPU...")
            try:
                unified_model.vla_model.to(cpu_device); unified_model.action_head.to(cpu_device)
                aggressive_memory_cleanup()
            except Exception as cleanup_e:
                logger.error(f"❌ VLA cleanup failed: {cleanup_e}")
    else:
        logger.error("❌ CoT analysis failed, skipping VLA execution.")
    
    logger.info(f"🏁 Phase 2 complete: CoT analysis + VLA execution ({t} steps).")
    return images_agent, images_eye, t, done, info

def eval_with_phase1_mfm_vla_phase2_e2e():
    """Main evaluation function with a two-phase approach."""
    # --- [Configuration] ---
    import gc
    BASE_VLA_PATH = "/root/autodl-tmp/openvla/weights/openvla-7b-finetuned-libero-spatial"
    BASE_QWEN_PATH = "/root/autodl-tmp/openvla/weights/MFM"
    FINETUNED_CHECKPOINT_DIR = "/root/autodl-tmp/openvla/MFM/data/spatial_checkpoints/best_model"

    # 🔥 Aggressive memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    DEVICE, CPU_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), torch.device("cpu")
    openvla_config, mfm_config = OpenVLAConfig(), MFMConfig()
    logger = setup_logger("Phase1MFM_Phase2E2E_Eval", mfm_config.log_level)
    
    # GPU memory management setup
    if torch.cuda.is_available(): safe_gpu_reset()
    LIGHTING_CONFIG = {'enabled': True, 'darkness_factor': 1, 'add_noise': True, 'noise_intensity': 0}
    PHASE2_ENABLED, has_decoupled_methods = False, False

    # --- [Results Directory Setup] ---
    run_id = f"PHASE1MFM-PHASE2E2E-EVAL-{openvla_config.task_suite_name}-{vla_utils.DATE_TIME}"
    results_dir = Path(mfm_config.log_dir) / run_id
    videos_dir, logs_dir, states_dir = results_dir / "videos", results_dir / "logs", results_dir / "states"
    results_dir.mkdir(parents=True, exist_ok=True); videos_dir.mkdir(); logs_dir.mkdir(); states_dir.mkdir()
    
    logger.info(f"🚀 Starting evaluation: Phase 1 (MFM-supervised VLA) + Phase 2 (Alternating CoT+VLA)")
    logger.info(f"📁 Results will be saved to: {results_dir}")

    # --- [1. Load Unified Model] ---
    logger.info("📦 Loading the end-to-end unified model...")
    try:
        unified_model = EvalTrueEndToEndDualViewVLA(vla_path=BASE_VLA_PATH, qwen_path=BASE_QWEN_PATH, use_lora=True, finetune_vla_lora=True)
        unified_model.model_dtype = torch.bfloat16
        if hasattr(unified_model, 'trainable_dtype'): unified_model.trainable_dtype = torch.bfloat16
        
        # Load fine-tuned weights
        logger.info("📂 Loading fine-tuned weights...")
        if os.path.exists(os.path.join(FINETUNED_CHECKPOINT_DIR, "qwen_lora")):
            unified_model.qwen_model = PeftModel.from_pretrained(unified_model.qwen_model, os.path.join(FINETUNED_CHECKPOINT_DIR, "qwen_lora"))
            logger.info("✅ Qwen LoRA weights loaded.")
        if os.path.exists(os.path.join(FINETUNED_CHECKPOINT_DIR, "vla_lora")):
            unified_model.vla_model = PeftModel.from_pretrained(unified_model.vla_model, os.path.join(FINETUNED_CHECKPOINT_DIR, "vla_lora"))
            logger.info("✅ VLA LoRA weights loaded.")
        connector_path = os.path.join(FINETUNED_CHECKPOINT_DIR, "connector_and_head.pth")
        if os.path.exists(connector_path):
            connector_dict = torch.load(connector_path, map_location='cpu')
            if 'instruction_projector' in connector_dict: unified_model.instruction_projector.load_state_dict(connector_dict['instruction_projector']); logger.info("✅ Instruction projector weights loaded.")
            if 'action_head' in connector_dict: unified_model.action_head.load_state_dict(connector_dict['action_head']); logger.info("✅ Action head weights loaded.")
        
        # Initial device placement
        logger.info("📍 Performing initial device placement...")
        unified_model.to(CPU_DEVICE) # Move all to CPU first
        unified_model.vla_model.to(DEVICE); unified_model.instruction_projector.to(DEVICE); unified_model.action_head.to(DEVICE)
        
        gc.collect(); torch.cuda.empty_cache()
        unified_model.eval()
        
        # Check for decoupled method support and sufficient memory
        has_decoupled_methods = hasattr(unified_model, 'forward_cot_only') and hasattr(unified_model, 'forward_vla_with_instruction')
        gpu_free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3 if torch.cuda.is_available() else 0
        if has_decoupled_methods and gpu_free_memory > 10.0:
            logger.info(f"✅ Memory sufficient ({gpu_free_memory:.2f}GB) and model supports decoupling. Phase 2 enabled.")
            PHASE2_ENABLED = True
        else:
            logger.error(f"❌ Phase 2 disabled. Decoupled support: {has_decoupled_methods}, Free VRAM: {gpu_free_memory:.2f}GB (needs > 10GB).")
            PHASE2_ENABLED = False

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}", exc_info=True)
        return

    # --- [2. Initialize MFM Module] ---
    logger.info("🧠 Initializing MFM module...")
    def cot_analyzer_func(state_collector, task_prompt, failure_type):
        """A dummy CoT analyzer that just returns a trigger signal for Phase 2."""
        logger.info(f"🧠 MFM detected failure '{failure_type}', triggering Phase 2.")
        return f"MFM detected {failure_type}. Triggering phase 2 execution."
    mfm_module = MetacognitiveFeedbackModuleEval(mfm_config, cot_analyzer_func)
    
    # --- [3. Main Evaluation Loop] ---
    task_suite = benchmark.get_benchmark_dict()[openvla_config.task_suite_name]()
    total_episodes, total_successes, total_interventions = 0, 0, 0
    all_results = []
    
    for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc="All Tasks"):
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        orig_env, task_desc = vla_utils.get_libero_env(task=task, model_family=openvla_config.model_family, camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=[512, 512], camera_widths=[512, 512])
        env = DarkenedEnvironmentWrapper(orig_env, **{k: v for k, v in LIGHTING_CONFIG.items() if k != 'enabled'}) if LIGHTING_CONFIG['enabled'] else orig_env
        
        for ep_idx in tqdm.tqdm(range(openvla_config.num_trials_per_task), desc=f"Task {task_id}", leave=False):
            env.reset(); obs = env.set_init_state(init_states[ep_idx])
            if hasattr(env, 'restore_darkened_lighting'): env.restore_darkened_lighting()
            original_prompt = task_desc.lower()
            
            # 🔥 Phase 1: MFM-Supervised VLA Execution
            logger.info(f"\n🎬 Starting Task {task_id} Episode {ep_idx} - Phase 1: MFM-Supervised VLA")
            mfm_module.start_monitoring(original_prompt)
            t, images_agent, images_eye, done, info, mfm_failure_detected, failure_step, pkl_path = 0, [], [], False, {}, False, -1, None

            while t < openvla_config.max_steps:
                images_agent.append(obs["agentview_image"][::-1, ::-1].copy())
                images_eye.append(obs["robot0_eye_in_hand_image"][::-1, ::-1].copy())
                img_for_vla = vla_utils.resize_image(images_agent[-1], (224, 224))
                with torch.no_grad():
                    inputs = unified_model.vla_processor(text=f"In: {original_prompt}\nOut:", images=Image.fromarray(img_for_vla), return_tensors="pt").to(DEVICE)
                    inputs['pixel_values'] = inputs['pixel_values'].to(unified_model.model_dtype)
                    raw_action = unified_model.vla_model.predict_action(**inputs, unnorm_key="libero_spatial", do_sample=False)
                
                obs, reward, done, info = env.step(process_vla_action(raw_action))
                t += 1
                
                intervention_needed, feedback = mfm_module.process_execution_step(obs, raw_action, reward, done, info, env)
                if info.get('success', False): logger.info(f"🏆 Phase 1 Success! Task {task_id} Ep {ep_idx} in {t} steps."); break
                if intervention_needed and feedback and PHASE2_ENABLED:
                    logger.warning(f"🚨 MFM detected failure at step {t}. Saving state for Phase 2."); mfm_failure_detected = True; failure_step = t; total_interventions += 1
                    safe_gpu_reset()
                    try: pkl_path = save_episode_state(env, task_id, ep_idx, t, init_states[ep_idx], states_dir); logger.info(f"💾 State saved to: {pkl_path}")
                    except Exception as e: logger.error(f"❌ State saving failed: {e}", exc_info=True)
                    break
                if done and not info.get('success', False): logger.error(f"❌ Phase 1 ended prematurely at step {t}."); break

            mfm_summary = mfm_module.stop_monitoring()
            
            # 🔥 Phase 2: Decoupled E2E Model Execution (if triggered)
            if mfm_failure_detected and pkl_path and PHASE2_ENABLED:
                logger.info(f"\n🔄 Starting Task {task_id} Episode {ep_idx} - Phase 2: Decoupled E2E Model")
                try:
                    safe_gpu_reset()
                    unified_model.vla_model.to(CPU_DEVICE); gc.collect(); torch.cuda.empty_cache()
                    load_episode_state(pkl_path, env)
                    if hasattr(env, 'restore_original_lighting'): env.restore_original_lighting(); logger.info("💡 Restored original lighting.")
                    
                    images_agent_ph2, images_eye_ph2, t2, done2, info2 = execute_phase2_with_decoupled_e2e_model(env, unified_model, original_prompt, openvla_config.max_steps - failure_step, logger, DEVICE, CPU_DEVICE)
                    images_agent.extend(images_agent_ph2); images_eye.extend(images_eye_ph2)
                    total_steps, final_info, final_done = failure_step + t2, info2, done2
                    logger.info(f"📊 Phase 2 finished: Executed {t2} more steps. Total: {total_steps}.")
                except Exception as e:
                    logger.error(f"❌ Phase 2 execution failed: {e}", exc_info=True)
                    total_steps, final_info, final_done = t, info, done
                finally:
                    logger.info("🧹 Resetting device state for next episode..."); unified_model.vla_model.to(DEVICE); safe_gpu_reset()
            else:
                if mfm_failure_detected and not PHASE2_ENABLED: logger.warning("⚠️ MFM detected failure but Phase 2 is disabled. Skipping E2E execution.")
                total_steps, final_info, final_done = t, info, done
            
            # --- Episode End Processing ---
            if final_info.get('success', False): result = {'success': True, 'steps': total_steps, 'reason': 'success'}
            else: result = {'success': False, 'steps': total_steps, 'reason': 'timeout' if total_steps >= openvla_config.max_steps else 'fail_early'}
            
            result.update({'mfm_failure_detected': mfm_failure_detected, 'original_prompt': original_prompt, 'failure_step': failure_step, 'phase1_steps': failure_step if mfm_failure_detected else total_steps, 'phase2_steps': total_steps - failure_step if mfm_failure_detected else 0, 'used_e2e_model': mfm_failure_detected, 'pkl_file': str(pkl_path) if pkl_path else None, 'mfm_summary': mfm_summary})
            all_results.append(result)
            total_episodes += 1;
            if result['success']: total_successes += 1
            
            mode_suffix = "_MFM_E2E" if mfm_failure_detected else "_MFM_VLA"
            video_filename = f"task{task_id}_ep{ep_idx}_{result['reason'].upper()}{mode_suffix}.mp4"
            save_dual_view_video_with_step_counter(images_agent, images_eye, videos_dir / video_filename, 20, f"[{result['reason'].upper()}] {original_prompt}")
            logger.info(f"📹 Video saved: {video_filename}")
            del images_agent, images_eye; safe_gpu_reset()

    # --- [Final Report] ---
    success_rate = total_successes / total_episodes if total_episodes > 0 else 0
    mfm_e2e_episodes = sum(1 for r in all_results if r.get('mfm_failure_detected', False))
    mfm_e2e_successes = sum(1 for r in all_results if r.get('mfm_failure_detected', False) and r['success'])
    mfm_vla_episodes, mfm_vla_successes = total_episodes - mfm_e2e_episodes, total_successes - mfm_e2e_successes
    mfm_e2e_sr, mfm_vla_sr = (mfm_e2e_successes / mfm_e2e_episodes if mfm_e2e_episodes > 0 else 0), (mfm_vla_successes / mfm_vla_episodes if mfm_vla_episodes > 0 else 0)
    
    final_report = {
        'success_rate': success_rate, 'total_interventions': total_interventions, 'detailed_results': all_results,
        'statistics': {
            'total_episodes': total_episodes, 'mfm_vla_only_episodes': mfm_vla_episodes,
            'mfm_e2e_episodes': mfm_e2e_episodes, 'mfm_vla_success_rate': mfm_vla_sr,
            'mfm_e2e_success_rate': mfm_e2e_sr, 'mfm_intervention_rate': mfm_e2e_episodes / total_episodes if total_episodes > 0 else 0
        }, 'config': {'task_suite': openvla_config.task_suite_name, 'lighting': LIGHTING_CONFIG}
    }
    with open(results_dir / "final_report.json", 'w') as f: json.dump(convert_numpy_types(final_report), f, indent=4)
    logger.info("="*60 + f"\n✅ Evaluation Complete! Overall Success Rate: {success_rate:.2%}")
    logger.info(f"  • MFM+VLA Only Success Rate: {mfm_vla_sr:.2%} ({mfm_vla_successes}/{mfm_vla_episodes})")
    logger.info(f"  • MFM+E2E (Phase 2) Success Rate: {mfm_e2e_sr:.2%} ({mfm_e2e_successes}/{mfm_e2e_episodes})")
    logger.info(f"  • MFM Intervention Rate: {mfm_e2e_episodes/total_episodes*100:.1f}%")
    logger.info(f"📂 Results saved to: {results_dir}\n" + "="*60)
    
    del unified_model, mfm_module; safe_gpu_reset()

if __name__ == "__main__":
    eval_with_phase1_mfm_vla_phase2_e2e()