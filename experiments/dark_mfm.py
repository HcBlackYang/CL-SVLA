# MFM/experiments/dark_mfm.py

import os
import sys
import torch
import numpy as np
import tqdm
import json
from pathlib import Path
from PIL import Image
import gc
import shutil
import cv2  # Ensure cv2 is imported
import pickle  # <<< [NEW FEATURE] >>> Import pickle for serializing simulation state
from typing import Any # <<< [NEW FEATURE] >>> Import Any for type hinting

# --- [Path and Module Imports] ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OPENVLA_SRC_DIR = PROJECT_ROOT / "openvla" / "OpenVLA"
if not OPENVLA_SRC_DIR.is_dir():
    raise FileNotFoundError(f"OpenVLA source directory not found at: {OPENVLA_SRC_DIR}")
sys.path.insert(0, str(OPENVLA_SRC_DIR))
from openvla.OpenVLA.config import Config as OpenVLAConfig
import openvla.OpenVLA.utils as vla_utils
from MFM.config import MFMConfig
from MFM.core import MetacognitiveFeedbackModule
from MFM.utils import setup_logger
from libero.libero import benchmark
from transformers import AutoModelForVision2Seq, AutoProcessor


def save_dual_view_video_with_step_counter(
    agentview_images: list, 
    eye_in_hand_images: list, 
    video_path: Path, 
    fps: int = 20, 
    task_prompt: str = ""
):
    """
    Stitches agent-view and eye-in-hand images side-by-side, adds a step counter and task text,
    and saves the result as a video.
    """
    if not agentview_images or not eye_in_hand_images:
        print("[Warning] No images provided to save video.")
        return

    num_frames = min(len(agentview_images), len(eye_in_hand_images))
    if num_frames == 0:
        return
        
    h, w, _ = agentview_images[0].shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w * 2, h))

    # Font settings for overlay text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255) # White
    line_type = 2
    
    for i in range(num_frames):
        frame_agent = agentview_images[i]
        frame_eye = eye_in_hand_images[i]
        combined_frame = np.concatenate((frame_agent, frame_eye), axis=1)

        # Add step counter text
        step_text = f"Step: {i+1}"
        text_size = cv2.getTextSize(step_text, font, font_scale, line_type)[0]
        text_x = combined_frame.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        
        # Add a dark background for better text visibility
        cv2.rectangle(combined_frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0,0,0), -1)
        cv2.putText(combined_frame, step_text, (text_x, text_y), font, font_scale, font_color, line_type)

        # Add task prompt text
        if task_prompt:
             prompt_text = (task_prompt[:40] + '...') if len(task_prompt) > 40 else task_prompt
             # Add a black outline for the text
             cv2.putText(combined_frame, prompt_text, (10, 20), font, 0.5, (0, 0, 0), 3)
             cv2.putText(combined_frame, prompt_text, (10, 20), font, 0.5, font_color, 1)

        writer.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))

    writer.release()

def convert_numpy_types(data):
    """
    Recursively converts numpy types in a data structure to native Python types for JSON serialization.
    """
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_numpy_types(item) for item in data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, (np.str_, np.unicode_)):
        return str(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

class DarkenedEnvironmentWrapper:
    """A safe environment wrapper to darken ambient lighting and add noise."""
    def __init__(self, env, darkness_factor=0.3, add_noise=False, noise_intensity=0.02):
        self.env = env
        self.darkness_factor = darkness_factor
        self.add_noise = add_noise
        self.noise_intensity = noise_intensity

    def darken_image(self, image: np.ndarray) -> np.ndarray:
        """Applies darkness and optional noise to an image."""
        if not (isinstance(image, np.ndarray) and len(image.shape) == 3):
            return image
        darkened = image.astype(np.float32) * self.darkness_factor
        if self.add_noise:
            noise = np.random.normal(0, self.noise_intensity * 255, image.shape)
            darkened += noise
        return np.clip(darkened, 0, 255).astype(np.uint8)

    def process_observation(self, obs: dict) -> dict:
        """Processes the observation dictionary to darken all image-like values."""
        if obs is None: return None
        processed_obs = {k: self.darken_image(v) if 'image' in k and isinstance(v, np.ndarray) else v for k, v in obs.items()}
        return processed_obs

    def step(self, action):
        """Wrapper for the step function."""
        obs, reward, done, info = self.env.step(action)
        return self.process_observation(obs), reward, done, info

    def reset(self, *args, **kwargs):
        """Wrapper for the reset function."""
        obs = self.env.reset(*args, **kwargs)
        return self.process_observation(obs)

    def set_init_state(self, *args, **kwargs):
        """Wrapper for the set_init_state function."""
        obs = self.env.set_init_state(*args, **kwargs)
        return self.process_observation(obs)

    def __getattr__(self, name):
        """Delegates other attribute access to the original environment."""
        return getattr(self.env, name)

def process_vla_action(raw_action: np.ndarray) -> np.ndarray:
    """Post-processes the raw action output from the VLA model."""
    action = raw_action.copy()
    # Normalize gripper action from [0, 1] to [-1, 1] and flip the sign.
    action[..., -1] = 2 * action[..., -1] - 1
    action[..., -1] = np.sign(action[..., -1])
    action[..., -1] = action[..., -1] * -1.0
    return action

def predict_vla_action(model, processor, prompt: str, image_np: np.ndarray, device: torch.device) -> np.ndarray:
    """Predicts a VLA action given a model, prompt, and image."""
    prompt_for_vla = f"In: {prompt}\nOut:"
    image_pil = Image.fromarray(image_np).convert("RGB")
    inputs = processor(text=prompt_for_vla, images=image_pil, return_tensors="pt")
    inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
    if 'pixel_values' in inputs_on_device:
        inputs_on_device['pixel_values'] = inputs_on_device['pixel_values'].to(dtype=torch.bfloat16)
    with torch.no_grad():
        raw_action = model.predict_action(**inputs_on_device, unnorm_key="libero_spatial", do_sample=True)
    return process_vla_action(raw_action)

# <<< [NEW FEATURE] >>> Helper function to save a simulation snapshot.
def save_simulation_snapshot(env: Any, path: Path, logger):
    """
    Saves the complete state of the current MuJoCo simulation environment.
    
    Args:
        env (Any): The environment instance.
        path (Path): The file path to save the snapshot to.
        logger: The logger instance.
    """
    try:
        # Unwrap the environment in case of wrappers (like our DarkenedEnvironmentWrapper).
        original_env = env
        while hasattr(original_env, 'env'):
            original_env = original_env.env
        
        if hasattr(original_env, 'sim') and hasattr(original_env.sim, 'get_state'):
            sim_state = original_env.sim.get_state()
            with open(path, 'wb') as f:
                pickle.dump(sim_state, f)
            logger.info(f"💾 MuJoCo simulation state successfully saved to {path.name}")
        else:
            logger.warning("Could not save simulation state: `env.sim.get_state()` not found.")
    except Exception as e:
        logger.error(f"Failed to save simulation snapshot: {e}", exc_info=True)


def eval_libero_with_mfm_persistent_feedback():
    """
    Main evaluation function, modified to capture a snapshot and interrupt on failure detection.
    """

    # --- [Configuration and Initialization] ---
    openvla_config = OpenVLAConfig()
    mfm_config = MFMConfig()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("MFM_Failure_Capture", mfm_config.log_level)
    
    # Configuration for environment lighting modifications
    LIGHTING_CONFIG = {
        'enabled': True,
        'darkness_factor':1,
        'add_noise': True,
        'noise_intensity': 0,
    }

    logger.info("🚀 Starting MFM failure capture process")
    logger.info(f"Using device: {DEVICE}")
    if LIGHTING_CONFIG['enabled']:
        logger.warning(f"🎛️ Environment modification is active: Darkness Factor={LIGHTING_CONFIG['darkness_factor']}, Add Noise={LIGHTING_CONFIG['add_noise']}")
    else:
        logger.info("☀️ Using standard, bright environment.")

    # --- [Results Directory Setup] ---
    env_suffix = "_DARKENED" if LIGHTING_CONFIG['enabled'] else ""
    run_id = f"MFM-FAILURE-CAPTURE-{openvla_config.task_suite_name}-{vla_utils.DATE_TIME}{env_suffix}"
    
    results_dir = Path(mfm_config.log_dir) / run_id
    videos_dir = results_dir / "videos"
    state_history_dir = results_dir / "state_history"
    snapshots_dir = results_dir / "snapshots"  # <<< [NEW FEATURE] >>> Directory for failure snapshots
    
    results_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(exist_ok=True)
    state_history_dir.mkdir(exist_ok=True)
    snapshots_dir.mkdir(exist_ok=True) # <<< [NEW FEATURE] >>> Create the directory
    
    logger.info(f"📂 All results will be saved to: {results_dir}")
    logger.info(f"📸 Failure snapshots will be saved to: {snapshots_dir}")

    # --- [One-time Model Loading] ---
    logger.info("📦 Loading VLA model and processor...")
    vla_model = AutoModelForVision2Seq.from_pretrained(
        openvla_config.pretrained_checkpoint, attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(DEVICE)
    vla_processor = AutoProcessor.from_pretrained(openvla_config.pretrained_checkpoint, trust_remote_code=True)
    vla_model.eval()
    logger.info("🧠 Initializing MFM module...")
    mfm = MetacognitiveFeedbackModule(mfm_config)
    logger.info("✅ All modules loaded.")

    # --- [Task Setup] ---
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[openvla_config.task_suite_name]()
    
    # --- [Main Evaluation Loop] ---
    total_episodes, total_successes, total_failures_captured = 0, 0, 0
    all_results = []

    for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc="Evaluating all tasks"):
        task = task_suite.get_task(task_id)
        task_init_states = task_suite.get_task_init_states(task_id)

        original_env, task_description = vla_utils.get_libero_env(
            task=task,
            model_family=openvla_config.model_family,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=[512, 512], 
            camera_widths=[512, 512]
        )

        if LIGHTING_CONFIG['enabled']:
            env = DarkenedEnvironmentWrapper(original_env, **{k:v for k,v in LIGHTING_CONFIG.items() if k != 'enabled'})
        else:
            env = original_env

        initial_task_prompt = task_description.lower()
        
        for episode_idx in tqdm.tqdm(range(openvla_config.num_trials_per_task), desc=f"Task {task_id} Trials", leave=False):
            # --- Episode Initialization ---
            logger.info("=" * 60)
            logger.info(f"🎬 Starting new Episode: Task {task_id}, Trial {episode_idx+1}/{openvla_config.num_trials_per_task}")
            logger.info(f"   - Task Goal: {initial_task_prompt}")
            logger.info("=" * 60)

            env.reset()
            obs = env.set_init_state(task_init_states[episode_idx])
            
            replay_agentview_images, replay_eye_in_hand_images = [], []
            episode_interventions, is_successful = 0, False
            current_prompt = initial_task_prompt
            
            mfm.start_monitoring(initial_task_prompt)

            # --- Step Loop ---
            t = 0
            done = False
            while t < openvla_config.max_steps and not done:
                # Collect images for replay video
                agentview_img_raw = obs["agentview_image"][::-1, ::-1].copy()
                eye_in_hand_img_raw = obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
                replay_agentview_images.append(agentview_img_raw)
                replay_eye_in_hand_images.append(eye_in_hand_img_raw)

                prompt_image_for_vla = vla_utils.resize_image(agentview_img_raw, (224, 224))
                prompt_display = (current_prompt[:75] + '...') if len(current_prompt) > 75 else current_prompt
                logger.debug(f"[Step {t+1}/{openvla_config.max_steps}] 🤖 VLA-Prompt: '{prompt_display}'")

                action = predict_vla_action(vla_model, vla_processor, current_prompt, prompt_image_for_vla, DEVICE)
                obs, reward, done, info = env.step(action)
                t += 1

                # Process the step with MFM
                should_intervene, _ = mfm.process_execution_step(
                    observation=obs, action=action, reward=reward, done=done, info=info, env=env
                )
                
                if info.get('success', False):
                    is_successful = True
                    logger.info(f"🏆 SUCCESS! Episode {episode_idx} completed successfully at step {t}!")
                    break

                # <<< [LOGIC MODIFICATION] >>> Replace MFM intervention with failure capture and interruption.
                if should_intervene:
                    total_failures_captured += 1
                    logger.warning("=" * 20 + " FAILURE DETECTED! CAPTURING SNAPSHOT. " + "=" * 20)
                    failure_type = mfm.execution_context.last_failure_type or "UnknownFailure"
                    logger.warning(f"   - Detected Failure Type: {failure_type}")
                    logger.warning(f"   - At Step: {t}")
                    
                    base_snapshot_name = f"task_{task_id}_episode_{episode_idx}_step_{t}_{failure_type}"

                    # 1. Save simulation state snapshot (.pkl)
                    snapshot_path = snapshots_dir / f"{base_snapshot_name}.pkl"
                    save_simulation_snapshot(original_env, snapshot_path, logger)

                    # 2. Save video from the start of the episode until the failure.
                    video_path = videos_dir / f"{base_snapshot_name}_FAIL.mp4"
                    save_dual_view_video_with_step_counter(
                        agentview_images=replay_agentview_images,
                        eye_in_hand_images=replay_eye_in_hand_images,
                        video_path=video_path,
                        fps=20,
                        task_prompt=f"[FAIL] {initial_task_prompt}"
                    )
                    logger.info(f"📹 Failure video saved to: {video_path.name}")
                    
                    # 3. Save state log from the start of the episode until the failure.
                    state_history_path = state_history_dir / f"{base_snapshot_name}_FAIL.json"
                    try:
                        state_data = mfm.state_collector.export_collected_data(include_detailed=True)
                        state_data_serializable = convert_numpy_types(state_data)
                        with open(state_history_path, 'w') as f:
                            json.dump(state_data_serializable, f, indent=2)
                        logger.info(f"📜 Failure state history saved to: {state_history_path.name}")
                    except Exception as e:
                        logger.error(f"Error saving failure state history: {e}", exc_info=True)

                    # 4. Record the result and immediately interrupt the current episode.
                    logger.info("🛑 Task interrupted due to detected failure. Moving to the next trial.")
                    is_successful = False
                    total_episodes += 1
                    episode_result = {
                        'task_id': task_id, 
                        'episode_idx': episode_idx, 
                        'success': False, 
                        'steps_before_fail': t, 
                        'failure_type': failure_type
                    }
                    all_results.append(episode_result)
                    
                    break # <--- Immediately break out of the current task loop.

            # --- End of Episode Processing ---
            # <<< [LOGIC MODIFICATION] >>> This block only executes if the loop finishes normally (i.e., not interrupted by MFM's break).
            if not mfm.intervention_triggered:
                mfm.stop_monitoring()
                if not is_successful: logger.error(f"❌ TIMEOUT! Episode {episode_idx} ended after {t} steps.")
                if is_successful: total_successes += 1
                total_episodes += 1
                
                episode_result = {
                    'task_id': task_id, 
                    'episode_idx': episode_idx, 
                    'success': is_successful, 
                    'steps': t, 
                    'interventions': 0, # Intervention count is meaningless in this mode.
                    'final_prompt': current_prompt
                }
                all_results.append(episode_result)
                
                base_filename = f"task_{task_id}_episode_{episode_idx}_{'SUCCESS' if is_successful else 'TIMEOUT'}"

                # Save state history for successful/timeout runs.
                try:
                    state_data = mfm.state_collector.export_collected_data(include_detailed=True)
                    state_filepath = state_history_dir / f"{base_filename}_state_history.json"
                    state_data_serializable = convert_numpy_types(state_data)
                    with open(state_filepath, 'w') as f:
                        json.dump(state_data_serializable, f, indent=2)
                except Exception as e:
                    logger.error(f"Error saving state history: {e}", exc_info=True)

                # Save video for successful/timeout runs.
                try:
                    target_video_path = videos_dir / f"{base_filename}.mp4"
                    save_dual_view_video_with_step_counter(
                        agentview_images=replay_agentview_images,
                        eye_in_hand_images=replay_eye_in_hand_images,
                        video_path=target_video_path,
                        fps=20,
                        task_prompt=initial_task_prompt
                    )
                except Exception as e:
                    logger.error(f"Error saving video: {e}", exc_info=True)

    # --- [Final Report] ---
    overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0
    
    final_report = {
        'experiment_config': {'run_id': run_id, 'task_suite': openvla_config.task_suite_name, 'mode': 'failure_capture', 'lighting_conditions': LIGHTING_CONFIG},
        'overall_summary': {
            'total_episodes_run': total_episodes, 
            'total_successes': total_successes, 
            'overall_success_rate': overall_success_rate, 
            'total_failures_captured': total_failures_captured
        },
        'detailed_results': all_results
    }
    
    final_results_path = results_dir / "final_report.json"
    with open(final_results_path, 'w') as f: json.dump(final_report, f, indent=4)
        
    logger.info("="*60)
    logger.info("✅ MFM failure capture complete!")
    logger.info(f"   - Overall Success Rate: {overall_success_rate:.2%}")
    logger.info(f"   - Successfully captured failure scenarios: {total_failures_captured}")
    logger.info(f"   - Detailed report saved to: {final_results_path}")
    logger.info("="*60)

    # Clean up resources
    del vla_model, vla_processor, mfm
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    eval_libero_with_mfm_persistent_feedback()