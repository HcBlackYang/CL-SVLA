# MFM/experiments/generate_expert_data.py
# Batch processing version - can handle a single file or an entire directory, with multi-task support.
# Example usage:
# python experiments/generate_expert_data.py --input-path /root/autodl-tmp/openvla/MFM/data/logs/task_1/snapshots --task 1

import os
import sys
import torch
import numpy as np
import pickle
import json
import cv2
import argparse
from pathlib import Path
from PIL import Image
import glob
from tqdm import tqdm

# --- Path Setup and Imports ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OPENVLA_SRC_DIR = PROJECT_ROOT / "openvla" / "OpenVLA"
if not OPENVLA_SRC_DIR.is_dir():
    raise FileNotFoundError(f"OpenVLA source directory not found at: {OPENVLA_SRC_DIR}")
sys.path.insert(0, str(OPENVLA_SRC_DIR))
from openvla.OpenVLA.config import Config as OpenVLAConfig
import openvla.OpenVLA.utils as vla_utils
from libero.libero import benchmark
from dark_mfm import save_dual_view_video_with_step_counter, convert_numpy_types

# Task-specific coordinate configurations for expert trajectories
TASK_COORDINATES = {
    1: { 'start_pos': np.array([-0.012, 0.058, 1.013]), 'bowl_approach_pos': np.array([-0.068, 0.054, 0.935]), 'bowl_pos': np.array([-0.075, 0.009, 0.899]), 'lift_pos': np.array([-0.078, 0.067, 0.995]), 'transport_pos': np.array([0.079, 0.213, 1.066]), 'place_approach_pos': np.array([0.083, 0.240, 0.946]), 'plate_pos': np.array([0.071, 0.200, 0.902]), 'bowl_offset': np.array([0.022, -0.047, 0.04]), 'plate_offset': np.array([-0.01, -0.047, 0.05]) },
    2: { 'start_pos': np.array([0.074, 0.062, 1.014]), 'bowl_approach_pos': np.array([0.055, 0.060, 0.947]), 'bowl_pos': np.array([0.059, 0.016, 0.918]), 'lift_pos': np.array([0.054, 0.080, 1.025]), 'transport_pos': np.array([0.079, 0.219, 1.025]), 'place_approach_pos': np.array([0.070, 0.236, 0.936]), 'plate_pos': np.array([0.069, 0.190, 0.902]), 'bowl_offset': np.array([0.022, -0.047, 0.04]), 'plate_offset': np.array([-0.01, -0.047, 0.05]) },
    3: { 'start_pos': np.array([0.036, -0.104, 1.127]), 'bowl_approach_pos': np.array([0.045, -0.092, 1.106]), 'bowl_pos': np.array([0.072, -0.132, 1.071]), 'lift_pos': np.array([0.066, -0.057, 1.212]), 'transport_pos': np.array([0.081, 0.187, 1.169]), 'place_approach_pos': np.array([0.046, 0.252, 0.956]), 'plate_pos': np.array([0.055, 0.209, 0.902]), 'bowl_offset': np.array([0.022, -0.047, 0.04]), 'plate_offset': np.array([-0.01, -0.047, 0.05]) },
    4: { 'start_pos': np.array([-0.147, 0.148, 1.070]), 'bowl_approach_pos': np.array([-0.219, 0.173, 0.967]), 'bowl_pos': np.array([-0.209, 0.215, 0.940]), 'lift_pos': np.array([-0.199, 0.165, 1.041]), 'transport_pos': np.array([0.102, 0.142, 1.056]), 'place_approach_pos': np.array([0.048, 0.148, 0.934]), 'plate_pos': np.array([0.073, 0.185, 0.903]), 'bowl_offset': np.array([0.022, -0.047, 0.04]), 'plate_offset': np.array([-0.01, -0.047, 0.05]) },
    5: { 'start_pos': np.array([0.134, -0.024, 1.040]), 'bowl_approach_pos': np.array([0.125, -0.037, 0.937]), 'bowl_pos': np.array([0.137, -0.081, 0.898]), 'lift_pos': np.array([0.122, -0.005, 1.064]), 'transport_pos': np.array([0.094, 0.231, 1.046]), 'place_approach_pos': np.array([0.091, 0.244, 0.949]), 'plate_pos': np.array([0.069, 0.194, 0.904]), 'bowl_offset': np.array([0.022, -0.047, 0.04]), 'plate_offset': np.array([-0.01, -0.047, 0.05]) },
    6: { 'start_pos': np.array([-0.225, -0.031, 1.050]), 'bowl_approach_pos': np.array([-0.259, -0.087, 0.947]), 'bowl_pos': np.array([-0.263, -0.131, 0.929]), 'lift_pos': np.array([-0.198, -0.043, 1.047]), 'transport_pos': np.array([0.045, 0.225, 1.055]), 'place_approach_pos': np.array([0.054, 0.239, 0.945]), 'plate_pos': np.array([0.059, 0.204, 0.902]), 'bowl_offset': np.array([0.022, -0.047, 0.04]), 'plate_offset': np.array([-0.01, -0.047, 0.05]) },
    7: { 'start_pos': np.array([0.088, -0.188, 1.254]), 'bowl_approach_pos': np.array([0.021, -0.204, 1.148]), 'bowl_pos': np.array([0.041, -0.259, 1.133]), 'lift_pos': np.array([0.000, -0.087, 1.270]), 'transport_pos': np.array([0.024, 0.181, 1.123]), 'place_approach_pos': np.array([0.042, 0.239, 0.938]), 'plate_pos': np.array([0.046, 0.196, 0.902]), 'bowl_offset': np.array([0.022, -0.047, 0.04]), 'plate_offset': np.array([-0.01, -0.047, 0.05]) },
    8: { 'start_pos': np.array([-0.131, 0.332, 1.097]), 'bowl_approach_pos': np.array([-0.189, 0.366, 0.918]), 'bowl_pos': np.array([-0.195, 0.315, 0.903]), 'lift_pos': np.array([-0.149, 0.355, 1.042]), 'transport_pos': np.array([0.048, 0.229, 1.094]), 'place_approach_pos': np.array([0.094, 0.218, 0.940]), 'plate_pos': np.array([0.069, 0.187, 0.902]), 'bowl_offset': np.array([0.022, -0.047, 0.04]), 'plate_offset': np.array([-0.01, -0.047, 0.05]) },
    9: { 'start_pos': np.array([-0.026, 0.139, 1.036]), 'bowl_approach_pos': np.array([-0.057, 0.159, 0.907]), 'bowl_pos': np.array([-0.057, 0.198, 0.903]), 'lift_pos': np.array([-0.009, 0.155, 1.039]), 'transport_pos': np.array([0.039, 0.153, 1.034]), 'place_approach_pos': np.array([0.045, 0.158, 0.954]), 'plate_pos': np.array([0.054, 0.204, 0.902]), 'bowl_offset': np.array([0.022, -0.047, 0.02]), 'plate_offset': np.array([-0.01, -0.047, 0.05]) },
    10: { 'use_dynamic_coords': True, 'bowl_offset': np.array([0.022, -0.047, 0.04]), 'plate_offset': np.array([-0.01, -0.047, 0.05]) }
}

def _obs_to_json_safe(obs_dict: dict) -> dict:
    """Creates a new dictionary without image data for JSON serialization."""
    return {key: value for key, value in obs_dict.items() if 'image' not in key}

def interpolate_waypoints(waypoints: list, points_per_segment: int) -> list:
    """Evenly inserts points between anchor waypoints."""
    interpolated_path = []
    if len(waypoints) < 2:
        return waypoints

    for i in range(len(waypoints) - 1):
        start_point, end_point = waypoints[i], waypoints[i+1]
        interpolated_path.append(start_point)
        
        for j in range(1, points_per_segment):
            alpha = j / float(points_per_segment)
            interp_point = {}
            interp_point['pos'] = (1 - alpha) * start_point['pos'] + alpha * end_point['pos']
            interp_quat = (1 - alpha) * start_point['quat'] + alpha * end_point['quat']
            interp_point['quat'] = interp_quat / np.linalg.norm(interp_quat)
            interp_point['gripper'] = start_point['gripper'] # Gripper state is held constant between anchors
            interpolated_path.append(interp_point)
            
    interpolated_path.append(waypoints[-1])
    return interpolated_path

class ExpertController:
    """
    A simple PD-based expert controller. This version advances to the next
    waypoint on each `get_action` call, ensuring a 1-to-1 mapping between
    interpolated points and environment steps.
    """
    def __init__(self, waypoints: list):
        self.waypoints = waypoints
        self.waypoint_idx = 0
        self.p_gain = 10.0
        self.d_gain = 0.8
        self.last_pos_error = np.zeros(3)

    def is_finished(self) -> bool:
        """Checks if all waypoints have been processed."""
        return self.waypoint_idx >= len(self.waypoints)

    def get_action(self, current_eef_pos: np.ndarray, current_eef_quat: np.ndarray) -> np.ndarray:
        """
        Calculates an action to move towards the current target waypoint and then
        advances the target to the next waypoint.
        """
        if self.is_finished():
            return np.zeros(7) # No-op action

        target_point = self.waypoints[self.waypoint_idx]
        target_pos, target_gripper = target_point['pos'], target_point['gripper']

        pos_error = target_pos - current_eef_pos
        pos_velocity = np.clip(self.p_gain * pos_error + self.d_gain * (pos_error - self.last_pos_error), -1.0, 1.0)
        self.last_pos_error = pos_error

        rot_velocity = np.zeros(3) # No rotation in this expert
        gripper_action = 1.0 if target_gripper > 0.5 else -1.0
        
        # print(f"Tracking waypoint {self.waypoint_idx + 1}/{len(self.waypoints)}")
        self.waypoint_idx += 1 # Advance to the next waypoint

        action = np.concatenate([pos_velocity, rot_velocity, [gripper_action]])
        return np.clip(action, -1.0, 1.0)

def get_expert_anchor_points(task_id: int, obs: dict, current_eef_quat: np.ndarray) -> list:
    """
    Generates expert anchor waypoints based on the task ID.
    
    Args:
        task_id (int): The ID of the task.
        obs (dict): The current observation dictionary from the environment.
        current_eef_quat (np.ndarray): The current end-effector quaternion.
    
    Returns:
        list: A list of anchor point dictionaries.
    """
    if task_id not in TASK_COORDINATES:
        raise ValueError(f"Unsupported task ID: {task_id}. Supported tasks: {list(TASK_COORDINATES.keys())}")
    
    task_config = TASK_COORDINATES[task_id]
    
    if task_id == 10 or task_config.get('use_dynamic_coords', False):
        # For Task 10, use the original dynamic coordinate acquisition logic
        try:
            bowl_pos_at_failure = obs['akita_black_bowl_1_pos']
            plate_pos_at_failure = obs['plate_1_pos']
            print(f"   - Bowl position at failure: {np.round(bowl_pos_at_failure, 3)}")
            print(f"   - Plate position at failure: {np.round(plate_pos_at_failure, 3)}")
        except KeyError as e:
            print(f"Fatal Error: Could not find object position '{e}' in observation. Aborting.")
            return None

        target_2_pos = bowl_pos_at_failure + task_config['bowl_offset']
        target_5_pos = plate_pos_at_failure + task_config['plate_offset']
        
        expert_anchor_points = [
            {'pos': np.array([-0.029, 0.170, 1.171]), 'quat': current_eef_quat, 'gripper': 0},
            {'pos': target_2_pos, 'quat': current_eef_quat, 'gripper': 0},
            {'pos': target_2_pos, 'quat': current_eef_quat, 'gripper': 1},
            {'pos': np.array([-0.014, 0.265, 0.948]), 'quat': current_eef_quat, 'gripper': 1},
            {'pos': np.array([0.032, 0.226, 1.064]), 'quat': current_eef_quat, 'gripper': 1},
            {'pos': target_5_pos, 'quat': current_eef_quat, 'gripper': 1},
            {'pos': target_5_pos, 'quat': current_eef_quat, 'gripper': 0},
        ]
        
        # Add final retraction anchor point
        retract_pos = np.array([0.032, 0.226, 1.064])
        final_anchor_point = {'pos': retract_pos, 'quat': current_eef_quat, 'gripper': 0}
        expert_anchor_points.append(final_anchor_point)
        
    else:
        # For other tasks, use the pre-defined coordinates
        print(f"   - Using pre-defined coordinate configuration for Task {task_id}")
        
        # Calculate target positions (based on pre-defined coords + offset)
        target_2_pos = task_config['bowl_pos'] + task_config['bowl_offset']
        target_5_pos = task_config['plate_pos'] + task_config['plate_offset']
        
        print(f"   - Bowl position: {np.round(task_config['bowl_pos'], 3)}")
        print(f"   - Plate position: {np.round(task_config['plate_pos'], 3)}")
        print(f"   - Adjusted grasp target: {np.round(target_2_pos, 3)}")
        print(f"   - Adjusted place target: {np.round(target_5_pos, 3)}")
        
        expert_anchor_points = [
            {'pos': task_config['start_pos'], 'quat': current_eef_quat, 'gripper': 0},        # Start position
            {'pos': target_2_pos, 'quat': current_eef_quat, 'gripper': 0},                   # Approach bowl
            {'pos': target_2_pos, 'quat': current_eef_quat, 'gripper': 1},                   # Grasp bowl
            {'pos': task_config['lift_pos'], 'quat': current_eef_quat, 'gripper': 1},        # Lift
            {'pos': task_config['transport_pos'], 'quat': current_eef_quat, 'gripper': 1},   # Mid-point
            {'pos': target_5_pos, 'quat': current_eef_quat, 'gripper': 1},                   # Approach placement
            {'pos': target_5_pos, 'quat': current_eef_quat, 'gripper': 0},                   # Place
        ]
        
        # Add final retraction anchor point (using the transport position)
        final_anchor_point = {'pos': task_config['transport_pos'], 'quat': current_eef_quat, 'gripper': 0}
        expert_anchor_points.append(final_anchor_point)
    
    return expert_anchor_points

def generate_expert_trajectory_single(snapshot_path: Path, task_id: int = 10, progress_callback=None):
    """
    Processes a single snapshot file to generate expert trajectory data.
    
    Args:
        snapshot_path (Path): Path to the .pkl snapshot file.
        task_id (int): The ID of the task to generate data for.
        progress_callback (Callable, optional): A callback to report progress.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    # If task_id is not provided, try to parse it from the filename
    if task_id is None:
        try:
            task_id = int(snapshot_path.stem.split('_')[1])
        except (IndexError, ValueError) as e:
            print(f"⚠️ Warning: Could not parse task_id from filename: {snapshot_path.name}. Error: {e}")
            return False

    # Setup output directories
    expert_data_dir = snapshot_path.parent.parent / "expert_data"
    expert_videos_dir = expert_data_dir / "videos"
    expert_data_dir.mkdir(exist_ok=True)
    expert_videos_dir.mkdir(exist_ok=True)
    
    output_basename = snapshot_path.stem
    expert_json_path = expert_data_dir / f"{output_basename}_EXPERT.json"
    expert_video_path = expert_videos_dir / f"{output_basename}_EXPERT.mp4"

    # Skip if the output files already exist
    if expert_json_path.exists() and expert_video_path.exists():
        print(f"⏭️ Skipping already existing file: {output_basename}")
        return True

    print(f"🚀 Starting expert data generation - Task {task_id}")
    print(f"   - Loading snapshot: {snapshot_path.name}")

    try:
        print("☀️ Loading clean environment (brightness=1, noise=0).")
        openvla_config = OpenVLAConfig()
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[openvla_config.task_suite_name]()
        task = task_suite.get_task(task_id)

        env, task_description = vla_utils.get_libero_env(
            task=task, model_family=openvla_config.model_family,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=[512, 512], camera_widths=[512, 512]
        )

        # Load and set the simulation state from the snapshot
        with open(snapshot_path, 'rb') as f:
            sim_state = pickle.load(f)

        env.reset()
        env.sim.set_state(sim_state)
        env.sim.forward()
        obs, _, _, _ = env.step(np.zeros(7)) # Take a null step to get the observation
        print("✅ Environment restored to the failure state.")

        print(f"🛠️ Defining expert anchor points for Task {task_id}...")
        current_eef_quat = obs['robot0_eef_quat']
        
        expert_anchor_points = get_expert_anchor_points(task_id, obs, current_eef_quat)
        if expert_anchor_points is None:
            env.close()
            return False

        # Interpolate waypoints and create the controller
        waypoints = interpolate_waypoints(expert_anchor_points, points_per_segment=25)
        controller = ExpertController(waypoints)
        print(f"✅ Expert path defined with {len(waypoints)} interpolated waypoints (including final retraction).")

        expert_trajectory_data = []
        replay_agentview_images, replay_eye_in_hand_images = [], []
        max_expert_steps = 600

        # Run the expert controller in the environment
        for step in range(max_expert_steps):
            if controller.is_finished():
                print("🎉 Expert controller completed its path, including final retraction.")
                break

            agentview_img = obs["agentview_image"][::-1, ::-1].copy()
            eye_in_hand_img = obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
            replay_agentview_images.append(agentview_img)
            replay_eye_in_hand_images.append(eye_in_hand_img)
            
            action = controller.get_action(obs['robot0_eef_pos'], obs['robot0_eef_quat'])
            
            obs_for_json = _obs_to_json_safe(obs)
            
            step_data = {
                'step': step,
                'observation': convert_numpy_types(obs_for_json),
                'expert_action': convert_numpy_types(action)
            }
            expert_trajectory_data.append(step_data)
            
            obs, reward, done, info = env.step(action)
                
        if not controller.is_finished():
            print(f"⚠️ Reached max steps ({max_expert_steps}). Expert trajectory may be incomplete.")

        # Save the generated expert data to a JSON file
        print("💾 Saving JSON file...")
        final_data = {
            'metadata': {
                'source_snapshot': snapshot_path.name,
                'task_id': task_id,
                'task_description': task_description,
                'generation_logic': f'Task {task_id} pre-defined trajectory with interpolation and final retraction'
            },
            'expert_trajectory': expert_trajectory_data
        }
        with open(expert_json_path, 'w') as f:
            json.dump(final_data, f, indent=2)
        print(f"✅ Expert JSON data saved to: {expert_json_path}")
        
        # Save the corresponding video
        print("📹 Encoding video...")
        save_dual_view_video_with_step_counter(
            agentview_images=replay_agentview_images,
            eye_in_hand_images=replay_eye_in_hand_images,
            video_path=expert_video_path,
            task_prompt=f"[EXPERT] Task {task_id}: {task_description}"
        )
        print(f"✅ Expert video saved to: {expert_video_path}")
        
        print(f"✅ Task {task_id} expert data generation complete.")
        env.close()
        
        if progress_callback:
            progress_callback()
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing file {snapshot_path.name}: {str(e)}")
        try:
            env.close()
        except:
            pass
        return False

def find_pkl_files(input_path: Path, task_id: int = None) -> list:
    """Finds all .pkl files in the specified path."""
    pkl_files = []
    
    if input_path.is_file():
        if input_path.suffix == '.pkl':
            pkl_files.append(input_path)
        else:
            print(f"⚠️ Warning: Specified file is not a .pkl file: {input_path}")
    elif input_path.is_dir():
        # Find all .pkl files in the directory and subdirectories
        pkl_files.extend(input_path.glob("*.pkl"))
        pkl_files.extend(input_path.glob("**/*.pkl"))
        
        # Filter to ensure we only process failure snapshots
        pkl_files = [f for f in pkl_files if "GraspFailure" in f.name]
        
    return sorted(pkl_files)

def batch_generate_expert_data(input_path: Path, task_id: int = None):
    """Batch-processes .pkl files to generate expert data."""
    print(f"🔍 Finding .pkl files in path: {input_path}")
    if task_id is not None:
        print(f"   - Using specified Task {task_id}")
    
    pkl_files = find_pkl_files(input_path, task_id)
    
    if not pkl_files:
        print("❌ No .pkl files found!")
        return
    
    print(f"📁 Found {len(pkl_files)} .pkl files:")
    for i, pkl_file in enumerate(pkl_files[:5]):
        print(f"   {i+1}. {pkl_file.name}")
    if len(pkl_files) > 5:
        print(f"   ... and {len(pkl_files)-5} more files")
    
    print(f"\n🚀 Starting batch processing...")
    
    success_count = 0
    error_count = 0
    
    with tqdm(total=len(pkl_files), desc="Processing files", unit="file") as pbar:
        def update_progress():
            pbar.update(1)
        
        for pkl_file in pkl_files:
            print(f"\n{'='*60}")
            print(f"Processing: {pkl_file.name}")
            print(f"{'='*60}")
            
            try:
                # Use specified task_id, or parse from filename as a fallback
                file_task_id = task_id
                if file_task_id is None:
                    try:
                        file_task_id = int(pkl_file.stem.split('_')[1])
                    except (IndexError, ValueError):
                        print(f"⚠️ Cannot parse task_id from filename, and --task not specified. Skipping: {pkl_file.name}")
                        error_count += 1
                        pbar.update(1)
                        continue
                
                print(f"   - Using Task ID: {file_task_id}")
                success = generate_expert_trajectory_single(pkl_file, file_task_id, update_progress)
                if success:
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"❌ An unexpected error occurred while processing the file: {str(e)}")
                error_count += 1
                pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"🎉 Batch processing complete!")
    print(f"✅ Successfully processed: {success_count} files")
    print(f"❌ Failed to process: {error_count} files")
    print(f"📊 Total files: {len(pkl_files)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate expert demonstration data - supports single file or batch processing, and multi-task configurations.")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Input path: can be a single .pkl file or a directory containing .pkl files."
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=list(TASK_COORDINATES.keys()),
        help="Specify the task ID (1-10). Strongly recommended for batch mode as .pkl filenames are not task-specific."
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Force processing as a single file (even if the input is a directory)."
    )
    
    args = parser.parse_args()
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"❌ Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    if args.task is not None and args.task not in TASK_COORDINATES:
        print(f"❌ Error: Unsupported task ID {args.task}. Supported tasks: {list(TASK_COORDINATES.keys())}")
        sys.exit(1)
    
    if args.single_file or input_path.is_file():
        # Single file mode
        print("🔄 Single file processing mode")
        if args.task is not None:
            print(f"   - Using specified Task ID: {args.task}")
        success = generate_expert_trajectory_single(input_path, args.task)
        if not success:
            sys.exit(1)
    else:
        # Batch processing mode
        print("🔄 Batch processing mode")
        if args.task is None:
            print("⚠️ Warning: --task parameter was not specified in batch mode.")
            print("   It's recommended to specify --task as .pkl filenames are not task-specific.")
            print("   Will attempt to parse task_id from filenames (may fail).")
        batch_generate_expert_data(input_path, args.task)