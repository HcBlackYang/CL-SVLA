# MFM/scripts/improved_prepare_dataset.py
# This script prepares a multi-task expert dataset (Tasks 1-10),
# generating CoT instructions as strings and including dual-view images.
# Example usage:
# python MFM/scripts/improved_prepare_dataset.py --expert-data-dirs /path/to/task1/expert_data /path/to/task2/expert_data --output-dir data/hdf5_datasets --output-filename multi_task_dataset.hdf5

import h5py
import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
import re

def extract_task_id_from_path(file_path: Path) -> int:
    """
    Extracts the task_id from the file path.
    
    Args:
        file_path (Path): The path to the JSON or PKL file.
        
    Returns:
        int: The extracted task ID, or -1 if not found.
    """
    path_str = str(file_path)
    
    # Method 1: Look for "task<number>" pattern in the parent directory path
    task_match = re.search(r'task(\d+)', path_str.lower())
    if task_match:
        task_id = int(task_match.group(1))
        if 1 <= task_id <= 10:
            return task_id
    
    # Method 2: Look for "task_<number>" pattern in the filename as a fallback
    filename = file_path.name
    task_match = re.search(r'task_?(\d+)', filename.lower())
    if task_match:
        task_id = int(task_match.group(1))
        if 1 <= task_id <= 10:
            return task_id
    
    print(f"⚠️ Could not extract task_id from path: {file_path}")
    return -1

def parse_task_objects(task_description):
    """
    Extracts key objects and actions from a task description string.
    """
    task_lower = task_description.lower()
    objects = {'source_object': None, 'target_object': None, 'action_verb': None}
    
    # Common pick-and-place task pattern
    if 'pick up' in task_lower and 'place' in task_lower:
        objects['action_verb'] = 'pick up and place'
        # Extract the source object (the one to be picked up)
        pick_match = re.search(r'pick up (?:the )?([^,]+?)(?:\s+(?:next to|on|near)|\s+and)', task_lower)
        if pick_match: objects['source_object'] = pick_match.group(1).strip()
        # Extract the target object (the placement location)
        place_match = re.search(r'place it (?:on|in) (?:the )?([^.]+)', task_lower)
        if place_match: objects['target_object'] = place_match.group(1).strip()
    
    # Fallback to generic object names if specific ones are not found
    if not objects['source_object']: objects['source_object'] = 'bowl' if 'bowl' in task_lower else 'object'
    if not objects['target_object']: objects['target_object'] = 'plate' if 'plate' in task_lower else 'target location'
    
    return objects

def create_failure_context_prompt(failure_info, step_progress, task_description):
    """
    Generates a fixed-template CoT instruction to ensure consistency.
    """
    task_objects = parse_task_objects(task_description)
    source_obj, target_obj = task_objects['source_object'], task_objects['target_object']
    
    # 🔥 Fixed template: concise and informative
    template = f"The correct action would be to open the gripper, pick up the {source_obj}, place it on the {target_obj}, then open the gripper, and then raise up."
    
    return template

def extract_failure_info_from_filename(snapshot_path):
    """Quickly extracts failure information from the snapshot filename."""
    filename = snapshot_path.name
    if "GraspFailure" in filename: return {'failure_type': "Grasp Failure"}
    if "PlacementFailure" in filename: return {'failure_type': "Placement Failure"}
    return {'failure_type': "Execution Failure"}

def process_enhanced_trajectory(json_path: Path, hdf5_file: h5py.File, trajectory_idx: int, task_id: int):
    """
    Processes trajectory data, generating actionable CoT instructions and including dual-view images.
    
    Args:
        json_path (Path): Path to the expert trajectory JSON file.
        hdf5_file (h5py.File): The HDF5 file object to write to.
        trajectory_idx (int): The index for this trajectory in the HDF5 file.
        task_id (int): The task ID for this trajectory.
    
    Returns:
        bool: True if processing was successful, False otherwise.
    """
    video_path = json_path.parent / "videos" / f"{json_path.stem}.mp4"
    if not video_path.exists():
        print(f"  [Warning] Video file not found: {video_path.name}")
        return False
    
    # 1. Load JSON data
    with open(json_path, 'r') as f: data = json.load(f)
    expert_trajectory = data.get('expert_trajectory', [])
    if not expert_trajectory:
        print(f"  [Warning] No trajectory data in {json_path.name}")
        return False
    
    # 2. Extract failure info from the corresponding snapshot filename
    snapshot_name = json_path.stem.replace('_EXPERT', '.pkl')
    snapshot_path = json_path.parent.parent / "snapshots" / snapshot_name
    failure_info = extract_failure_info_from_filename(snapshot_path)
    
    # 3. Process the video to extract frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [Error] Cannot open video: {video_path.name}")
        return False
    
    num_steps = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(expert_trajectory))
    expert_trajectory = expert_trajectory[:num_steps]
    
    # 4. Prepare data containers (including for the second view)
    actions, qpos, eef_pos, gripper_qpos = [], [], [], []
    agentview_images, robot_eye_images = [], [] # Agent-view (3rd person), Robot-eye (1st person)
    failure_context_prompts = []
    
    task_description = data['metadata']['task_description']
    
    # 5. Process each step to generate CoT instructions and collect dual-view images
    for i in range(num_steps):
        ret, frame = cap.read()
        if not ret: break
        
        # Split the combined video frame into two views
        h, w, _ = frame.shape; mid_point = w // 2
        frame_agent = cv2.cvtColor(frame[:, :mid_point, :], cv2.COLOR_BGR2RGB)
        frame_robot = cv2.cvtColor(frame[:, mid_point:, :], cv2.COLOR_BGR2RGB)
        
        step_data, obs = expert_trajectory[i], expert_trajectory[i]['observation']
        
        # 🔥 Generate the consistent CoT instruction
        context_prompt = create_failure_context_prompt(failure_info, i / max(1, num_steps - 1), task_description)
        
        # Collect data for this step (including both views)
        actions.append(step_data['expert_action'])
        qpos.append(obs['robot0_joint_pos']); eef_pos.append(obs['robot0_eef_pos']); gripper_qpos.append(obs['robot0_gripper_qpos'])
        agentview_images.append(frame_agent); robot_eye_images.append(frame_robot)
        failure_context_prompts.append(context_prompt)
    
    cap.release()
    
    # 6. Save the collected data to the HDF5 file
    if not actions:
        print(f"  [Warning] No data extracted from {json_path.name}")
        return False
    
    traj_group = hdf5_file.create_group(f'trajectory_{trajectory_idx}')
    
    # Metadata
    traj_group.attrs['task_description'] = task_description
    traj_group.attrs['task_id'] = task_id # 🔥 Store task_id
    traj_group.attrs['data_type'] = 'multi_task_cot_dual_view' # 🔥 Update type identifier
    
    # Observation and Action Data
    traj_group.create_dataset('actions', data=np.array(actions, dtype=np.float32))
    obs_group = traj_group.create_group('observations')
    obs_group.create_dataset('qpos', data=np.array(qpos, dtype=np.float32))
    obs_group.create_dataset('eef_pos', data=np.array(eef_pos, dtype=np.float32))
    obs_group.create_dataset('gripper_qpos', data=np.array(gripper_qpos, dtype=np.float32))
    
    # Dual-view Image Data
    img_group = traj_group.create_group('images')
    img_group.create_dataset('agentview', data=np.array(agentview_images, dtype=np.uint8), chunks=(1, 512, 512, 3), compression="gzip")
    img_group.create_dataset('robot0_eye_in_hand', data=np.array(robot_eye_images, dtype=np.uint8), chunks=(1, 512, 512, 3), compression="gzip")
    
    # 🔥 Robust string storage
    context_group = traj_group.create_group('failure_context')
    string_prompts = [str(p) for p in failure_context_prompts]
    try:
        max_byte_length = max(len(p.encode('utf-8')) for p in string_prompts) + 1
        ascii_dtype = h5py.string_dtype('utf-8', max_byte_length)
        context_group.create_dataset('prompts', data=string_prompts, dtype=ascii_dtype)
    except Exception as e:
        # Fallback to variable-length strings if fixed-length fails
        print(f"    ⚠️ Fixed-length string encoding failed ({e}), using variable-length.")
        dt = h5py.special_dtype(vlen=str)
        context_group.create_dataset('prompts', data=string_prompts, dtype=dt)

    if string_prompts: traj_group.attrs['sample_cot_instruction'] = string_prompts[0]
    
    print(f"  ✅ Task {task_id} trajectory {trajectory_idx} saved ({len(actions)} steps, dual-view)")
    return True

def main(args):
    all_json_files_with_task = [] # List of (json_path, task_id) tuples
    
    for data_dir_str in args.expert_data_dirs:
        expert_data_dir = Path(data_dir_str)
        if not expert_data_dir.is_dir(): continue
        
        print(f"🔍 Searching in: {expert_data_dir}")
        found_files = list(expert_data_dir.glob('*_EXPERT.json'))
        
        # Extract task_id for each found file
        for json_file in found_files:
            task_id = extract_task_id_from_path(json_file)
            if task_id != -1: all_json_files_with_task.append((json_file, task_id))
    
    # Sort files by task_id and then by name
    all_json_files_with_task.sort(key=lambda x: (x[1], x[0].name))
    
    if not all_json_files_with_task:
        print("Error: No '*_EXPERT.json' files with valid task IDs found.")
        return
    
    # Count trajectories per task
    task_counts = {}
    for _, task_id in all_json_files_with_task:
        task_counts[task_id] = task_counts.get(task_id, 0) + 1
    
    print(f"\n✅ Total files to process: {len(all_json_files_with_task)}")
    print("📊 Task distribution:")
    for task_id in sorted(task_counts.keys()): print(f"   Task {task_id}: {task_counts[task_id]} trajectories")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / args.output_filename
    if output_path.exists(): output_path.unlink()
    
    with h5py.File(output_path, 'w') as hdf5_file:
        # 🔥 Update metadata to reflect multi-task nature
        hdf5_file.attrs['description'] = "Multi-task (Tasks 1-10) expert dataset with fixed template CoT instructions and dual-view images"
        hdf5_file.attrs['data_format'] = "multi_task_cot_dual_view"
        
        trajectory_count = 0
        for json_path, task_id in tqdm(all_json_files_with_task, desc="Processing trajectories"):
            if process_enhanced_trajectory(json_path, hdf5_file, trajectory_count, task_id):
                trajectory_count += 1
    
    print(f"\n🎉 Multi-task expert dataset ready! {trajectory_count} trajectories from {len(task_counts)} tasks in {output_path}")
    print(f"📊 Dataset size: {output_path.stat().st_size / (1024 * 1024):.1f} MB")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a multi-task expert dataset with actionable CoT instructions and dual-view images.")
    parser.add_argument("--expert-data-dirs", type=str, nargs='+', required=True, help="Paths to expert data directories for all tasks.")
    parser.add_argument("--output-dir", type=str, default="data/hdf5_datasets", help="Output directory.")
    parser.add_argument("--output-filename", type=str, default="multi_task_expert_dataset.hdf5", help="Output filename.")
    args = parser.parse_args()
    main(args)