import time
import os
import imageio
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np 
import cv2 


DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DATE = time.strftime("%Y_%m_%d")



def get_libero_env(
    task,
    model_family, 
    resolution=256,
    camera_names="agentview",
    camera_heights=None,
    camera_widths=None
):
    """
    Initializes and returns the LIBERO environment, along with the task description.
    """
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    
    env_args = {
        "bddl_file_name": task_bddl_file,
    }

    env_args["camera_names"] = camera_names

    if camera_heights is not None:
        env_args["camera_heights"] = camera_heights
    else:

        if isinstance(camera_names, list):
            env_args["camera_heights"] = [resolution] * len(camera_names)
        else:
            env_args["camera_heights"] = resolution

    if camera_widths is not None:
        env_args["camera_widths"] = camera_widths
    else:
        if isinstance(camera_names, list):
            env_args["camera_widths"] = [resolution] * len(camera_names)
        else:
            env_args["camera_widths"] = resolution
            
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.
    (This function remains unchanged)
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """
    Saves an MP4 replay of an episode.
    (This function is now DEPRECATED in favor of the new dual-view function in dark_mfm.py, 
     but we keep it for compatibility with other potential scripts in the repo.)
    """
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path