# -*- coding: utf-8 -*-
"""
Enhanced State Collector
Responsible for collecting and managing multimodal state information,
supporting both detailed online collection and batch processing.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import logging
import time
import copy

from MFM.config.mfm_config import MonitoringConfig
from MFM.utils.logging_utils import setup_logger

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

class StateCollector:
    """Enhanced multimodal state information collector."""
    
    def __init__(self, config: MonitoringConfig):
        """
        Initializes the StateCollector.

        Args:
            config (MonitoringConfig): Configuration for state monitoring and collection.
        """
        self.config = config
        self.logger = setup_logger("StateCollector")
        
        # State buffers
        self.state_buffer = deque(maxlen=config.state_buffer_size)
        self.visual_buffer = deque(maxlen=config.visual_buffer_size) # Limited size for MFM's internal real-time visual analysis
        self.action_history = deque(maxlen=config.state_buffer_size)
        self.reward_history = deque(maxlen=config.state_buffer_size)
        
        # Detailed state collection buffer
        self.detailed_state_buffer = deque(maxlen=config.state_buffer_size)
        
        # Current state snapshot
        self.current_observation_no_visuals = None # [MODIFIED] Only stores non-visual parts
        self.current_robot_state = None
        self.current_action = None
        self.current_step_count = 0
        
        self.collection_frequency = config.detailed_collection_frequency
        self.enable_detailed_collection = config.enable_detailed_collection
        
        self.logger.info(f"Enhanced State Collector initialized - Detailed collection frequency: {self.collection_frequency} steps")

    
    def reset(self):
        """Resets all internal buffers and counters."""
        self.state_buffer.clear()
        self.visual_buffer.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.detailed_state_buffer.clear()
        
        self.current_observation = None
        self.current_robot_state = None
        self.current_action = None
        self.current_step_count = 0
        
        self.logger.info("State Collector has been reset.")

    def _extract_non_visual_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Removes image data from the observation dictionary, keeping only vector and scalar data.
        """
        non_visual_obs = {}
        for key, value in observation.items():
            if not ('image' in key or 'depth' in key or 'segmentation' in key):
                non_visual_obs[key] = value
        return non_visual_obs


    def collect_step_data(
        self, 
        observation: Dict[str, Any], 
        action: np.ndarray, 
        reward: float, 
        done: bool, 
        info: Dict[str, Any],
        env: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Collects data from a single execution step (v3, simplified and corrected data structure).
        
        Args:
            observation (Dict[str, Any]): The observation from the environment.
            action (np.ndarray): The action taken by the agent.
            reward (float): The reward received.
            done (bool): Whether the episode has ended.
            info (Dict[str, Any]): Additional info from the environment.
            env (Optional[Any]): The environment instance for detailed collection.

        Returns:
            Dict[str, Any]: A dictionary containing the collected data for the step.
        """
        
        self.current_step_count += 1
        
        # Extract visual and non-visual data from the observation.
        visual_data = self._extract_visual_data(observation)
        non_visual_obs = self._extract_non_visual_observation(observation)
        
        self.current_observation_no_visuals = non_visual_obs
        self.current_action = action.copy() if isinstance(action, np.ndarray) else action
        
        # <<< [CORE FIX] >>> Construct a clearer state data packet.
        state_data = {
            'step': self.current_step_count,
            'observation_non_visual': non_visual_obs, # All non-visual observations are here, no nesting.
            'action': action,
            'reward': reward,
            'done': done,
            'info': info,
            'timestamp': time.time()
        }
        
        # Append to basic buffers
        self.state_buffer.append(state_data)
        if visual_data is not None:
            self.visual_buffer.append({'step': self.current_step_count, 'visual_data': visual_data})
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Detailed collection logic (remains unchanged).
        detailed_data = None
        if self.enable_detailed_collection and (self.current_step_count % self.collection_frequency == 0):
            detailed_data = self._collect_detailed_state(observation, action, reward, done, info, env)
            if detailed_data:
                self.detailed_state_buffer.append(detailed_data)
        
        return {
            'basic_state': state_data, # Return the entire state_data packet.
            'detailed_state': detailed_data,
            'collection_step': self.current_step_count
        }


    def _collect_detailed_state(
        self, 
        observation: Dict[str, Any], 
        action: np.ndarray, 
        reward: float, 
        done: bool, 
        info: Dict[str, Any],
        env: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Collects detailed state information, similar to the verbose info in eval.py.
        """
        detailed_data = {
            'step': self.current_step_count,
            'timestamp': self._get_timestamp(),
            'action': convert_numpy_types(action.tolist() if isinstance(action, np.ndarray) else action),
            'reward': convert_numpy_types(reward),
            'done': convert_numpy_types(done),
            'info': convert_numpy_types(info)
        }
        
        # Record observation details (including stats for images).
        detailed_data['observations'] = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                if 'image' in key:
                    # For image data, store statistics to save space.
                    detailed_data['observations'][key] = {
                        'shape': list(value.shape),
                        'dtype': str(value.dtype),
                        'brightness_stats': {
                            'mean': float(np.mean(value)),
                            'std': float(np.std(value)),
                            'min': int(np.min(value)),
                            'max': int(np.max(value))
                        }
                    }
                else:
                    # For other array data, save it completely.
                    detailed_data['observations'][key] = convert_numpy_types(value.tolist())
            else:
                detailed_data['observations'][key] = convert_numpy_types(value)
        
        # Collect detailed robot state, sensor data, and contact info if env is provided.
        robot_state_detailed = self._collect_detailed_robot_state(env)
        if robot_state_detailed:
            detailed_data['robot_state'] = robot_state_detailed
        
        sensor_data = self._collect_sensor_data(env)
        if sensor_data:
            detailed_data['sensor_data'] = sensor_data
        
        contact_data = self._collect_contact_data(env)
        if contact_data:
            detailed_data['contact_data'] = contact_data
        
        return detailed_data
    
    def _collect_detailed_robot_state(self, env: Optional[Any]) -> Optional[Dict[str, Any]]:
        """Collects detailed robot state information from the simulation."""
        if env is None: return None
        robot_state = {}
        try:
            original_env = env
            while hasattr(original_env, 'env'): # Handle environment wrappers.
                original_env = original_env.env
            
            if hasattr(original_env, 'sim') and hasattr(original_env.sim, 'data'):
                sim_data = original_env.sim.data
                # Collect joint states, forces, torques, etc.
                if hasattr(sim_data, 'qpos'): robot_state['qpos'] = convert_numpy_types(sim_data.qpos.copy().tolist())
                if hasattr(sim_data, 'qvel'): robot_state['qvel'] = convert_numpy_types(sim_data.qvel.copy().tolist())
                if hasattr(sim_data, 'qacc'): robot_state['qacc'] = convert_numpy_types(sim_data.qacc.copy().tolist())
                if hasattr(sim_data, 'ctrl'): robot_state['ctrl'] = convert_numpy_types(sim_data.ctrl.copy().tolist())
                if hasattr(sim_data, 'qfrc_actuator'): robot_state['actuator_torques'] = convert_numpy_types(sim_data.qfrc_actuator.copy().tolist())
                if hasattr(sim_data, 'qfrc_applied'): robot_state['applied_forces'] = convert_numpy_types(sim_data.qfrc_applied.copy().tolist())
                if hasattr(sim_data, 'sensordata'): robot_state['sensor_data'] = convert_numpy_types(sim_data.sensordata.copy().tolist())
                if hasattr(sim_data, 'time'): robot_state['sim_time'] = float(sim_data.time)
        except Exception as e:
            self.logger.warning(f"Error collecting detailed robot state: {e}")
            robot_state['error'] = str(e)
        return robot_state if robot_state else None
    
    def _collect_sensor_data(self, env: Optional[Any]) -> Optional[Dict[str, Any]]:
        """Collects sensor data from the simulation."""
        if env is None: return None
        sensor_data = {}
        try:
            original_env = env
            while hasattr(original_env, 'env'): # Handle wrappers
                original_env = original_env.env
            
            if hasattr(original_env, 'sim') and hasattr(original_env.sim, 'data'):
                sim_data = original_env.sim.data
                if hasattr(sim_data, 'sensordata') and sim_data.sensordata is not None:
                    sensor_data['force_torque'] = convert_numpy_types(sim_data.sensordata.copy().tolist())
                if hasattr(sim_data, 'cacc'):
                    sensor_data['accelerometer'] = convert_numpy_types(sim_data.cacc.copy().tolist())
                if hasattr(sim_data, 'cvel'):
                    sensor_data['gyroscope'] = convert_numpy_types(sim_data.cvel.copy().tolist())
        except Exception as e:
            self.logger.warning(f"Error collecting sensor data: {e}")
            sensor_data['error'] = str(e)
        return sensor_data if sensor_data else None
    
    def _collect_contact_data(self, env: Optional[Any]) -> Optional[Dict[str, Any]]:
        """Collects contact force information from the simulation."""
        if env is None: return None
        contact_data = {}
        try:
            original_env = env
            while hasattr(original_env, 'env'): # Handle wrappers
                original_env = original_env.env
            
            if hasattr(original_env, 'sim') and hasattr(original_env.sim, 'data'):
                sim_data = original_env.sim.data
                if hasattr(sim_data, 'contact') and sim_data.ncon > 0:
                    contacts = []
                    for i in range(min(sim_data.ncon, 10)):  # Limit to a max of 10 contact points
                        contact = sim_data.contact[i]
                        contact_info = {
                            'geom1': int(contact.geom1), 'geom2': int(contact.geom2),
                            'pos': convert_numpy_types(contact.pos.tolist()),
                            'frame': convert_numpy_types(contact.frame.tolist()),
                            'dist': float(contact.dist)
                        }
                        contacts.append(contact_info)
                    contact_data['contacts'] = contacts
                    contact_data['num_contacts'] = int(sim_data.ncon)
                else:
                    contact_data['contacts'] = []
                    contact_data['num_contacts'] = 0
        except Exception as e:
            self.logger.warning(f"Error collecting contact data: {e}")
            contact_data['error'] = str(e)
        return contact_data if contact_data else None
    
    def _extract_robot_state(self, observation: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts basic robot state information from observation and info dicts."""
        robot_state = {}
        robot_keys = ['robot0_joint_pos', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
        for key in robot_keys:
            if key in observation:
                new_key = key.replace('robot0_', '')
                robot_state[new_key] = observation[key]
        if 'robot_state' in info:
            robot_state.update(info['robot_state'])
        return robot_state
    
    def _extract_visual_data(self, observation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extracts the primary visual observation data (image)."""
        for key in observation:
            if 'image' in key and isinstance(observation[key], np.ndarray):
                return observation[key]
        return None
    
    def _get_timestamp(self) -> float:
        """Returns the current timestamp."""
        return time.time()
    
    def get_recent_states(self, n: int = 10) -> List[Dict[str, Any]]:
        """Returns the n most recent basic states."""
        return list(self.state_buffer)[-n:]
    
    def get_recent_detailed_states(self, n: int = 5) -> List[Dict[str, Any]]:
        """Returns the n most recent detailed states."""
        return list(self.detailed_state_buffer)[-n:]
    
    def get_recent_actions(self, n: int = 10) -> List[np.ndarray]:
        """Returns the n most recent actions."""
        return list(self.action_history)[-n:]
    
    def get_recent_rewards(self, n: int = 10) -> List[float]:
        """Returns the n most recent rewards."""
        return list(self.reward_history)[-n:]
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """Calculates and returns statistics about the collected states."""
        if not self.reward_history: return {}
        rewards = list(self.reward_history)
        return {
            'total_steps': len(self.state_buffer),
            'detailed_collections': len(self.detailed_state_buffer),
            'avg_reward': np.mean(rewards), 'total_reward': np.sum(rewards), 'reward_std': np.std(rewards),
            'buffer_utilization': len(self.state_buffer) / self.config.state_buffer_size,
            'collection_efficiency': len(self.detailed_state_buffer) / max(1, self.current_step_count // self.collection_frequency)
        }
    
    def get_current_state_snapshot(self) -> Dict[str, Any]:
        """Returns a snapshot of the current collection state."""
        return {
            'current_step': self.current_step_count,
            'observation': self.current_observation,
            'robot_state': self.current_robot_state,
            'action': self.current_action,
            'state_history_length': len(self.state_buffer),
            'visual_history_length': len(self.visual_buffer),
            'detailed_history_length': len(self.detailed_state_buffer)
        }
    
    def collect_environment_info(self, env) -> Dict[str, Any]:
        """Collects additional information from the environment instance."""
        env_info = {}
        try:
            if hasattr(env, 'sim') and hasattr(env.sim, 'data'):
                env_info['sim_time'] = getattr(env.sim.data, 'time', None)
                env_info['physics_timestep'] = getattr(env.sim.model, 'opt', {}).get('timestep', None)
            if hasattr(env, 'get_state'):
                env_info['env_state'] = env.get_state()
            if hasattr(env, '_max_episode_steps'):
                env_info['max_episode_steps'] = env._max_episode_steps
        except Exception as e:
            self.logger.warning(f"Error collecting environment info: {e}")
        return env_info
    
    def get_batch_data_for_analysis(self, window_size: int = None) -> Dict[str, Any]:
        """Gets a batch of recent data for analysis."""
        if window_size is None: window_size = self.collection_frequency
        return {
            'states': self.get_recent_detailed_states(window_size),
            'actions': self.get_recent_actions(window_size),
            'rewards': self.get_recent_rewards(window_size),
            'window_size': window_size, 'collection_timestamp': self._get_timestamp(),
            'step_range': (max(0, self.current_step_count - window_size), self.current_step_count)
        }
    
    def set_collection_frequency(self, frequency: int):
        """Dynamically sets the detailed collection frequency."""
        self.collection_frequency = frequency
        self.logger.info(f"Detailed collection frequency updated to: {frequency} steps")
    
    def enable_detailed_collection(self, enable: bool = True):
        """Enables or disables detailed state collection."""
        self.enable_detailed_collection = enable
        status = "Enabled" if enable else "Disabled"
        self.logger.info(f"Detailed state collection has been {status}")

    def export_collected_data(self, include_detailed: bool = True) -> Dict[str, Any]:
        """
        Exports the collected data, now without embedding images directly.
        The calling function (e.g., CoT engine) should fetch the latest image separately.
        """
        export_data = {
            'metadata': {'total_steps': self.current_step_count},
            'basic_states': list(self.state_buffer), # `basic_states` no longer contains images.
            'actions': list(self.action_history),
            'rewards': list(self.reward_history)
        }
        if include_detailed and self.detailed_state_buffer:
            export_data['detailed_states'] = list(self.detailed_state_buffer)
        return export_data
    
    # [NEW] A method for the CoT engine to directly get the latest image.
    def get_latest_visual(self) -> Optional[np.ndarray]:
        """
        Gets the most recent visual data from the visual buffer.
        
        Returns:
            Optional[np.ndarray]: The latest image as a numpy array, or None if the buffer is empty.
        """
        if not self.visual_buffer:
            return None
        return self.visual_buffer[-1]['visual_data']