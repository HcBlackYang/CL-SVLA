# -*- coding: utf-8 -*-
import torch

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class MonitoringConfig:
    frequency: int = 10
    state_buffer_size: int = 300
    visual_buffer_size: int = 50
    enable_real_time: bool = True
    detailed_collection_frequency: int = 10
    enable_detailed_collection: bool = True
    save_intermediate_data: bool = True
    max_contact_points: int = 10
    enable_brightness_stats: bool = True
    enable_sensor_data: bool = True

@dataclass
class StateSummarizerConfig:
    history_window: int = 20
    stuck_threshold: float = 0.001
    oscillation_threshold: float = 0.005
    min_movement_steps: int = 3

@dataclass
class SuccessDetectionConfig:
    model_path: Optional[str] = None 
    failure_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "Stuck": ["stuck", "not moving", "no movement", "minimal movement"],
        "ObjectDropped": ["dropped", "gripper opened unexpectedly", "lost contact"],
        "PositioningError": ["oscillating", "overshooting", "wrong direction", "shaking"],
        "Collision": ["collision detected", "force spike"]
    })

@dataclass
class CoTConfig:
    model_name: str = "gpt-4"
    max_reasoning_steps: int = 10
    temperature: float = 0.7
    max_tokens: int = 500
    enable_few_shot: bool = True
    template_path: str = "config/cot_templates.json"

@dataclass
class FeedbackConfig:
    model_name: str = "gpt-3.5-turbo"
    max_length: int = 200
    temperature: float = 0.5
    enable_contextual: bool = True
    template_path: str = "config/feedback_templates.json"

@dataclass
class RetryConfig:
    max_attempts: int = 3
    decay_factor: float = 0.8
    timeout_seconds: int = 300
    enable_adaptive: bool = True


@dataclass
class MFMConfig:
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    summarizer: StateSummarizerConfig = field(default_factory=StateSummarizerConfig) 
    success_detection: SuccessDetectionConfig = field(default_factory=SuccessDetectionConfig)
    cot: CoTConfig = field(default_factory=CoTConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "float16"
    
    log_level: str = "INFO"
    log_dir: str = "data/logs"
    enable_wandb: bool = False
    wandb_project: str = "mfm-vla"
    
    data_dir: str = "data"
    cache_dir: str = "data/cache"
    state_save_dir: str = "/root/autodl-tmp/openvla/MFM/data/state"
    save_intermediate: bool = True

def load_config(config_path: Optional[str] = None) -> MFMConfig:
    if config_path is None:
        return MFMConfig()
    # TODO:
    return MFMConfig()