# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class LiberoTaskConfig:
    task_suite: str = "libero_10"  # libero_10, libero_90, libero_goal
    num_trials_per_task: int = 50
    max_steps_per_episode: int = 600
    resolution: int = 768
    enable_modification: bool = True

@dataclass
class EnvironmentConfig:
    name: str = "libero"
    render_mode: str = "rgb_array"
    enable_visualization: bool = True
    save_videos: bool = True
    video_dir: str = "data/videos"

@dataclass
class EvaluationConfig:
    metrics: List[str] = None
    save_detailed_logs: bool = True
    enable_ablation: bool = True
    comparison_baselines: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["success_rate", "step_efficiency", "retry_rate"]
        if self.comparison_baselines is None:
            self.comparison_baselines = ["vanilla_vla", "oracle_feedback"]

@dataclass
class TaskConfig:
    libero: LiberoTaskConfig = LiberoTaskConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    
    enable_curriculum: bool = False
    curriculum_stages: List[str] = None
    adaptive_difficulty: bool = True
    
    def __post_init__(self):
        if self.curriculum_stages is None:
            self.curriculum_stages = ["easy", "medium", "hard"]
