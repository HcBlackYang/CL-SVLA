from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Config:
    model_family: str = "openvla"


    pretrained_checkpoint: str = "/root/autodl-tmp/openvla/weights/openvla-7b-finetuned-libero-spatial"
    # pretrained_checkpoint: str = "/root/autodl-tmp/openvla/weights/openvla-7b"
    # pretrained_checkpoint: str = "/mnt/workspace/openvla-7b"

    lora_checkpoint: str | None = None

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    task_suite_name: str = "libero_spatial"
    
    num_trials_per_task: int = 10

    max_steps: int = 400

    local_log_dir: str = "root/autodl-tmp/openvla/openvla/experiments/logs"        # Local directory for eval logs

    # unnorm_key: str = "action" 
    unnorm_key: str = "libero_spatial" 
