# -*- coding: utf-8 -*-


from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class VLAModelConfig:
    model_name: str = "openvla/openvla-7b"
    model_path: str = "data/models/openvla"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

@dataclass
class EncoderConfig:
    vision_model: str = "clip-vit-base-patch32"
    language_model: str = "bert-base-uncased"
    state_embedding_dim: int = 256
    fusion_dim: int = 512
    dropout_rate: float = 0.1

@dataclass
class ClassifierConfig:
    input_dim: int = 512
    hidden_dims: List[int] = None
    num_classes: int = 2
    dropout_rate: float = 0.3
    activation: str = "relu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]

@dataclass
class ModelConfig:
    vla: VLAModelConfig = VLAModelConfig()
    encoder: EncoderConfig = EncoderConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
