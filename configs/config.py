import os
import yaml
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Any

def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def hash_dict(d: dict) -> str:
    raw = yaml.dump(d, sort_keys=True).encode('utf-8')
    return hashlib.md5(raw).hexdigest()

@dataclass
class BaseConfig:
    _raw: dict = field(default_factory=dict)
    _hash: str = ""

    def validate(self):
        """Override if needed"""
        pass

@dataclass
class LoRAConfig(BaseConfig):
    # ===== Switch =====
    enabled: bool = False
    task_type: str = "CAUSAL_LM"

    # ===== Structure =====
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=list)
    bias: str = "none"

    # ===== Init =====
    init_lora_weights: bool = True
    fan_in_fan_out: bool = False

    # ===== Precision / Device =====
    dtype: str = "bf16"
    device: str = "cuda"

    # ===== Training Control =====
    freeze_base_model: bool = True
    train_lora_only: bool = True
    lora_grad_clip: float = 1.0
    scale_grad_by_rank: bool = True

    # ===== Save / Merge =====
    save_strategy: str = "adapter_only"
    save_steps: int = 500
    merge_weights_on_save: bool = False

    # ===== Compatibility Flags =====
    allow_distillation: bool = True
    allow_pruning: bool = True
    allow_quantization: bool = True

    def validate(self):
        if self.enabled:
            assert self.task_type in ("CAUSAL_LM", "SEQ_CLS")
            assert self.r > 0, "LoRA r must be positive"
            assert self.lora_alpha >= self.r, "alpha should >= r"
            assert 0.0 <= self.lora_dropout < 1.0
            assert self.bias in ("none", "all", "lora_only")
            assert len(self.target_modules) > 0, "target_modules empty"
            assert self.dtype in ("fp16", "bf16", "fp32")
            assert self.device in ("cuda", "cpu")
            assert self.save_strategy in ("adapter_only", "merged")

@dataclass
class DistillationConfig(BaseConfig):
    enabled: bool = False
    temperature: float = 2.0
    alpha: float = 0.7

    teacher_model_name: Optional[str] = None
    teacher_fp16: bool = True

    def validate(self):
        if self.enabled:
            assert self.teacher_model_name, "teacher_model_name required"
            assert self.temperature > 0
            assert 0.0 <= self.alpha <= 1.0

@dataclass
class PruningConfig(BaseConfig):
    enabled: bool = False
    sparsity: float = 0.3
    target_modules: List[str] = field(default_factory=list)
    start_step: int = 0
    end_step: int = 0

    def validate(self):
        if self.enabled:
            assert 0.0 < self.sparsity < 1.0
            assert self.start_step < self.end_step
            assert self.target_modules, "pruning target_modules empty"

@dataclass
class QuantizationConfig(BaseConfig):
    enabled: bool = False
    bits: int = 8
    backend: str = "bitsandbytes"

    def validate(self):
        if self.enabled:
            assert self.bits in (4, 8)
            assert self.backend in ("bitsandbytes", "gptq")

@dataclass
class CompressionConfig(BaseConfig):
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    def validate(self):
        self.distillation.validate()
        self.pruning.validate()
        self.quantization.validate()

@dataclass
class ControlConfig(BaseConfig):
    num_train_epochs: int = 1
    save_steps: int = 500

    def validate(self):
        assert self.num_train_epochs > 0
        assert self.save_steps > 0

@dataclass
class TrainingConfig(BaseConfig):
    output_dir: str = "outputs"
    seed: int = 42
    max_seq_length: int = 2048

    def validate(self):
        assert self.max_seq_length > 0

@dataclass
class Config:
    training: TrainingConfig
    control: ControlConfig
    lora: LoRAConfig
    compression: CompressionConfig

    config_hash: str = ""

    def validate(self):
        self.training.validate()
        self.control.validate()
        self.lora.validate()
        self.compression.validate()

def load_config(train_yaml: str, lora_yaml: str) -> Config:
    train_raw = load_yaml(train_yaml)
    lora_raw = load_yaml(lora_yaml)

    compression_raw = train_raw.get("compression", {})

    cfg = Config(
        training=TrainingConfig(**train_raw.get("training", {})),
        control=ControlConfig(**train_raw.get("control", {})),
        lora=LoRAConfig(**lora_raw.get("lora", {})),
        compression=CompressionConfig(
            distillation=DistillationConfig(
                **compression_raw.get("distillation", {})
            ),
            pruning=PruningConfig(
                **compression_raw.get("pruning", {})
            ),
            quantization=QuantizationConfig(
                **compression_raw.get("quantization", {})
            ),
        ),
    )

    # ===== Raw & Hash =====
    merged_raw = {
        "train": train_raw,
        "lora": lora_raw,
    }
    cfg.config_hash = hash_dict(merged_raw)

    cfg.validate()
    return cfg
