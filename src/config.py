import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class DataConfig:
    db_path: str = "ensanut_2018.db"
    target_column: str = "p3_1"
    exclude_gestational: bool = True

@dataclass
class FeaturesConfig:
    use_demographics: bool = True
    use_anthropometrics: bool = True
    use_comorbidities: bool = True
    use_family_history: bool = True
    use_lifestyle: bool = True
    use_household_assets: bool = True
    use_survey_weights: bool = True

@dataclass
class GraphConfig:
    k: int = 10
    metric: str = "cosine"
    algorithm: str = "brute"

@dataclass
class SplitConfig:
    strategy: str = "clustered_upm"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42

@dataclass
class ModelConfig:
    name: str = "GATv2"
    hidden_dim: int = 64
    heads: int = 4
    dropout: float = 0.3

@dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 0.005
    weight_decay: float = 0.0001
    focal_loss_alpha: float = 0.75
    focal_loss_gamma: float = 2.0
    device: str = "cpu"
    save_checkpoint: bool = True
    checkpoint_path: str = "checkpoints/best_gatv2.pt"

@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

def load_config(config_path: str = "configs/default.yaml") -> PipelineConfig:
    if not os.path.exists(config_path):
        return PipelineConfig()
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return PipelineConfig(
        data=DataConfig(**raw.get("data", {})),
        features=FeaturesConfig(**raw.get("features", {})),
        graph=GraphConfig(**raw.get("graph", {})),
        split=SplitConfig(**raw.get("split", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
