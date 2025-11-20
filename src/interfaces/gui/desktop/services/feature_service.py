from typing import List

import pandas as pd
import yaml

from src.features.config_parser import FeatureConfig
from src.features.generator import FeatureGenerator
from src.features.indicators.registry import IndicatorRegistry


class FeatureService:
    def get_available_indicators(self) -> List[str]:
        return IndicatorRegistry.list_all()

    def load_config(self, path: str) -> FeatureConfig:
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        # Assuming simple structure for now, might need parsing adjustment
        return FeatureConfig(**config_dict)

    def save_config(self, config: FeatureConfig, path: str):
        with open(path, "w") as f:
            yaml.dump(config.model_dump(mode="json"), f, sort_keys=False)

    def generate_features(self, data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        generator = FeatureGenerator(config)
        return generator.generate(data, use_cache=False)  # Force generation for GUI preview
