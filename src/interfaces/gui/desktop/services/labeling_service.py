from typing import Any, Dict, List

import pandas as pd

from src.labeling.pipeline import LabelingPipeline


class LabelingService:
    def get_available_methods(self) -> List[str]:
        return ["horizon", "triple_barrier", "regression", "custom"]

    def get_default_params(self, method: str) -> Dict[str, Any]:
        if method == "horizon":
            return {"horizon": 5, "threshold": 0.01}
        elif method == "triple_barrier":
            return {"horizon": 10, "ub": 0.02, "lb": 0.01}
        return {}

    def run_labeling(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        # Mock dataset_id for now since we might work with in-memory data in GUI
        pipeline = LabelingPipeline.from_config(config, data)
        labeled_data, metadata = pipeline.run(data, save_results=False)
        return labeled_data
