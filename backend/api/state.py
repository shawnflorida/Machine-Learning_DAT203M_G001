from dataclasses import dataclass

import pandas as pd


@dataclass
class ModelState:
    models: list
    best_model: object
    pipeliner: object
    predictor: object
    profile_generator: object
    source_df: pd.DataFrame
