from dataclasses import dataclass


@dataclass
class ModelState:
    models: list
    pipeliner: object
    predictor: object
    profile_generator: object
