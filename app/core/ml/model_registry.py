from typing import Dict


MODEL_REGISTRY: Dict[str, dict] = {
    "ftg_net_v1": {
        "name": "FTG-NET v1",
        "checkpoint": "models/checkpoints_v4_metadata/best_model_1.pt",
        "description": "Latest optimized FTG-NET"
    },
    "ftg_net_v2": {
        "name": "FTG-NET v2",
        "checkpoint": "models/checkpoints_v4_metadata/best_model_2.pt",
        "description": "Baseline FTG-NET trained on CICIDS2019"
    },
    "ftg_net_v3": {
        "name": "FTG-NET v3",
        "checkpoint": "models/checkpoints_v4_metadata/best_model_3.pt",
        "description": "Improved generalization"
    },
    "ftg_net_v4": {
        "name": "FTG-NET v4",
        "checkpoint": "models/checkpoints_v4_metadata/best_model_1_finetuned.pt",
        "description": "Finetuned for Lab Simulation"
    },
}

DEFAULT_MODEL_ID = "ftg_net_v4"
