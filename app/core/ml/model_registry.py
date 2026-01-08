from typing import Dict

# DO NOT include ensemble models
MODEL_REGISTRY: Dict[str, dict] = {
    "ftgnet_v1": {
        "name": "FTG-NET v1",
        "checkpoint": "models/checkpoints_v4_metadata/best_model_1.pt"
    },
    "ftgnet_v2": {
        "name": "FTG-NET v2",
        "checkpoint": "models/checkpoints_v4_metadata/best_model_2.pt"
    },
    "ftgnet_v3": {
        "name": "FTG-NET v3",
        "checkpoint": "models/checkpoints_v4_metadata/best_model_3.pt"
    }
}

DEFAULT_MODEL_ID = "ftgnet_v1"
