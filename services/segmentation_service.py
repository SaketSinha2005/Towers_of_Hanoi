from backend.config import MODEL_PATH
from src.train import load_best_model

model = None

def load_model():
    global model

    if model is None:
        model = load_best_model(MODEL_PATH)

    return model