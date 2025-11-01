# utils.py

import pickle

def load_model(model_path, scaler_path):
    """Load trained model and scaler from pickle files."""
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    return model, scaler


def get_crop_recommendations():
    """Return dictionary of crop recommendations for each soil type."""
    return {
        "Sandy": ["Carrot", "Potato", "Groundnut"],
        "Loamy": ["Wheat", "Sugarcane", "Tomato"],
        "Clay": ["Rice", "Jute", "Paddy"],
        "Silty": ["Maize", "Soybean", "Barley"]
    }
