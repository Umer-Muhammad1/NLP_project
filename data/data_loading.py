import json
from datasets import Dataset

def load_training_data():
    with open("data/alpaca_data.json", "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)
