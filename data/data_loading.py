import json
from datasets import Dataset

def load_training_data():
    with open("NLP_project/data/alpaca_data.json", "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)
