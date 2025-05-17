from datasets import load_dataset

from datasets import load_dataset

def load_training_data():
    dataset = load_dataset(
        "tatsu-lab/alpaca",
        split="train",
        download_mode="force_redownload",
        cache_dir=None  # Avoid local cache altogether
    )
    return dataset
