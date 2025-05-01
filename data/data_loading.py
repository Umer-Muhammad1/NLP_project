from datasets import load_dataset

def load_training_data():
    dataset = load_dataset("tatsu-lab/alpaca")
    print(f"Dataset loaded with {len(dataset['train'])} training examples")
    return dataset
