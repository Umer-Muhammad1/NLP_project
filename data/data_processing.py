# Assuming you're importing load_training_data from another file
from data_loading import load_training_data  # Adjust the import path as needed
from data_tokenizing import data_tokenizing


def process_dataset(max_length, split="train", num_proc=4):
    """
    Load and process the dataset by applying the prepare_training_example function.
    
    Args:
        max_length: Maximum sequence length for the model
        split: Dataset split to use for column names (default: "train")
        num_proc: Number of processes for parallel processing (default: 4)
        
    Returns:
        The processed dataset
    """
    # First load the dataset using the imported function
    dataset = load_training_data()
    
    # Then process it
    processed_dataset = dataset.map(
        lambda x: data_tokenizing(x, max_length=max_length),
        remove_columns=dataset[split].column_names,
        desc="Processing dataset",
        num_proc=num_proc
    )
    
    return processed_dataset