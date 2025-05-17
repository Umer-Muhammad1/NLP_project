from data.data_loading import load_training_data  
from model.model_evaluation import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer()



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
    model, tokenizer = load_model_and_tokenizer()
    def prepare_training_example(example, max_length=512):
        """
        Prepares a dataset example for instruction fine-tuning by tokenizing and
        properly formatting inputs and labels.

        Args:
            example (dict): Dictionary containing 'instruction', 'input', and 'output' keys
            max_length (int): Maximum sequence length

        Returns:
            dict: Processed example with input_ids, attention_mask, and labels
        """
        try:
            # Format the prompt and full text
            if example['input']:
                prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput:"
            else:
                # Handle cases where input is empty
                prompt = f"Instruction: {example['instruction']}\nOutput:"

            full_text = prompt + " " + example["output"]

            # Tokenize the full sequence
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_attention_mask=True
            )

            # Tokenize just the prompt to find its length
            prompt_tokens = tokenizer(
                prompt,
                add_special_tokens=False
            )

            # Set up labels: -100 for prompt tokens (to be ignored by loss function)
            labels = tokenized["input_ids"].copy()
            prompt_length = len(prompt_tokens["input_ids"])

            # Set prompt tokens to -100 so they're ignored in loss calculation
            for i in range(prompt_length):
                if i < len(labels):
                    labels[i] = -100

            # Also mask padding tokens in the labels
            for i in range(len(labels)):
                if tokenized["attention_mask"][i] == 0:  # This is a padding token
                    labels[i] = -100

            tokenized["labels"] = labels
            return tokenized

        except Exception as e:
            print(f"Error processing example: {e}")

            return {
                "input_ids": [tokenizer.pad_token_id] * 2,
                "attention_mask": [0] * 2,
                "labels": [-100] * 2
            }
    
    # Then process it
    processed_dataset = dataset.map(
        lambda x: prepare_training_example(x, max_length=max_length),
        remove_columns=dataset[split].column_names,
        desc="Processing dataset",
        num_proc=num_proc
    )
    
    return processed_dataset