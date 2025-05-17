from transformers import AutoModelForCausalLM, AutoTokenizer
from config import model_name
import torch 
import json

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

def load_test_examples(file_path="./data/test_examples.json"):
    """
    Load test examples from a JSON file.
    
    Args:
        file_path: Path to the test examples file
        
    Returns:
        list: Test examples
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading test examples: {e}")
        return []

def create_test_examples():
    """
    Create a sample set of test examples.
    Useful when you don't have an existing test set.
    
    Returns:
        list: Sample test examples
    """
    return [
        {
            "instruction": "Write a short poem about artificial intelligence.",
            "input": "",
            "reference_output": "Silicon minds in digital space,\nLearning, growing at rapid pace.\nNot alive, yet thinking deep,\nPromises to keep, and miles to leap."
        },
        {
            "instruction": "Explain the concept of machine learning to a 10-year-old.",
            "input": "",
            "reference_output": "Machine learning is like teaching a computer to learn from examples, just like how you learn from your teacher. When you show the computer lots of pictures of cats, it starts to recognize cats on its own!"
        },
        {
            "instruction": "Summarize the following text.",
            "input": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
            "reference_output": "NLP is a field that focuses on enabling computers to understand and work with human language, combining linguistics, computer science, and AI to analyze text data."
        }
        # Add more examples as needed
    ]