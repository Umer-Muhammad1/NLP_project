import torch
import json
import os
from model.model_loader import load_model_and_tokenizer , create_test_examples
import numpy as np
from tqdm import tqdm
from config import device, max_length


model , tokenizer = load_model_and_tokenizer()
test_examples= create_test_examples()

def evaluate_model(model, tokenizer, test_examples, device="cuda", max_new_tokens=100):
    """
    Evaluates a model on a list of test examples.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        test_examples: List of dictionaries with 'instruction', 'input', and 'reference_output' keys
        device: Device to run inference on
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        dict: Dictionary with generated responses and metrics
    """
    model.eval()
    results = []

    for example in test_examples:
        instruction = example['instruction']
        input_text = example.get('input', '')
        reference = example.get('reference_output', '')

        # Format the prompt based on whether input is provided
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
        else:
            prompt = f"Instruction: {instruction}\nOutput:"

        # Tokenize and move to device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode and extract only the response part
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_output[len(prompt):].strip()

        results.append({
            'instruction': instruction,
            'input': input_text,
            'generated_output': response,
            'reference_output': reference
        })

    return results


# Function to run tests and display results
def run_model_evaluation(model_name="Base Model", save_results=False):
    print(f"\n===== {model_name} Evaluation =====")

    model.to(device)
    results = evaluate_model(model, tokenizer, test_examples, device)

    for i, result in enumerate(results):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {result['instruction']}")
        if result['input']:
            print(f"Input: {result['input']}")
        print(f"\nGenerated output: {result['generated_output']}")
        if result['reference_output']:
            print(f"Reference output: {result['reference_output']}")
        print("-" * 50)

    if save_results:
        import json
        import os
        results_dir = "./evaluation_results"
        os.makedirs(results_dir, exist_ok=True)

        with open(f"{results_dir}/{model_name.replace(' ', '_').lower()}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_dir}/{model_name.replace(' ', '_').lower()}_results.json")

    return results