from model.model_loader import load_model_and_tokenizer, load_test_examples , create_test_examples
from data.data_loading import load_training_data
#from data.data_tokenizing import data_tokenizing
from model.model_evaluation import run_model_evaluation
from config import device, max_length
import os, json

def main():
    
    #print(f"Using device: {device}")
# Lo#ad or create test examples
    #test_examples = load_test_examples()
    #if not test_examples:
    #    print("No test examples found. Creating sample test examples.")
    #    test_examples = create_test_examples()
    #    
    #    # Save the created examples for future use
    #    os.makedirs("./data", exist_ok=True)
    #    with open("./data/test_examples.json", "w") as f:
    #        json.dump(test_examples, f, indent=2)
    #    print("Sample test examples created and saved to ./data/test_examples.json")
    ##, tokenizer = load_model_and_tokenizer()
    # Run evaluation
    # 1. Test before fine-tuning
    print("\n\n=============== BEFORE FINE-TUNING EVALUATION ===============")
    before_results = run_model_evaluation("Before Fine-tuning", save_results=True)
    
    
    #results = run_model_evaluation(
    #    model_name="Your Fine-tuned Model",
    #    test_examples=test_examples,
    #    save_results=True,
    #    display_examples=3,  # Show first 3 examples
    #    load_model_fn=lambda: load_model_and_tokenizer(),
    #    device=device
    #)
    
    # You can do additional analysis with the results here
    #print(f"\nTotal examples evaluated: {len(results['results'])}")
    
if __name__ == "__main__":
    main()