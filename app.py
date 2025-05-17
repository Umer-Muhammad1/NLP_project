from model.model_loader import load_model_and_tokenizer, load_test_examples , create_test_examples
from data.data_loading import load_training_data
#from data.data_tokenizing import data_tokenizing
from model.model_evaluation import run_model_evaluation
from model.model_finetunning import train_lora_model 
from config import device, max_length
import os, json

def main():
    print(f"Using device: {device}")

    # Load or create test examples (uncomment if needed)
    # test_examples = load_test_examples()
    # if not test_examples:
    #     print("No test examples found. Creating sample test examples.")
    #     test_examples = create_test_examples()
    #     os.makedirs("./data", exist_ok=True)
    #     with open("./data/test_examples.json", "w") as f:
    #         json.dump(test_examples, f, indent=2)
    #     print("Sample test examples created and saved to ./data/test_examples.json")

    # 1. Evaluate before fine-tuning
    print("\n\n=============== BEFORE FINE-TUNING EVALUATION ===============")
    before_results = run_model_evaluation("Before Fine-tuning", save_results=True)

    # 2. Train (fine-tune) the model with LoRA
    print("\n\n=============== STARTING FINE-TUNING ===============")
    finetuned_model, tokenizer = train_lora_model(output_dir="./gpt2-alpaca-lora", max_steps=500)

    # 3. Evaluate after fine-tuning using the fine-tuned model directly
    print("\n\n=============== AFTER FINE-TUNING EVALUATION ===============")
    after_results = run_model_evaluation(
        model_name="After Fine-tuning",
        load_model_fn=lambda: (finetuned_model, tokenizer),
        save_results=True,
        display_examples=3,
        device=device
    )

    print("\nFine-tuning and evaluation complete. Model saved to ./gpt2-alpaca-lora")

if __name__ == "__main__":
    main()
