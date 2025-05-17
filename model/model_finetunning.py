from  peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, get_peft_model
from model.model_evaluation import load_model_and_tokenizer
import torch
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from data.data_processing import process_dataset


def train_lora_model(output_dir="./gpt2-alpaca-lora", max_steps=1000):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Process dataset
    processed_dataset = process_dataset(max_length=512, split="train", num_proc=4)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["c_attn", "c_proj"],
        inference_mode=False
    )

    # Prepare model for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=max_steps,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        report_to="tensorboard",
        run_name="gpt2-alpaca-lora",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    output_dir = "outputs/fine_tuned_model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer