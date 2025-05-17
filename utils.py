def data_tokenizing(example, tokenizer, max_length=512):
    try:
        if example['input']:
            prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput:"
        else:
            prompt = f"Instruction: {example['instruction']}\nOutput:"

        full_text = prompt + " " + example["output"]

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True
        )

        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        labels = tokenized["input_ids"].copy()
        prompt_length = len(prompt_tokens["input_ids"])

        for i in range(prompt_length):
            if i < len(labels):
                labels[i] = -100

        for i in range(len(labels)):
            if tokenized["attention_mask"][i] == 0:
                labels[i] = -100

        tokenized["labels"] = labels
        return tokenized

    except Exception as e:
        print(f"Error processing example: {e}")
        return None
