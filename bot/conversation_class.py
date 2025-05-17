from transformers import TextGenerationPipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def load_model(model_path="./gpt2-alpaca-lora"):
    # Load LoRA configuration
    config = PeftConfig.from_pretrained(model_path)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

    # Apply LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


class ChatbotConversation:
    def __init__(self, model_path):
        self.model, self.tokenizer = load_model(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.pipeline = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        self.conversation_history = []

    def reset(self):
        self.conversation_history = []
        return "Conversation has been reset."

    def format_prompt(self):
        if not self.conversation_history:
            return ""

        formatted_history = []
        for role, message in self.conversation_history:
            formatted_history.append(f"{role}: {message}")

        return "\n".join(formatted_history) + "\nAssistant:"

    def generate_response(self, user_input, max_length=100, temperature=0.7):
        # Add user input to history
        self.conversation_history.append(("User", user_input))

        # Format the prompt with conversation history
        prompt = self.format_prompt()

        # Generate response
        generated_text = self.pipeline(
            prompt,
            max_length=len(self.tokenizer.encode(prompt)) + max_length,
            do_sample=True,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )[0]["generated_text"]

        # Extract only the model's response
        response = generated_text[len(prompt):].strip()

        # Clean up the response if it contains start of a new turn
        if "User:" in response:
            response = response.split("User:")[0].strip()

        # Add the response to history
        self.conversation_history.append(("Assistant", response))

        return response