from bot.gradio import create_chatbot_interface

if __name__ == "__main__":

    MODEL_PATH = "./gpt2-alpaca-lora"

    print(f"Loading model from {MODEL_PATH}...")
    demo = create_chatbot_interface(MODEL_PATH)
    print("Starting Gradio interface...")
    demo.launch(share=True)  #