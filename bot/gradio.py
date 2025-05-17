import gradio as gr
from bot.conversation_class import ChatbotConversation



#Creating a Gradio interface
def create_chatbot_interface(model_path):
    chatbot = ChatbotConversation(model_path)

    def user_interaction(message, history):
        return chatbot.generate_response(message)

    def reset_conversation():
        return chatbot.reset()

    with gr.Blocks() as demo:
        gr.Markdown("# Your Fine-Tuned GPT-2 Chatbot")

        chatbot_interface = gr.ChatInterface(
            user_interaction,
            examples=[
                "Hello, how can you help me today?",
                "Tell me something interesting.",
                "What capabilities do you have?"
            ],
            title="Chat with the AI Assistant"
        )

        with gr.Row():
            reset_btn = gr.Button("Reset Conversation")
            reset_btn.click(reset_conversation, outputs=gr.Textbox())

    return demo