import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai-community/gpt2"
max_length = 512