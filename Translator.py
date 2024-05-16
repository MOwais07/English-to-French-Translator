import torch
import math
from transformers import MarianMTModel, MarianTokenizer

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    tokens = text.split()
    return tokens


# Function for positional encoding
def positional_encoding(tokens, d_model):
    pos_enc = torch.zeros(len(tokens), d_model)
    for pos in range(len(tokens)):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
    return pos_enc

# Load pre-trained MarianMT model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Take input from user
user_input = input("Enter a sentence to translate: ")

# Preprocess the input text
tokens = preprocess_text(user_input)
print("Tokens:", tokens)

# Define the dimension for positional encoding
d_model = 512

# Get positional encodings
pos_encodings = positional_encoding(tokens, d_model)

# Tokenize the input text
inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)

# Perform the translation
translated = model.generate(**inputs)

# Decode the translated sentence
translated_sentence = tokenizer.decode(translated[0], skip_special_tokens=True)
print("Original Sentence:", user_input)
print("Translated Sentence:", translated_sentence)