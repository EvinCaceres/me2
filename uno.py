import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar el modelo y el tokenizador
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Configurar el dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Función para generar respuestas
def generate_response(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Función para iniciar la conversación con el chatbot
def chatbot_conversation():
    print("Hola, soy un chatbot. Escribe 'adiós' para salir.")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ["adiós", "chao"]:
            print("Chatbot: Adiós, ¡cuídate!")
            break
        else:
            prompt = f"Tú: {user_input}\nChatbot:"
            response = generate_response(prompt)
            print("Chatbot:", response)

# Iniciar la conversación
if __name__ == "__main__":
    chatbot_conversation()


