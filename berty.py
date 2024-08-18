import fitz
from transformers import AutoModel, AutoTokenizer
import faiss
import torch
import requests
import json
import ollama
import time
conversation_history = ""

dimension = 768
index = faiss.IndexFlatL2(dimension)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
        return text

def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def add_embeddings_to_db(embedding):
    index.add(embedding)

def query_vector_db(query_text):
    query_embedding = generate_embeddings(query_text)
    distances, indices = index.search(query_embedding, k=5)
    return indices[0]

def update_conversation_history(user_input, model_response):
    global conversation_history
    conversation_history += f"User: {user_input}\nAI: {model_response}\n"
"""
def generate_augmented_response(user_input, indices, docs):
    global conversation_history

    relevant_doc = [docs[i] for i in indices if i < len(docs)]
    combined_content = conversation_history + f"User: {user_input}\n" + " ".join(relevant_doc[:2])

    prompt = f"You are the ultimate warrior of code and also a helpful assistant.
    Context: {combined_content}
    User: {user_input}
    AI:"
    
    api_url = "http://localhost:11434/api/generate"
    header = { "Content-Type": "application/json" }
    data = { "model": "deepseek-coder:6.7b-base-q4_K_M", "prompt": prompt, "stream": False }
    
    response = requests.post(api_url, headers=header, data=json.dumps(data))
    
    if response.staus_code == 200:
        result = response.json()
        ai_response = result['response']
    else:
        ai_response = "Get Stuffed i'm not answering that you pervert!"

    update_conversation_history(user_input, ai_response)
    
    return ai_response
    
    # return prompt
"""

def generate_augmented_response(user_input, indices, docs, max_retries=3, retry_delay=5):
    global conversation_history
    
    relevant_texts = [docs[i] for i in indices if i < len(docs)]
    combined_context = conversation_history + f"User: {user_input}\n" + " ".join(relevant_texts[:2])
    
    prompt = f"""You are a helpful assistant. 
    Context: {combined_context}
    User: {user_input}
    AI:"""

    for attempt in range(max_retries):
        try:
            print(f"Sending request to Ollama API (Attempt {attempt + 1}/{max_retries})...")
            start_time = time.time()
            
            response = ollama.generate(model='deepseek-coder:6.7b-base-q4_K_M', prompt=prompt)
            
            end_time = time.time()
            
            print(f"Request completed in {end_time - start_time:.2f} seconds")
            
            ai_response = response['response']
            break  # Success, exit the retry loop
        except ollama.ResponseError as e:
            print(f"Ollama response error: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Unable to get a response.")
                ai_response = "I'm sorry, but I'm having trouble connecting to my knowledge base right now. Could you please try again later?"
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            ai_response = f"An unexpected error occurred: {str(e)}"
            break  # Exit the retry loop for unexpected errors

    update_conversation_history(user_input, ai_response)
    return ai_response
pdf_paths = ["Extract[1].pdf"]
docs = []

for pdf_path in pdf_paths:
    text = extract_text_from_pdf(pdf_path)
    embedding = generate_embeddings(text)
    add_embeddings_to_db(embedding)
    docs.append(text)



while True:
    user_input = input(">> ")
    indices = query_vector_db(user_input)
    response = generate_augmented_response(user_input, indices, docs)

    print("AI:", response)
    
    if user_input.lower() in ['exit', 'quit', 'bye']:
        break
