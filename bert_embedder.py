from transformers import AutoModel, AutoTokenizer
import torch
import faiss

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

my_embeddings = generate_embeddings("the red drone was here, and i am a fish")

# print(my_embeddings)

# adding to db.

dimension = 768
index = faiss.IndexFlatL2(dimension)

def add_embeddings_to_db(embedding):
    index.add(embedding)

def query_vector_db(query_text):
    query_embedding = generate_embeddings(query_text)
    distances, indices = index.search(query_embedding, k=5)
    return indices[0]

add_embeddings_to_db(my_embeddings)

result = query_vector_db("what color is the drone?")

print(result)
