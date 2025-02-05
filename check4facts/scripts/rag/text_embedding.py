from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load model and tokenizer
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)




#
def single_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state
    sentence_embedding = torch.mean(embeddings, dim=1)
    sentence_embedding = sentence_embedding.squeeze().numpy()

    return sentence_embedding

#returns the cosine similarity between two text embeddings
def cos_sim(embedding1, embedding2):
    # Ensure the embeddings are numpy arrays
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    # Calculate the dot product and magnitudes
    dot_product = np.dot(embedding1, embedding2)
    magnitude1 = np.linalg.norm(embedding1)
    magnitude2 = np.linalg.norm(embedding2)

    # Calculate cosine similarity
    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # If either embedding has zero magnitude, similarity is zero
    
    return dot_product / (magnitude1 * magnitude2)

