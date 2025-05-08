from sentence_transformers import SentenceTransformer
from DataTrain import description_list

# Charger le modèle
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_vector_from_model(sentence):
    vector = model.encode(sentence)
    return vector

#Génération des vecteurs associés aux phrases descriptions du dataset
vectors = [get_vector_from_model(description) for description in description_list]