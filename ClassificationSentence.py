from VectorsSimilarity import cosine_similarity
from ModelEmbedding import get_vector_from_model
import numpy as np

def classify_sentence(test_desc, dataset_vectors,dataset_labels):
    # Vectorisation de la phrase
    test_desc_vector = get_vector_from_model(test_desc)
    # Calcul des similarités cosinus entre le vecteur de la phrase 
    # à classifier et les vecteurs du dataset
    similarity_values = [cosine_similarity(test_desc_vector, vect) for vect in dataset_vectors]
    # Obtenir l'indice du cosinus le plus élevé et projeter 
    # pour obtenir la valeur de classe prédite
    pred_label, cosine_max = dataset_labels[np.argmax(similarity_values)], max(similarity_values)
    return pred_label, cosine_max