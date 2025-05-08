import numpy as np

def cosine_similarity(vector1, vector2):
    #Produit scalaire des vecteurs
    scalar_product = np.dot(vector1, vector2)
    #Norme euclidienne des vecteurs
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    #Expression analytique du cosinus dans un espace euclidien
    cosine = scalar_product / (norm_vector1 * norm_vector2)
    return cosine