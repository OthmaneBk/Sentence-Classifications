
from DataTrain import labels_list
from DataSet import descriptifs_test
from ClassificationSentence import classify_sentence
from ModelEmbedding import vectors

if __name__ == "__main__":

    
    for sentence in descriptifs_test:
        #Classification de la phrase
        predicted_label, cosine_value = classify_sentence(sentence, vectors, labels_list)
        print(f"Phrase : {sentence}\nClasse prédite : {predicted_label}\nCosinus max : {predicted_label}\n")