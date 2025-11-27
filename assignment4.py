# Name this file assignment4.py when you submit
import numpy as np
import os

TRAINING_DOCUMENTS = "training_documents"

class bag_of_words_model:

  vocabulary = {}
  documents =[]

  def __init__(self, directory):
    # directory is the full path to a directory containing trials through state space
    
    self.parse_documents(directory)

    # Return nothing

  def parse_documents(self, directory):
    # adds all words to the dictionary and puts in into an alphabetically-sorted list
    #stores documents in a list of the form documents[document number][word index]
    vocabulary_set = set()
    
    for root, dirs, files in os.walk(directory):
      for file in files:
        words_in_document = []
        if file.endswith(".txt"):
          file_path = os.path.join(root, file)
          with open(file_path, 'r') as f:
            for line in f:
              # skip empty lines
              line = line.strip()
              if not line:
                continue
              words = line.split()
              words_in_document.extend(words)
              for w in words:
                vocabulary_set.add(w)
        self.documents.append(words_in_document)

    self.vocabulary = sorted(self.vocabulary)
    
  def tf_idf(self, document_filepath):
    # document_filepath is the full file path to a test document
    # Return the term frequency-inverse document frequency vector for the document

    # compute idf using training documents
    document_in_training_set = len(self.documents)
    occurence_vector = []
    for vocabulary_word in self.vocabulary:
      occurences = 0
      for document in self.documents:
        if(document.contains(vocabulary_word)):
          occurences += 1
      occurence_vector.append(occurences)
    idf = np.log2(np.array(occurence_vector)/document_in_training_set)

    # compute tf in document at document_filepath
    tf = np.array({})
    with open(document_filepath, 'r') as f:
      for line in f:
         # skip empty lines
        line = line.strip()
        if not line:
          continue
        words = line.split()
        for word in self.vocabulary:
          numerator = words.count(word)
          denominator = len(words)
          np.append(tf, numerator/denominator)

    # compute tf-idf vector for document at document_filepath
    tf_idf_vector = tf * idf

    #compute tf
    return tf_idf_vector

  def predict(self, document_filepath, business_weights, entertainment_weights, politics_weights):
    # document_filepath is the full file path to a test document
    # business_weights is a list of weights for the business artificial neuron
    # entertainment_weights is a list of weights for the entertainment artificial neuron
    # politics_weights is a list of weights for the politics artificial neuron

    # Return the predicted label from the neural network model
    # Return the score from each neuron
    return predicted_label, scores
  
learner = bag_of_words_model(r"C:\Users\adamm\OneDrive\Desktop\Year 4\COMP 3106\ass4\COMP-3106-A4\Examples\Example0\training_documents")
print(learner.tf_idf(r"C:\Users\adamm\OneDrive\Desktop\Year 4\COMP 3106\ass4\COMP-3106-A4\Examples\Example0\test_document.txt"))