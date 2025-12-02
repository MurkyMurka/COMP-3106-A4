# Name this file assignment4.py when you submit
import numpy as np
import os

class bag_of_words_model:

  # class variables; initialized in __init__
  vocabulary = {}
  idf = None

  def __init__(self, directory):
    # directory is the full path to a directory containing trials through state space
    
    self.train(directory)

    # Return nothing

  def train(self, directory):
    # builds vocabulary and stores in an alphabetically-sorted list
    # stores documents in a 2D-list of the form documents[doc][word]
    
    vocabulary_set = set()
    documents = []
    
    # read and temporarily store documents
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
        documents.append(words_in_document)

    # store vocabulary
    self.vocabulary = sorted(vocabulary_set)

    # compute and store idf
    num_documents = len(documents)
    occurence_vector = []
    for word in self.vocabulary:
      occurences = 0
      for document in documents:
        if(word in document):
          occurences += 1
      occurence_vector.append(occurences)
    idf = np.log2(num_documents/np.array(occurence_vector))
    self.idf = idf

  def tf(self, document_filepath):
    # compute the tf of the document at document_filepath

    with open(document_filepath, 'r') as f:
      for line in f:
         # skip empty lines
        line = line.strip()
        if not line:
          continue
        words = line.split()
        tf = np.empty(len(self.vocabulary), dtype=float)
        index = 0
        for word in self.vocabulary:
          num = words.count(word)
          den = len(words)
          tf[index] = num/den
          index = index + 1
        return tf
    
  def tf_idf(self, document_filepath):
    # document_filepath is the full file path to a test document
    # Return the term frequency-inverse document frequency vector for the document

    # compute tf vector for document at document_filepath
    tf = self.tf(document_filepath)

    # compute and return tf-idf
    tf_idf = tf * self.idf
    return tf_idf.tolist()

  def predict(self, document_filepath, business_weights, entertainment_weights, politics_weights):
    # document_filepath is the full file path to a test document
    # business_weights is a list of weights for the business artificial neuron
    # entertainment_weights is a list of weights for the entertainment artificial neuron
    # politics_weights is a list of weights for the politics artificial neuron

    topics = ["business", "entertainment", "politics"]

    # get tf-idf from doc
    tf_idf = self.tf_idf(document_filepath)

    # get aggregate from neurons
    o_business = self.aggregate(tf_idf, business_weights)
    o_entertainment = self.aggregate(tf_idf, entertainment_weights)
    o_politics = self.aggregate(tf_idf, politics_weights)

    # get scores and predict label
    scores = self.softmax([o_business, o_entertainment, o_politics])
    predicted_label = topics[scores.index(max(scores))]

    # Return the predicted label from the neural network model
    # Return the score from each neuron
    return predicted_label, scores
  
  def aggregate(self, x, weights):
    # returns the aggregate value of a neuron using input and weights

    return np.dot(weights, x)
        
  def softmax(self, y_vector):
    # returns the softmax values of the given list as a list

    ret_vector = []
    den = 0
    for y_j in y_vector:
      den += np.exp(y_j)
    for y_i in y_vector:
      num = np.exp(y_i)
      ret_vector.append(float(num/den))
    return ret_vector

# TEST
'''
example = 'Example2'

learner = bag_of_words_model(f"Examples/{example}/training_documents")
print(learner.tf_idf(f"Examples/{example}/test_document.txt"))

# read weights from .txt file
def read_weights(document_filepath):
  with open(document_filepath, 'r') as f:
    for line in f:
      # skip empty lines
      line = line.strip()
      if not line:
        continue
      stringlist = line.split(",")
  floatlist = []
  for string in stringlist:
    floatlist.append(float(string))
  return floatlist

print(learner.predict(f"Examples/{example}/test_document.txt", read_weights(f"Examples/{example}/business_weights.txt"), read_weights(f"Examples/{example}/entertainment_weights.txt"), read_weights(f"Examples/{example}/politics_weights.txt")))
'''
