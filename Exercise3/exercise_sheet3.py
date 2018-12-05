################################################################################
## SNLP exercise sheet 3
################################################################################
import math
import sys
import numpy as np


'''
This function can be used for importing the corpus.
Parameters: path_to_file: string; path to the file containing the corpus
Returns: list of list; the first layer list contains the sentences of the corpus;
    the second layer list contains tuples (token,label) representing a labelled sentence
'''
def import_corpus(path_to_file):
    sentences = []
    sentence = []
    f = open(path_to_file)

    while True:
        line = f.readline()
        if not line: break

        line = line.strip()
        if len(line) == 0:
            sentences.append(sentence)
            sentence = []
            continue

        parts = line.split(' ')
        sentence.append((parts[0].lower(), parts[-1]))

    f.close()
    return sentences




class MaxEntModel(object):
    # training corpus
    corpus = None
    
    # (numpy) array containing the parameters of the model
    # has to be initialized by the method 'initialize'
    theta = None
    
    # dictionary containing all possible features of a corpus and their corresponding index;
    # has to be set by the method 'initialize'; hint: use a Python dictionary
    feature_indices = None
    
    # set containing a list of possible lables
    # has to be set by the method 'initialize'
    labels = None
    
    
    # Exercise 1 a) ###################################################################
    def initialize(self, corpus):
        '''
        Initialize the maximun entropy model, i.e., build the set of all features, the set of all labels
        and create an initial array 'theta' for the parameters of the model.
        Parameters: corpus: list of list representing the corpus, returned by the function 'import_corpus'
        '''
        self.corpus = corpus
        words = []
        labels = []
        for f in corpus:
            for (a,b) in f:
                words.append(a)
                labels.append(b)
        words = set(words)
        self.labels = set(labels)
        #I create a dictionary with every feature
        #I've to create a dictionary of dictionary
        #otherwise every word has associated only one label
        #In the dictionary we have all possible feature
        #but we have to understand where is 1 or 0
        feature = {}
        for i in words:
            for j in labels:
                #Every word with every label
                feature[i] = {j : 1}
                #Every label with itself
                feature[j] = {j : 1}
                #Every label with the next
                if j != labels[0]:
                    feature[last_label] = {j : 1}
                #I save the last label
                last_label = j
        print(feature)
        n_feature = 0
        for i in feature.keys():
            n_feature = sum(feature[i].values()) + n_feature
            print(n_feature)
        print(n_feature)
        self.theta = [[1]*n_feature]
        print(self.theta)
        return True
    
    
    
    
    # Exercise 1 b) ###################################################################
    def get_active_features(self, word, label, prev_label):
        '''
        Compute the vector of active features.
        Parameters: word: string; a word at some position i of a given sentence
                    label: string; a label assigned to the given word
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing only zeros and ones.
        '''
        
        # your code here
        
        pass
        



    # Exercise 2 a) ###################################################################
    def cond_normalization_factor(self, word, prev_label):
        '''
        Compute the normalization factor 1/Z(x_i).
        Parameters: word: string; a word x_i at some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: float
        '''
        
        # your code here
        
        pass
    
    
    
    
    # Exercise 2 b) ###################################################################
    def conditional_probability(self, label, word, prev_label):
        '''
        Compute the conditional probability of a label given a word x_i.
        Parameters: label: string; we are interested in the conditional probability of this label
                    word: string; a word x_i some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: float
        '''
        
        # your code here    
    
    
    
    
    # Exercise 3 a) ###################################################################
    def empirical_feature_count(self, word, label, prev_label):
        '''
        Compute the empirical feature count given a word, the actual label of this word and the label of the previous word.
        Parameters: word: string; a word x_i some position i of a given sentence
                    label: string; the actual label of the given word
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing the empirical feature count
        '''
        
        # your code here
        
        pass
    
    
    
    
    # Exercise 3 b) ###################################################################
    def expected_feature_count(self, word, prev_label):
        '''
        Compute the expected feature count given a word, the label of the previous word and the parameters of the current model
        (see variable theta)
        Parameters: word: string; a word x_i some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: (numpy) array containing the expected feature count
        '''
        
        # your code here
        
        pass
    
    
    
    
    # Exercise 4 a) ###################################################################
    def parameter_update(self, word, label, prev_label, learning_rate):
        '''
        Do one learning step.
        Parameters: word: string; a randomly selected word x_i at some position i of a given sentence
                    label: string; the actual label of the selected word
                    prev_label: string; the label of the word at position i-1
                    learning_rate: float
        '''
        
        # your code here
        
        pass
    
    
    
    
    # Exercise 4 b) ###################################################################
    def train(self, number_iterations, learning_rate=0.1):
        '''
        Implement the training procedure.
        Parameters: number_iterations: int; number of parameter updates to do
                    learning_rate: float
        '''
        
        # your code here
        
        pass
    
    
    
    
    # Exercise 4 c) ###################################################################
    def predict(self, word, prev_label):
        '''
        Predict the most probable label of the word referenced by 'word'
        Parameters: word: string; a word x_i at some position i of a given sentence
                    prev_label: string; the label of the word at position i-1
        Returns: string; most probable label
        '''
        
        # your code here
        
        pass
    
if __name__ == '__main__':
    corpus = import_corpus("corpus_pos.txt")
    prova = MaxEntModel()
    prova.initialize(corpus)

