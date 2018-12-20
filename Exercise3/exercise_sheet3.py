################################################################################
## SNLP exercise sheet 3
################################################################################
import math
import sys
import random
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

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
    '''
    Initialize the maximun entropy model, i.e., build the set of all features, the set of all labels
    and create an initial array 'theta' for the parameters of the model.
    Parameters: corpus: list of list representing the corpus, returned by the function 'import_corpus'
    '''
    def initialize(self, corpus):
        self.corpus = corpus
        words = []
        self.labels = []
        for sentence in corpus:
            for (a,b) in sentence:
                words.append(a)
                self.labels.append(b)
        words = set(words)
        self.labels = set(self.labels)
        self.feature_indices = {}
        list_feature = []
        list_labels = list(self.labels)
        for label1 in list_labels:
            if ('start',label1) not in list_feature:
                list_feature.append(('start',label1))
            for label2 in list_labels:
                if (label1,label2) not in list_feature:
                    list_feature.append((label1,label2))
        for word in words:
            if (word,'start') not in list_feature:
                list_feature.append((word,'start'))
            for label in list_labels:
                if (word,label) not in list_feature:
                    list_feature.append((word,label))
        for i in range(list_feature.__len__()):
            self.feature_indices[list_feature[i]] = i
        n_feature = self.feature_indices.__len__()
        #I create the theta based on the number of features
        self.theta = np.array([1]*n_feature)
    

    '''
    Compute the vector of active features.
    Parameters: word: string; a word at some position i of a given sentence
                label: string; a label assigned to the given word
                prev_label: string; the label of the word at position i-1
    Returns: (numpy) array containing only zeros and ones.
    '''
    # Exercise 1 b) ###################################################################
    def get_active_features(self, word, label, prev_label):
        index_feature1 = 0
        index_feature2 = 0
        #The actives feature can be 2 for every word:
        #1.For the word with that label
        #2.The prev_label and the follow
        #I search the word inside the dict of feature
        if (word,label) in self.feature_indices.keys():
            index_feature1 = self.feature_indices.get((word,label))
        if (prev_label,label) in self.feature_indices.keys():
            index_feature2 = self.feature_indices.get((prev_label,label))
        #I create an array with all zeros with the same shape of theta
        active_feature = np.zeros(self.theta.__len__())
        #I trasform the two feature active features to 1 inside the vector
        if index_feature1 != 0:
            active_feature[index_feature1] = 1
        if index_feature2 != 0:
            active_feature[index_feature2] = 1
        return active_feature

    ''' 
    Compute the normalization factor 1/Z(x_i).
    Parameters: word: string; a word x_i at some position i of a given sentence
                prev_label: string; the label of the word at position i-1
    Returns: float
    '''
    # Exercise 2 a) ###################################################################
    #pERCHé ABBIAMO ANCHE IL LABEL PRECEDENTE?
    #Abbiamo il prev_label perché cerchiamo le feature a 1 con tutti i label per la parola ed il
    #label precedente
    #Perché nell'esempio sulle slide non sono considerate le feature con i due label uno dietro all'altro?
    def cond_normalization_factor(self, word, prev_label):
        Z = np.float()
        tot_active_feature = np.float()
        #Uso la lista delle label per trovare tutte le feature a 1
        for x in self.labels:
            #Chiamo la funzione che mi ritorna l'array con le feature a 1
            q = self.get_active_features(word,x,prev_label)
            w = np.count_nonzero(q)
            tot_active_feature += w
        Z = np.exp(tot_active_feature)
        return Z
    
    
    
    '''
    Compute the conditional probability of a label given a word x_i.
    Parameters: label: string; we are interested in the conditional probability of this label
                word: string; a word x_i some position i of a given sentence
                prev_label: string; the label of the word at position i-1
    Returns: float
    '''
    # Exercise 2 b) ###################################################################
    def conditional_probability(self, label, word, prev_label):
        Z = self.cond_normalization_factor(word,prev_label)
        prob = np.float()
        q = self.get_active_features(word,label,prev_label)
        w = np.count_nonzero(q)
        prob = (1/Z)*np.exp(w)
        return prob
    
    
    '''
    Compute the empirical feature count given a word, the actual label of this word and the label of the previous word.
    Parameters: word: string; a word x_i some position i of a given sentence
                label: string; the actual label of the given word
                prev_label: string; the label of the word at position i-1
    Returns: (numpy) array containing the empirical feature count
    '''
    # Exercise 3 a) ###################################################################
    def empirical_feature_count(self, word, label, prev_label):
        return self.get_active_features(word,label,prev_label)


    '''
    Compute the expected feature count given a word, the label of the previous word and the parameters of the current model
    (see variable theta)
    Parameters: word: string; a word x_i some position i of a given sentence
                prev_label: string; the label of the word at position i-1
    Returns: (numpy) array containing the expected feature count
    '''
    # Exercise 3 b) ###################################################################
    def expected_feature_count(self, word, prev_label):
        expected_feature = np.zeros(self.feature_indices.__len__())
        for (x,y) in self.feature_indices.keys():
            for label in self.labels:
                prob = self.conditional_probability(label,word,prev_label)
                act_feature = self.get_active_features(word,label,prev_label)
                feat_index = self.feature_indices.get((x,y))
                expected_feature[feat_index] += prob*act_feature[feat_index]
        return expected_feature

    
    
    '''
    Do one learning step.
    Parameters: word: string; a randomly selected word x_i at some position i of a given sentence
                label: string; the actual label of the selected word
                prev_label: string; the label of the word at position i-1
                learning_rate: float
    '''
    # Exercise 4 a) ###################################################################
    def parameter_update(self, word, label, prev_label, learning_rate):
        theta_zero = self.expected_feature_count(word,prev_label)
        empirical = self.empirical_feature_count(word,label,prev_label)
        gradient = np.subtract(empirical,theta_zero)
        self.theta = theta_zero + learning_rate*gradient



    '''
    Implement the training procedure.
    Parameters: number_iterations: int; number of parameter updates to do
                learning_rate: float
    '''
    # Exercise 4 b) ###################################################################
    def train(self, number_iterations, learning_rate=0.1):
        #I execute number_iterations time the train
        for i in range(number_iterations):
            for sentence in self.corpus:
                for (word,label) in sentence:
                    if (word,label) == sentence[0]:
                        prev_label = 'start'
                    else:
                        prev_label = label
                    #I call parameter_update for updating parameters
                    self.parameter_update(word,label,prev_label,learning_rate)


    '''
    Predict the most probable label of the word referenced by 'word'
    Parameters: word: string; a word x_i at some position i of a given sentence
                prev_label: string; the label of the word at position i-1
    Returns: string; most probable label
    '''
    # Exercise 4 c) ###################################################################
    def predict(self, word, prev_label):
        index_max = np.argmax(self.theta)
        label = str()
        for (x,y) in self.feature_indices.keys():
            if self.feature_indices.get((x,y)) == index_max:
                label = y
        return label

    '''
    Predict the empirical feature count for a set of sentences
    Parameters: sentences: list; a list of sentences; should be a sublist of the list returned by 'import_corpus'
    Returns: (numpy) array containing the empirical feature count
    '''
    # Exercise 5 a) ###################################################################
    def empirical_feature_count_batch(self, sentences):
        empirical_batch = np.array()
        prev_label = None
        for sentence in sentences:
            for (word, label) in sentence:
                if (word, label) == sentence[0]:
                    prev_label = 'start'
                np.append(empirical_batch,self.empirical_feature_count(word,label,prev_label))
                prev_label = word
        return empirical_batch


    '''
    Predict the expected feature count for a set of sentences
    Parameters: sentences: list; a list of sentences; should be a sublist of the list returned by 'import_corpus'
    Returns: (numpy) array containing the expected feature count
    '''
    # Exercise 5 a) ###################################################################
    def expected_feature_count_batch(self, sentences):
        expected_batch = np.array()
        prev_label = None
        for sentence in sentences:
            for (word,label) in sentence:
                if (word,label) == sentence[0]:
                    prev_label = 'start'
                np.append(expected_batch,self.expected_feature_count(word, prev_label))
                prev_label = word
        return expected_batch



    '''
    Implement the training procedure which uses 'batch_size' sentences from to training corpus
    to compute the gradient.
    Parameters: number_iterations: int; number of parameter updates to do
                batch_size: int; number of sentences to use in each iteration
                learning_rate: float
    '''
    # Exercise 5 b) ###################################################################
    def train_batch(self, number_iterations, batch_size, learning_rate=0.1):
        #I create a new list new_corpus for containing the sentences selected randomly from the corpus
        # based on the number batch_size
        new_corpus = []
        for i in range(batch_size):
            y = np.random.choice(self.corpus)
            if y not in new_corpus:
                new_corpus.append(y)
        #I execute number_iterations times the train
        for i in range(number_iterations):
            for sentence in new_corpus:
                for (word,label) in sentence:
                    if (word,label) == sentence[0]:
                        prev_label = 'start'
                    else:
                        prev_label = label
                    #I call parameter_update for updating parameters
                    self.parameter_update(word,label,prev_label,learning_rate)

    '''
    Compare the training methods 'train' and 'train_batch' in terms of convergence rate
    Parameters: corpus: list of list; a corpus returned by 'import_corpus'
    '''
    # Exercise 5 c) ###################################################################
    def evaluate(corpus):
        testSetLength = corpus
        testSet = random.sample(corpus, testSetLength)

        trainingSet = corpus.copy()
        for elem in testSet:
            trainingSet.remove(elem)
        A = MaxEntModel()
        B = MaxEntModel()
        A.initialize(trainingSet)
        B.initialize(trainingSet)
        A.train(100, 0.2)
        B.train_batch(100, 1, 0.2)





    
if __name__ == '__main__':
    #corpus = import_corpus("corpus_pos.txt")
    corpus = import_corpus("prova.txt")
    model = MaxEntModel()
    model.initialize(corpus)
    model.train(1)
    most_prob_label = {}
    prev_label = None
    for sentence in model.corpus:
            for (word,label) in sentence:
                if (word,label) == sentence[0]:
                    prev_label = 'start'
                prob_label = model.predict(word,prev_label)
                most_prob_label[word] = prob_label
                prev_label = label
    model.evaluate()
