################################################################################
## SNLP exercise sheet 3
################################################################################
import math
import sys
import random
from builtins import range

import numpy as np

'''
DUBBI:
Ha senso considerare la label 'start' come successiva?
Vanno considerate le feature con due label una dopo l'altra nell'esercizio 2a) e 2b)?
Probabilmente devo trattare word, come possa essere la label e non per forza la parola
Perchè comunque expectation count è uno per ogni feature, in questo caso un array per ogni parola
#Dove prendere il prev_label? nel 4 b)
Es. 5b) dove prendo il prev_label quando non ho più il corpus?
Probabilmente nella maggior parte degli esercizi in cui serve prev_label mi serve il corpus
'''


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
    #print(sentences)
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
        '''#Start dovrebbe essere solo all'inizio della frase
        self.labels.add('start')'''
        #I create a dictionary with every feature
        #I've to create a dictionary of dictionary
        #otherwise every word has associated only one label
        #The values of the second dictionary is the number of the feature
        #In the dictionary we have all possible feature
        #but we have to understand where is 1 or 0
        self.feature_indices = {}
        list_labels = list(self.labels)
        prev_label = 0
        k = 1
        #print(words.__len__())
        #print(list_labels.__len__())
        for j in list_labels:
            #Every label with itself
            self.feature_indices[j] = {j : k}
            #I add start with every label next
            if 'start' not in self.feature_indices:
                self.feature_indices['start'] = {j : k}
                k += 1
            else:
                self.feature_indices['start'][j] = k
                k += 1
            #If I am in the first postìtion I'll start from the next cycle
            if j == list_labels[0]:
                #I save the last label
                prev_label = j
            else:
                #Every label with the next
                if prev_label not in self.feature_indices:
                    self.feature_indices[prev_label] = {j : k}
                    k += 1
                else:
                    self.feature_indices[prev_label][j] = k
                    k += 1
                prev_label = j
        for i in words:
            for j in list_labels:
                #I add start
                if 'start' not in self.feature_indices:
                    self.feature_indices['start'] = {j : k}
                    k += 1
                else:
                    self.feature_indices['start'][j] = k
                    k += 1
                #Every word with every label
                if i not in self.feature_indices:
                    self.feature_indices[i] = {j : k}
                    k += 1
                else:
                    self.feature_indices[i][j] = k
                    k += 1
                    #I add also the word with 'start'
                    self.feature_indices[i]['start'] = k
                    k += 1
        print(k)
        n_feature = 0
        print(self.feature_indices)
        for i in self.feature_indices:
            n_feature += (list(self.feature_indices[i].values())).__len__()
        print(n_feature)
        #I create the theta based on the number of features
        self.theta = np.array([1]*n_feature)
        return True
    

    '''
    Compute the vector of active features.
    Parameters: word: string; a word at some position i of a given sentence
                label: string; a label assigned to the given word
                prev_label: string; the label of the word at position i-1
    Returns: (numpy) array containing only zeros and ones.
    '''
    # Exercise 1 b) ###################################################################
    #Ma lo start lo dobbiamo considerare? Si
    def get_active_features(self, word, label, prev_label):
        y = 0
        w = 0
        #The actives feature can be 2 for every word:
        #1.For the word with that label
        #2.The prev_label and the follow
        #I search the word inside the dict of feature
        if word in self.feature_indices:#Va bene così o devo scrivere self.feature_indices.keys()
            x = self.feature_indices.get(word)
            #I search for that word the label and I take the number of the feature
            if label in self.feature_indices[word]:#Va bene così o devo scrivere self.feature_indices[word].keys()
                y = x.get(label)
        if prev_label in self.feature_indices:#Va bene così o devo scrivere self.feature_indices.keys()
            #I search the prev_label inside the dict of feature
            Z = self.feature_indices.get(prev_label)
            #I search the label with that prev_label and I take the number of the feature
            if label in self.feature_indices[prev_label]:#Va bene così o devo scrivere self.feature_indices[prev_label].keys()
                w = Z.get(label)
        #I create an array with all zeros with the same shape of theta
        active_feature = np.zeros(self.theta.__len__())
        #print("active_feature1")
        #print(active_feature)
        #print("theta")
        #print(self.theta.__len__())
        #I trasform the two feature active features to 1 inside the vector
        #print("y")
        #print(y)
        #print("w")
        #print(w)
        if y != 0:
            active_feature[y] = 1
        if w != 0:
            active_feature[w] = 1
        #print("active_featuref2")
        #print(active_feature)
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
            print("q")
            print(q)
            #Mi viene ritornato il numero di non zero in quell'array
            w = np.count_nonzero(q)
            '''
            #In w vanno a finire gli indici di tutti gli elementi che non sono a 0
            #A me interessa unicamente sapere quanti sono gli elementi
            #perciò se è una lista, basta sapere quanto è lunga
            w = np.nonzero(q)
            print("w2")
            print(w)
            print(type(w))
            #r rappresenta il numero di feature attive per quella combinazione
            if not np.empty(w[0]):
                print("w diverso da 0")
                w = list(w[0])
                r = w.count()
                print("r")
                print(r)'''
            #Conto il numero totale di feature attive per quella parola, ciclando sulla label
            #Conto unicamente quante sono le feature attive perché mi interessa solo sapere quante
            #sono, non quali sono
            tot_active_feature += w
        #Calcolo l'esponenziale con l'indice precedentemente calcolato
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
        #E' uguale al calcolo di Z con la differenza che non itero su tutte le label ma solo
        #su quella che mi viene passata
        Z = self.cond_normalization_factor(word,prev_label)
        prob = np.float()
        #Chiamo la funzione che mi ritorna l'array se la feature è a 1
        q = self.get_active_features(word,label,prev_label)
        #In w vanno a finire gli indici di tutti gli elementi che non sono a 0
        #A me interessa unicamente sapere quanti sono gli elementi
        #perciò se è una lista, basta sapere quanto è lunga
        w = np.count_nonzero(q)
        #Calcolo la probabilità per quella data label p(y|x), in questo caso p(label|word)
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
        for feature in self.feature_indices.keys():
            for label in self.labels:
                #Uso la funzione per il calcolo della probabilità
                prob = self.conditional_probability(label,word,prev_label)
                #Trovo tutte le feature attive per quella parola e label
                act_feature = self.get_active_features(word,label,prev_label)
                feat_index = self.feature_indices[feature]
                expected_feature[feat_index] += prob*act_feature[feat_index]
                '''
                indexes = np.nonzero(act_feature)
                #Dovrebbe tornare un array di tuple
                (feature,n_feature) = self.feature_indices.get(word)'''
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
        for x in self.feature_indices.keys():
            [label for label,v in self.feature_indices[x].items() if v == index_max]
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
            #Chiamare la funzione che calcolo l'empirical, serve word, label e prev_label
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
            #Chiamare la funzione che calcolo l'expected, serve word, label e prev_label
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
        corpus = np.random.rand(100, 5)
        indices = np.random.permutation(corpus.shape[0])
        training_idx, test_idx = indices[:80], indices[80:]
        training, test = corpus[training_idx,:], corpus[test_idx,:]
        A = MaxEntModel()
        B = MaxEntModel()
        A.initialize(training)
        B.initialize(training)
        A.train(1)
        B.train_batch(1,1)
        pass








    
if __name__ == '__main__':
    #corpus = import_corpus("corpus_pos.txt")
    corpus = import_corpus("Prova.txt")
    prova = MaxEntModel()
    prova.initialize(corpus)
    prova.get_active_features(word='formed',label='CC',prev_label='WRB')
    prova.cond_normalization_factor('formed','CC')
