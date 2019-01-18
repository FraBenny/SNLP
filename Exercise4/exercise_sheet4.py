################################################################################
## SNLP exercise sheet 4
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
        sentence.append((parts[0], parts[-1]))

    f.close()
    return sentences




class LinearChainCRF(object):
    # training corpus
    corpus = None
    
    # (numpy) array containing the parameters of the model
    # has to be initialized by the method 'initialize'
    theta = None
    
    # set containing all features observed in the corpus 'self.corpus'
    # choose an appropriate data structure for representing features
    # each element of this set has to be assigned to exactly one component of the vector 'self.theta'
    features = None
    
    # set containing all lables observed in the corpus 'self.corpus'
    labels = None
    
    
    def initialize(self, corpus):
        '''
        build set two sets 'self.features' and 'self.labels'
        '''
        self.corpus = corpus
        words = set()
        self.labels = set()
        for sentence in corpus:
            for (a,b) in sentence:
                words.add(a)
                self.labels.add(b)
        self.feature_indices = {}
        #credo che list_feature non possa essere un set
        list_feature = set()
        list_labels = list(self.labels)
        for label1 in list_labels:
            if ('start',label1) not in list_feature:
                list_feature.add(('start',label1))
            for label2 in list_labels:
                if (label1,label2) not in list_feature:
                    list_feature.add((label1,label2))
        for word in words:
            if (word,'start') not in list_feature:
                list_feature.add((word,'start'))
            for label in list_labels:
                if (word,label) not in list_feature:
                    list_feature.add((word,label))
        for i in range(list_feature.__len__()):
            self.feature_indices[list_feature[i]] = i
        n_feature = self.feature_indices.__len__()
        #I create the theta based on the number of features
        self.theta = np.array([1]*n_feature)

    '''
    Va modificato perché vanno considerate in contemporanea la word, con label che label con label precedente
    '''
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
        active_feature = set(active_feature)
        return active_feature
        '''
    def get_factor_forward_var(self,sentence):
        n = len(sentence)
        factor = []
        list_labels = list(self.labels)
        list_labels.append('start')
        for i in range(n-1, 1, -1):
            #In teoria dovrei scorrere la sentence al contrario per il backward
            for (word,label) in sentence:
                for prev_label in list_labels:
                    '''
                    if (word,label) == sentence[0]:
                        prev_label = 'start'
                    else:
                        prev_label = label
                    '''
                    self.theta = self.get_active_features(word,label,prev_label)
                    #La somma rappresenta unicamente le feature attive perciò è come se moltiplicassi per la feature attiva
                    sum = np.sum(self.theta)
                    factor[i] = np.exp(sum)
        return factor

    def get_factor_backward_var(self,sentence):
        n = len(sentence)
        factor = []
        list_labels = list(self.labels)
        list_labels.append('start')
        for i in range(n-1, 1, -1):
            #In teoria dovrei scorrere la sentence al contrario per il backward
            for (word,label2) in sentence:
                if (word,label) == sentence[0]:
                    prev_label = 'start'
                else:
                    prev_label = label
                for label in list_labels:
                    self.theta = self.get_active_features(word,label,prev_label)
                    #La somma rappresenta unicamente le feature attive perciò è come se moltiplicassi per la feature attiva
                    sum = np.sum(self.theta)
                    factor[i] = np.exp(sum)
        return factor

        for i in range(n-1, 1, -1):
            #In teoria dovrei scorrere la sentence al contrario per il backward
            for (word,label) in sentence:
                    if (word,label) == sentence[0]:
                        prev_label = 'start'
                    else:
                        prev_label = label

            self.theta = self.get_active_features(word,label,prev_label)
            #La somma rappresenta unicamente le feature attive perciò è come se moltiplicassi per la feature attiva
            sum = np.sum(self.theta)
            factor[i] = np.exp(sum)

'''
    # Exercise 1 a) ###################################################################
    def forward_variables(self, sentence):
        '''
        Compute the forward variables for a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: data structure containing the matrix of forward variables
        '''
        n = len(sentence)
        factor = np.array()
        tmp_array = np.array()
        forw_var = np.matrix()
        forw_var[0] = factor[0]
        factor = self.get_factor_forward_var(sentence)
        for i in range(n-1, 1, -1):
            for (word,label) in sentence:
                for prev_label in list_labels:
                    column = 0
                    self.theta = self.get_active_features(word,label,prev_label)
                    #La somma rappresenta unicamente le feature attive perciò è come se moltiplicassi per la feature attiva
                    sum = np.sum(self.theta)
                    factor[i] = np.exp(sum)
                    tmp_array = np.append(factor[i]*forw_var[i-1][column])
                    column += 1
            #form_var has to be a matrix based on labels
            forw_var = np.vstack([forw_var,tmp_array])
        return forw_var
        
        
    def backward_variables(self, sentence):
        '''
        Compute the backward variables for a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: data structure containing the matrix of backward variables
        '''
        '''
        Il factor rappresenta l'esponenziale della somma di ogni feature moltiplicata per theta che rappresenta perciò 
        il vettore delle feature attive.
        Visto che l'indice di theta va di pari passo con quello delle feature il prodotto sarà 1 solo se la feature è
        attiva.
        La somma risultante penso possa essere al massimo 1 perché si devono verificare in contemporanea che valga la
        feature con word e label, che tra i due label.
        Diamo per scontato di aver cambiato le feature e che siano fatte da due label più la parola
        Il factor deve essere legato alla parola, nel senso che deve esserci un factor per parola
        '''
        n = len(sentence)
        #fare la print per vedere se il numero è giusto
        print(n)
        factor = np.array()
        tmp_array = np.array()
        back_var = np.matrix()
        back_var[n] = 1
        prev_label = None
        for i in range(n-1, 1, -1):
            #In teoria dovrei scorrere la sentence al contrario per il backward
            for (word,label2) in sentence:
                if (word,label2) == sentence[0]:
                    prev_label = 'start'
                for label in list_labels:
                    column = 0
                    self.theta = self.get_active_features(word,label,prev_label)
                    #La somma rappresenta unicamente le feature attive perciò è come se moltiplicassi per la feature attiva
                    sum = np.sum(self.theta)
                    factor[i] = np.exp(sum)
                    #back_var è una matrice perciò devo ciclare anche su quella, sulla colonna, cioè sulle label
                    tmp_array = np.append(factor[i]*back_var[i-1][column])
                    column += 1
                prev_label = label2
            #back_var has to be a matrix based on labels
            back_var = np.vstack([back_var,tmp_array])
        return back_var
        
        
        
    
    # Exercise 1 b) ###################################################################
    def compute_z(self, sentence):
        '''
        Compute the partition function Z(x).
        Parameters: sentence: list of strings representing a sentence.
        Returns: float;
        '''
        
        
        # your code here
        
        pass
        
        
        
            
    # Exercise 1 c) ###################################################################
    def marginal_probability(self, sentence, y_t, y_t_minus_one):
        '''
        Compute the marginal probability of the labels given by y_t and y_t_minus_one given a sentence.
        Parameters: sentence: list of strings representing a sentence.
                    y_t: element of the set 'self.labels'; label assigned to the word at position t
                    y_t_minus_one: element of the set 'self.labels'; label assigned to the word at position t-1
        Returns: float: probability;
        '''
        
        # your code here
        
        pass
    
    
    
    
    # Exercise 1 d) ###################################################################
    def expected_feature_count(self, sentence, feature):
        '''
        Compute the expected feature count for the feature referenced by 'feature'
        Parameters: sentence: list of strings representing a sentence.
                    feature: a feature; element of the set 'self.features'
        Returns: float;
        '''
        
        # your code here
        
        pass
    
    
    
    
    
    # Exercise 1 e) ###################################################################
    def train(self, num_iterations, learning_rate=0.01):
        '''
        Method for training the CRF.
        Parameters: num_iterations: int; number of training iterations
                    learning_rate: float
        '''
        
        # your code here
        
        pass
    
    

    
    
    
    
    # Exercise 2 ###################################################################
    def most_likely_label_sequence(self, sentence):
        '''
        Compute the most likely sequence of labels for the words in a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: list of lables; each label is an element of the set 'self.labels'
        '''
        
        # your code here
        
        pass

    
