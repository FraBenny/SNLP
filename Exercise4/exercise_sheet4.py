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
        #L'array è scorribile come se fosse una lista?
        self.labels = np.array(self.labels)
        for label1 in self.labels:
            if ('start',label1) not in list_feature:
                list_feature.add(('start',label1))
            for label2 in self.labels:
                if (label1,label2) not in list_feature:
                    list_feature.add((label1,label2))
        for word in words:
            if (word,'start') not in list_feature:
                list_feature.add((word,'start'))
            for label in self.labels:
                if (word,label) not in list_feature:
                    list_feature.add((word,label))
        for i in range(list_feature.__len__()):
            self.feature_indices[list_feature[i]] = i
        n_feature = self.feature_indices.__len__()
        #I create the theta based on the number of features
        self.theta = np.array([1]*n_feature)


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
        prev_label = None
        #forw_var[0] = factor[0]
        #factor = self.get_factor_forward_var(sentence)
        #Devo ciclare sul label e poi dopo anche sugli altri, perciò parola e prev_label rimangono uguali
        for i in range(0, n-1, 1):
            for (word,label2) in sentence:
                if (word,label2) == sentence[0]:
                    prev_label = 'start'
                for label in self.labels:
                    if prev_label == 'start':
                        self.theta = self.get_active_features(word,label,prev_label)
                        sum = np.sum(self.theta)
                        factor[i] = np.exp(sum)
                        tmp_array = np.append(factor[i])
                    else:
                        column = 0
                        self.theta = self.get_active_features(word,label,prev_label)
                        #La somma rappresenta unicamente le feature attive perciò è come se moltiplicassi per la feature attiva
                        sum = np.sum(self.theta)
                        factor[i] = np.exp(sum)
                        tmp_array = np.append(factor[i]*forw_var[i-1][column])
                        column += 1
                prev_label = label2
            forw_var = np.vstack([forw_var,tmp_array])
            tmp_array = np.empty()
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
        La somma risultante penso possa essere al massimo 2 perché si devono verificare in contemporanea che valga la
        feature con word e label, che tra i due label.
        Il factor deve essere legato alla parola, nel senso che deve esserci un factor per parola
        '''
        n = len(sentence)
        #fare la print per vedere se il numero è giusto
        print(n)
        factor = np.array()
        tmp_array = np.array()
        back_var = np.matrix()
        ones = np.ones(n)
        #Vedere se funziona l'assegnamento dell'ultima riga della matrice
        back_var[n] = ones
        prev_label = None
        for i in range(n-1, 1, -1):
            for (word,label) in sentence:
                if i == 0:
                    prev_label = 'start'
                    self.theta = self.get_active_features(word,label,prev_label)
                    #La somma rappresenta unicamente le feature attive perciò è come se moltiplicassi per la feature attiva
                    sum = np.sum(self.theta)
                    factor[i] = np.exp(sum)
                    tmp_array = np.append(factor[i])
                else:
                    for prev_label in self.labels:
                        #Controllare se il valore di column è corretto
                        column = 0
                        self.theta = self.get_active_features(word,label,prev_label)
                        #La somma rappresenta unicamente le feature attive perciò è come se moltiplicassi per la feature attiva
                        sum = np.sum(self.theta)
                        factor[i] = np.exp(sum)
                        tmp_array = np.append(factor[i]*forw_var[i-1][column])
                        column += 1
            forw_var = np.vstack([forw_var,tmp_array])
            tmp_array = np.empty()
        return back_var
        
        
        
    
    # Exercise 1 b) ###################################################################
    def compute_z(self, sentence):
        '''
        Compute the partition function Z(x).
        Parameters: sentence: list of strings representing a sentence.
        Returns: float;
        '''
        for_matrix = forward_variables(sentence)
        last_row = len(sentence) - 1
        Z = np.sum(for_matrix[last_row])
        return Z
        
        
    # Exercise 1 c) ###################################################################
    def marginal_probability(self, sentence, y_t, y_t_minus_one,t):
        '''
        Compute the marginal probability of the labels given by y_t and y_t_minus_one given a sentence.
        Parameters: sentence: list of strings representing a sentence.
                    y_t: element of the set 'self.labels'; label assigned to the word at position t
                    y_t_minus_one: element of the set 'self.labels'; label assigned to the word at position t-1
                    t: int; position of the word the label y_t is assigned to
        Returns: float: probability;
        '''
        #Calcolus forward and backward matrices
        for_matrix = forward_variables(sentence)
        back_matrix = backward_variables(sentence)
        #I've to take the column with the label y_t from the forward matrix and the column with the label y_t-1 from the
        #backward matrix
        #Per sapere quale valore prendere dalle matrice back e for devo sapere l'ordine dei label, che quindi deve
        # essere fisso per far questo posso, al posto di ricordare i label come set lo ricordo come lista e vado a
        # prendere la posizione
        #I take the index of the labels for knowing where take values from the two matrices
        label_index = np.where(a == y_t)
        prev_label_index = np.where(a == y_t_minus_one)
        #I take the two values from the two matrices
        back_value = back_matrix[t][label_index]
        for_value = for_matrix[t-1][prev_label_index]
        #I take the word at position t from the sentence
        (word,label) = sentence[t]
        #I compute the partition function
        Z = compute_z(sentence)
        #I compute theta for obtaining the factor
        self.theta = self.get_active_features(word,y_t,y_t_minus_one)
        sum = np.sum(self.theta)
        factor[i] = np.exp(sum)
        #Finally I compute the marginal_probability
        marg_prob = (for_matrix[column]*factor[i]*back_matrix[column])/Z
        return marg_prob

    
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
        for i in range(number_iterations):

        # your code here
        
        pass
    
    

    
    
    
    
    # Exercise 2 ###################################################################
    def most_likely_label_sequence(self, sentence):
        '''
        Compute the most likely sequence of labels for the words in a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: list of labels; each label is an element of the set 'self.labels'
        '''
        
        # your code here
        
        pass

    
