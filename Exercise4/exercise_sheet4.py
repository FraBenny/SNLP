################################################################################
## SNLP exercise sheet 4
################################################################################
import math
import sys
from builtins import print

import numpy as np
import random

from numpy import format_parser

'''
Va modificato il fatto il feature_indices perché in primis si chiama features e poi deve essere un set
How much have to be the gradient?
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
            for (a, b) in sentence:
                words.add(a)
                self.labels.add(b)
        #credo che list_feature non possa essere un set
        self.features = list()
        #L'array è scorribile come se fosse una lista?
        self.labels = list(self.labels)
        for label1 in self.labels:
            if ('start',label1) not in self.features:
                self.features.append(('start',label1))
            for label2 in self.labels:
                if (label1,label2) not in self.features:
                    self.features.append((label1,label2))
        for word in words:
            if (word,'start') not in self.features:
                self.features.append((word,'start'))
            for label in self.labels:
                if (word,label) not in self.features:
                    self.features.append((word,label))
        #self.features = np.array(self.features)
        n_feature = self.features.__len__()
        #I create the theta based on the number of features
        self.theta = np.array([1]*n_feature)
        #print("Finito")

    def get_active_features(self, word, label, prev_label):
        index_feature1 = 0
        index_feature2 = 0
        #The actives feature can be 2 for every word:
        #1.For the word with that label
        #2.The prev_label and the follow
        #I search the word inside the dict of feature
        ''' Da modificare perché il dizionario non esiste più'''
        if (word,label) in self.features:
            index_feature1 = self.features.index((word,label))
        if (prev_label,label) in self.features:
            index_feature2 = self.features.index((prev_label,label))
        #I create an array with all zeros with the same shape of theta
        active_feature = np.zeros(self.theta.__len__())
        #I trasform the two feature active features to 1 inside the vector
        if index_feature1 != 0:
            active_feature[index_feature1] = 1
        if index_feature2 != 0:
            active_feature[index_feature2] = 1
        return active_feature

    def get_all_active_features(self,sentence):
        prev_label = None
        active_features = np.array()
        for (word,label) in sentence:
            if (word,label) == sentence[0]:
                prev_label = 'start'
            active_features = np.append(active_features,active_features*self.get_active_features(word,label,prev_label))
            prev_label = label
        return active_features


    # Exercise 1 a) ###################################################################
    def forward_variables(self, sentence):
        '''
        Compute the forward variables for a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: data structure containing the matrix of forward variables
        '''
        n = len(sentence)
        #factor = np.array(None)
        factor = None
        tmp_array = np.array(None)
        forw_var = {}
        prev_label = None
        row = 1
        #forw_var[0] = factor[0]
        #factor = self.get_factor_forward_var(sentence)
        #Devo ciclare sul label e poi dopo anche sugli altri, perciò parola e prev_label rimangono uguali
        for (word,label2) in sentence:
            if (word,label2) == sentence[0]:
                prev_label = 'start'
            for label in self.labels:
                if prev_label == 'start':
                    self.theta = self.get_active_features(word,label,prev_label)
                    sum = np.sum(self.theta)
                    factor = np.exp(sum)
                    #Che senso ha che factor sia una lista se poi lo aggiungo direttamente a tmp_array?
                    forw_var[(row,label)] = factor
                else:
                    self.theta = self.get_active_features(word,label,prev_label)
                    #La somma rappresenta unicamente le feature attive perciò è come se moltiplicassi per la feature attiva
                    sum = np.sum(self.theta)
                    factor = np.exp(sum)
                    forw_var[(row,label)] = (factor*forw_var.get((row-1,label)))
            prev_label = label2
            row += 1
        print(forw_var)
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
        factor = None
        tmp_array = np.array(None)
        back_var = {}
        prev_label = None
        row = n-1
        back_var[(n,'start')] = 1
        for prev_label in self.labels:
            back_var[(n,prev_label)] = 1
        for (word,label) in sentence:
            if (word,label) == sentence[0]:
                prev_label = 'start'
                self.theta = self.get_active_features(word,label,prev_label)
                #La somma rappresenta unicamente le feature attive perciò è come se moltiplicassi per la feature attiva
                sum = np.sum(self.theta)
                factor = np.exp(sum)
                back_var[(row,prev_label)] = (factor*back_var.get((row+1,prev_label)))
            else:
                for prev_label in self.labels:
                    #Controllare se il valore di column è corretto
                    self.theta = self.get_active_features(word,label,prev_label)
                    #La somma rappresenta unicamente le feature attive perciò è come se moltiplicassi per la feature attiva
                    sum = np.sum(self.theta)
                    factor = np.exp(sum)
                    back_var[(row,prev_label)] = (factor*back_var.get((row+1,prev_label)))
                row -= 1
        print(back_var)
        return back_var
        

    # Exercise 1 b) ###################################################################
    def compute_z(self, sentence):
        '''
        Compute the partition function Z(x).
        Parameters: sentence: list of strings representing a sentence.
        Returns: float;
        '''
        for_var = self.forward_variables(sentence)
        last_row = len(sentence)
        Z = 0
        for label in self.labels:
            Z += for_var.get((last_row,label))
        return Z
        
        
    # Exercise 1 c) ###################################################################
    def marginal_probability(self, sentence, y_t, y_t_minus_one,t):
        #Ricordarsi che deve essere basata sulle label contenute nella frase, non su tutte
        '''
        Compute the marginal probability of the labels given by y_t and y_t_minus_one given a sentence.
        Parameters: sentence: list of strings representing a sentence.
                    y_t: element of the set 'self.labels'; label assigned to the word at position t
                    y_t_minus_one: element of the set 'self.labels'; label assigned to the word at position t-1
                    t: int; position of the word the label y_t is assigned to
        Returns: float: probability;
        '''
        #Calcolus forward and backward matrices
        for_var = self.forward_variables(sentence)
        back_var = self.backward_variables(sentence)
        marg_prob = None
        factor = np.array(None)
        labelf_values = []
        labelb_values = []
        #I've to take the column with the label y_t from the forward matrix and the column with the label y_t-1 from the
        #backward matrix
        #Per sapere quale valore prendere dalle matrici back e for devo sapere l'ordine dei label, che quindi deve
        # essere fisso per far questo posso, al posto di ricordare i label come set lo ricordo come lista e vado a
        # prendere la posizione
        #I take the index of the labels for knowing where take values from the two matrices
        #Potrebbe essere sbagliato perché cerca l'indice nella lista dei label, ma dove sono presenti tutte
        # le label, non solo quelle della frase, perciò meglio cambiare e trasformare le due matrici in dizionari
        '''for i in range(len(sentence)):
            labelf_values.append(for_var.get((i,y_t)))
        for i in range(len(sentence),1,-1):
            labelb_values.append(back_var.get((i,y_t_minus_one)))'''
        #label_index = np.where(a == y_t)
        #prev_label_index = np.where(a == y_t_minus_one)
        #I take the two values from the two matrices
        #back_value = back_matrix[t][label_index]
        back_value = back_var.get((t,y_t))
        #for_value = for_matrix[t-1][prev_label_index]
        for_value = for_var.get((t,y_t_minus_one))
        #I take the word at position t from the sentence
        (word,label) = sentence[t-1]
        #I compute the partition function
        Z = self.compute_z(sentence)
        #I compute theta for obtaining the factor
        self.theta = self.get_active_features(word,y_t,y_t_minus_one)
        sum = np.sum(self.theta)
        factor = np.exp(sum)
        #Finally I compute the marginal_probability
        #Alla forward devo passare la label y_t o y_t_minus_one e stessa cosa per backward?
        '''print(t)
        print(y_t_minus_one)
        print(for_var.get((t-1,y_t_minus_one)))
        #print(factor)
        #print(back_var.get((t,y_t)))'''
        if y_t_minus_one == 'start':
            marg_prob = (factor*back_var.get((t,y_t)))
        else:
            marg_prob = (for_var.get((t-1,y_t_minus_one))*factor*back_var.get((t,y_t)))
        return marg_prob

    
    # Exercise 1 d) ###################################################################
    def expected_feature_count(self, sentence, feature):
        '''
        Compute the expected feature count for the feature referenced by 'feature'
        Parameters: sentence: list of strings representing a sentence.
                    feature: a feature; element of the set 'self.features'
        Returns: float;
        '''
        y_t_minus_one = None
        marg_prob = np.array(None)
        t = 1
        for (word,y_t) in sentence:
            if (word,y_t) == sentence[0]:
                    y_t_minus_one = 'start'
            marg_prob = np.append(marg_prob,self.marginal_probability(sentence,y_t,y_t_minus_one,t))
            y_t_minus_one = y_t
            t += 1
        #Intanto devo sapere se cercare un prev_label o una word e poi la devo trovare
        (a,b) = feature
        prev_label = None
        tmp_theta = np.array(None)
        if a in self.labels and b in self.labels:
            #allora devo trovare le words
            for (word,label) in sentence:
                if (word,label) == sentence[0]:
                        prev_label = 'start'
                if a == prev_label & b == label:
                    tmp_theta = self.get_active_features(word,label,prev_label)
                    self.theta = self.theta*tmp_theta
                prev_label = label
            #In theta abbiamo n array e dobbiamo sommarli tutti ma solo elemento per elemento,
            # alla fine dovrà risultare un array della lunghezza degli altri con tutte le feature attive
            #self.theta = np.sum(theta)
        else:
            #allora devo trovare la prev_label
            for (word,label) in sentence:
                if (word,label) == sentence[0]:
                        prev_label = 'start'
                if a == word and b == label:
                    tmp_theta = self.get_active_features(word,label,prev_label)
                    self.theta = self.theta*tmp_theta
                prev_label = label
            #In theta abbiamo n array e dobbiamo sommarli tutti ma solo elemento per elemento,
            # alla fine dovrà risultare un array della lunghezza degli altri con tutte le feature attive
            #self.theta = np.sum(theta)
        #eseguisco la marginal per ogni word,label e prev_label e poi eseguisco anche
        #quello che mi torna theta per vedere quali sono le feature attive e faccio la moltiplicazione
        #elemento per elemento di ogni valore
        '''print(self.features)
        print(self.theta)
        print(marg_prob)'''
        expected = np.sum(self.theta*marg_prob)
        return expected
    
    
    # Exercise 1 e) ###################################################################
    def train(self, num_iterations, learning_rate=0.01):
        '''
        Method for training the CRF.
        Parameters: num_iterations: int; number of training iterations
                    learning_rate: float
        '''
        expected_count = np.array(None)
        tmp_array = np.array(None)
        sentence = random.choice(self.corpus)
        #Calcolo l'expected come dice nelle linee guida per ogni sentence
        '''
        for sentence in self.corpus:
            for feature in self.features:
                 tmp_array = np.append(tmp_array,self.expected_feature_count(sentence,feature))
            expected_count = np.append(expected_count,tmp_array)
            tmp_array = np.empty()
        '''
        #Calcolo l'expected solo per quella frase
        for feature in self.features:
             tmp_array = np.append(tmp_array,self.expected_feature_count(sentence,feature))
        expected_count = np.append(expected_count,tmp_array)
        empirical_count = self.get_all_active_features(sentence)
        for i in range(num_iterations):
            self.theta += learning_rate*(empirical_count-expected_count)

    
    # Exercise 2 ###################################################################
    def most_likely_label_sequence(self, sentence):
        '''
        Compute the most likely sequence of labels for the words in a given sentence.
        Parameters: sentence: list of strings representing a sentence.
        Returns: list of labels; each label is an element of the set 'self.labels'
        '''
        most_lky_lbl_sqn = np.array()
        #delta è un matrice non un array, viene calcolata una per ogni label
        delta_values = np.matrix()
        psi = np.matrix()
        factor = np.array()
        delta_values = np.array()
        #for i in range(len(sentence)):
        row = 0
        for (word,label) in sentence:
            if (word,label) == sentence[0]:
                prev_label = 'start'
                for label in self.labels:
                    self.theta = self.get_active_features(word,label,prev_label)
                    #La somma rappresenta unicamente le feature attive perciò è come se moltiplicassi per la feature attiva
                    sum = np.sum(self.theta)
                    factor = np.exp(sum)
                    tmp_array = np.append(tmp_array,factor)
                delta_values = np.vstack([delta_values,tmp_array])
                tmp_array = np.empty()
            else:
                for label in self.labels:
                    #Controllare se il valore di column è corretto
                    self.theta = self.get_active_features(word,label,prev_label)
                    #La somma rappresenta unicamente le feature attive perciò è come se moltiplicassi per la feature attiva
                    sum = np.sum(self.theta)
                    factor = np.exp(sum)
                    #tmp_array = np.append(factor)
                    #Devo calcolare il prodotto tra i fattori e il delta della riga precedente, poi di questo prendere il massimo
                    tmp_array = np.append(tmp_array,np.amax(factor*delta_values[row]))
                    row += 1
                delta_values = np.vstack([delta_values,tmp_array])
                tmp_array = np.empty()

        for i in random(len(sentence)):
            index_max = np.argmax(delta_values[i])
            most_lky_lbl_sqn.append(self.labels[index_max])
        return most_lky_lbl_sqn

if __name__ == '__main__':
    corpus = import_corpus("corpus_temp.txt")
    #corpus = import_corpus("corpus_pos.txt")
    model = LinearChainCRF()
    model.initialize(corpus)
    model.train(1)

    
