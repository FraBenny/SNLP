################################################################################
## SNLP exercise sheet 2
################################################################################

import operator
import numpy as np
'''
This function can be used for importing the corpus.
Parameters: path_to_file: string; path to the file containing the corpus
Returns: list of list; the second layer list contains tuples (token,label);
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

def divide_in_training_and_test(corpus):
    test_corpus = corpus.pop()
    return (corpus,test_corpus)

#Aggiungere gli unknown senza tenere in considerazione l'ordine, poi per l'unica a cui interessa l'ordine che
# è il transitional uso quello vecchio tanto mi interessano solo i tag
#L'importante è usare sempre quello con una stringa in meno
def preprocess(corpus):
    unknown_tokens = []
    corpus_dict = {}
    #Every f is a sentence, a list of tuple
    #I've to take the corpus and transform it in a list
    #I controll if is a list of list(corpus_training) or a simple list(corpus_test)
    if any(isinstance(el, list) for el in corpus):
        for f in corpus:
            for i in f:
                ##I create a dictionary with tags and number of occurrences
                (a, b) = i
                a = a.lower()
                if (a,b) in corpus_dict.keys():
                    corpus_dict[(a,b)] += 1
                else:
                    corpus_dict[(a,b)] = 1
    else:
        for i in corpus:
            (a, b) = i
            a = a.lower()
            if (a,b) in corpus_dict.keys():
                corpus_dict[(a,b)] += 1
            else:
                corpus_dict[(a,b)] = 1
    #I create a list with all tokens which occur one time
    for k,v in corpus_dict.items():
        if v == 1:
            unknown_tokens.append(k[0])
    if any(isinstance(el, list) for el in corpus):
        for f in corpus:
            temp = f
            new_list = []
            value = False
            for i in f:
                for x in unknown_tokens:
                    if i[0].lower() == x:
                        value = True
                        (a,b) = i
                        new_list.append(('unknown',b))
            #If I have found an unknown value i remove all the list and put the new one
            if value:
                corpus.remove(temp)
                corpus.append(new_list)
    else:
        for i in corpus:
            for x in unknown_tokens:
                if i[0].lower() == x:
                    (a,b) = i
                    corpus.remove(i)
                    corpus.append(('unknown',b))
    return corpus




# Exercise 1 ###################################################################
'''
Implement the probability distribution of the initial states.
Parameters:	state: string
            internal_representation: data structure representing the parametrization of this probability distribution;
                this data structure is returned by the function estimate_initial_state_probabilities
Returns: float; initial probability of the given state
'''
def initial_state_probabilities(state, internal_representation):
    return internal_representation.get(state, 0)
    
    
    
'''
Implement the matrix of transition probabilities.
Parameters:	from_state: string;
            to_state: string;
            internal_representation: data structure representing the parameterization of the matrix of transition probabilities;
                this data structure is returned by the function estimate_transition_probabilities
Returns: float; probability of transition from_state -> to_state
'''
def transition_probabilities(from_state, to_state, internal_representation):
    return internal_representation[from_state].get(to_state, 0)
    
    
    
    
'''
Implement the matrix of emmision probabilities.
Parameters:	state: string;
            emission_symbol: string;
            internal_representation: data structure representing the parameterization of the matrix of emission probabilities;
                this data structure is returned by the function estimate_emission_probabilities
Returns: float; emission probability of the symbol emission_symbol if the current state is state
'''
def emission_probabilities(state, emission_symbol, internal_representation):
    return internal_representation[state].get(emission_symbol, 0)
    
    
    
    
'''
Implement a function for estimating the parameters of the probability distribution of the initial states.
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the probability distribution of the initial states;
            use this data structure for the argument internal_representation of the function initial_state_probabilities
'''
def estimate_initial_state_probabilities(corpus):
    initial_tags = {}
    #Take only first word of a sentence
    for f in corpus:
        #For every sentence I take only the first token and his tag
        (a,b) = f[0]
        #I create a dictionary with initial tags and number of occurrences
        if b in initial_tags.keys():
            initial_tags[b] += 1
        else:
            initial_tags[b] = 1
    #I take the sum of all occurrences
    tot = sum(initial_tags.values())
    #I iterate on items of the dictionary
    for k,v in initial_tags.items():
        #I calculate the probability of every tag
        prob = v/tot
        #I update the dictionary with the probability
        initial_tags[k] = prob
    return initial_tags

    
'''
Implement a function for estimating the parameters of the matrix of transition probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of transition probabilities;
            use this data structure for the argument internal_representation of the function transition_probabilities
'''
def estimate_transition_probabilities(corpus):
    transition_tags = {}
    for f in corpus:
        #For every sentence I take the token and the tag of one and his next, two tuple a time
        for i in range(len(f)-1):
            (a, b) = f[i]
            (c, d) = f[i+1]
            #I create a dictionary with tag and his following and number of occurrences
            if b not in transition_tags.keys():
                transition_tags[b] = {d : 1}
            else:
                if d not in transition_tags[b].keys():
                    transition_tags[b][d] = 1
                else:
                    transition_tags[b][d] += 1
    #I take the sum of all occurernces
    sum_for_tag = {}
    for k,v in transition_tags.items():
        tot = sum(transition_tags[k].values())
        sum_for_tag[k] = tot
    for k,v in transition_tags.items():
        #I calculate tot one time for every k and remain the same for every tag that follow it
        tot = sum_for_tag.get(k)
        for k2,v2 in transition_tags[k].items():
            #I calculate the probability that the token follow is follow by the other
            prob = v2/tot
            #I update the dictionary with the probability
            transition_tags[k][k2] = prob
    return transition_tags

    
'''
Implement a function for estimating the parameters of the matrix of emission probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of emission probabilities;
            use this data structure for the argument internal_representation of the function emission_probabilities
'''
def estimate_emission_probabilities(corpus):
    emission_tags = {}
    for f in corpus:
        for i in f:
            (a, b) = i
            #I create a dictionary with tag and his token with number of occurrences
            if b not in emission_tags.keys():
                emission_tags[b] = {a.lower() : 1}
            else:
                if a.lower() not in emission_tags[b].keys():
                    emission_tags[b][a.lower()] = 1
                else:
                    emission_tags[b][a.lower()] += 1
    #I iterate on items of the dictionary
    sum_for_tag = {}
    for k,v in emission_tags.items():
        tot = sum(emission_tags[k].values())
        sum_for_tag[k] = tot
    for k,v in emission_tags.items():
        #I calculate 'tot' one time for every k and remain the same for every token link to that
        tot = sum_for_tag.get(k)
        for k2,v2 in emission_tags[k].items():
            #I calculate the probability that a tag is linked to that token
            prob = v2/tot
            #I update the dictionary with the probability
            emission_tags[k][k2] = prob
    return emission_tags

    
# Exercise 2 ###################################################################
''''
Implement the Viterbi algorithm for computing the most likely state sequence given a sequence of observed symbols.
Parameters: observed_symbols: list of strings; the sequence of observed symbols
            initial_state_probabilities_parameters: data structure containing the parameters of the probability distribution of the initial states, returned by estimate_initial_state_probabilities
            transition_probabilities_parameters: data structure containing the parameters of the matrix of transition probabilities, returned by estimate_transition_probabilities
            emission_probabilities_parameters: data structure containing the parameters of the matrix of emission probabilities, returned by estimate_emission_probabilities
Returns: list of strings; the most likely state sequence
'''
def most_likely_state_sequence(observed_symbols, initial_state_probabilities_parameters, transition_probabilities_parameters, emission_probabilities_parameters):
    state_token_matrix = []
    most_lky_sqn = []
    delta_values = []
    temp_delta_values = []
    states = list(transition_probabilities_parameters.keys())
    for i in range(states.__len__()):
        state_token_matrix.append([])
        for j in range(observed_symbols.__len__()):
            in_value = initial_state_probabilities(states[i],initial_state_probabilities_parameters)
            #print(in_value)
            em_value = emission_probabilities(states[i],observed_symbols[j],emission_probabilities_parameters)
            #print(em_value)
            if j == 0:
                state_token_matrix[i].append(in_value*em_value)
                delta_values.append(state_token_matrix[i][j])
            else:
                tr_value = transition_probabilities(states[i-1],states[i],transition_probabilities_parameters)
                for x in range(states.__len__()):
                    tmp =
                delta_values = delta_values[i-1]
                state_token_matrix[i].append(max(state_token_matrix[i-1])*tr_value*em_value)

    #Estrarre il massimo dalla colonna o riga in base a come decido di implementarlo
    #Serve per ogni parola lo stato con probabilità massima andando in backward
    #Trovato il massimo, ritorno l'indice e con questo cerco lo stato corrispondente
    #Potrei tornare direttamente l'indice visto che il valore del massimo non ci interessa
    #val_max = []
    state_token_matrix = np.asarray(state_token_matrix)
    state_token_matrix_T = state_token_matrix.T
    for i in range(len(state_token_matrix.T)):
        index_max = np.argmax(state_token_matrix_T[i])
        most_lky_sqn.append(states[index_max])
    return most_lky_sqn


#For transitional and emission is better to create a nested dictionary
if __name__ == '__main__':
    corpus = import_corpus("corpus_ner.txt")
    #corpus.pop()
    (corpus_training, corpus_test) = divide_in_training_and_test(corpus)
    #Il preprocess non funziona bene, ricontrollare, sul corpus training sembra funzionare bene
    #corpus_tr_with_unknown = preprocess(corpus_training)
    #corpus_test = preprocess(corpus_test)
    in_prob = estimate_initial_state_probabilities(corpus_training)
    tr_prob = estimate_transition_probabilities(corpus_training)
    #em_prob = estimate_emission_probabilities(corpus_tr_with_unknown)
    em_prob = estimate_emission_probabilities(corpus_training)
    observed_symbols = []
    for x in corpus_test:
        (a,b) = x
        observed_symbols.append(a.lower())
    most_lky_sqn = most_likely_state_sequence(observed_symbols,in_prob,tr_prob,em_prob)
    print(most_lky_sqn)









