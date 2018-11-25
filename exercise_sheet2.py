################################################################################
## SNLP exercise sheet 2
################################################################################

import operator
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
        #print("second"+parts[0]+"terzo"+parts[-1])
        sentence.append((parts[0], parts[-1]))
        #print(sentence)
    f.close()        
    return sentences

def divide_in_training_and_test(corpus):
    test_corpus = corpus.pop()
    training_corpus = corpus
    return (training_corpus,test_corpus)

def preprocess(corpus):
    prova = {}
    lista = []
    words = {}
    #Every f is a sentence, a list of tuple
    for f in corpus:
        #Every i is a tuple, I take one tuple a time
        for i in f:
            #I get every character lower
            #I create a list with all tokens
            lista.append(i[0].lower())
            #I create a dictionary with token and tag
            prova[i[0].lower()] = i[1]
    #print(lista)
    #print(prova.keys())
    print(lista.__len__())
    #I evaluate for every token the number of occurences
    for i in lista:
        if i in words.keys():
            words[i] += 1
        else:
            words[i] = 1
    x = list(words.keys())
    #I put unknown for every token that occurence only one time
    for i in x:
        words['unknown'] = words.pop(i) if words.get(i) == 1 else False
    #print(words)
    return words
#Se nel preprocess lo trasformo in dictionary, lo posso poi usare così per tutto il resto



# Exercise 1 ###################################################################
'''
Implement the probability distribution of the initial states.
Parameters:	state: string
            internal_representation: data structure representing the parametrization of this probability distribution;
                this data structure is returned by the function estimate_initial_state_probabilities
Returns: float; initial probability of the given state
'''
def initial_state_probabilities(state, internal_representation):
    return internal_representation.get(state)
    
    
    
'''
Implement the matrix of transition probabilities.
Parameters:	from_state: string;
            to_state: string;
            internal_representation: data structure representing the parameterization of the matrix of transition probabilities;
                this data structure is returned by the function estimate_transition_probabilities
Returns: float; probability of transition from_state -> to_state
'''
def transition_probabilities(from_state, to_state, internal_representation):
    return internal_representation.get((from_state,to_state))
    
    
    
    
'''
Implement the matrix of emmision probabilities.
Parameters:	state: string;
            emission_symbol: string;
            internal_representation: data structure representing the parameterization of the matrix of emission probabilities;
                this data structure is returned by the function estimate_emission_probabilities
Returns: float; emission probability of the symbol emission_symbol if the current state is state
'''
def emission_probabilities(state, emission_symbol, internal_representation):
    return internal_representation.get((state,emission_symbol))
    
    
    
    
'''
Implement a function for estimating the parameters of the probability distribution of the initial states.
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the probability distribution of the initial states;
            use this data structure for the argument internal_representation of the function initial_state_probabilities
'''
def estimate_initial_state_probabilities(corpus):
    prova = {}
    lista = []
    initial_tags = {}
    #Take only first word of a sentence
    for f in corpus:
        #For every sentence I take only the first token and his tag
        (a,b) = f[0]
        #I create a dictionary with tags and number of occurences
        if b in initial_tags.keys():
            initial_tags[b] += 1
        else:
            initial_tags[b] = 1
    #print(initial_tags)
    #I take the sum of all occurences
    tot = sum(initial_tags.values())
    #I iterate on items of the dictionary
    for k,v in initial_tags.items():
        #I calculate the probability of every tag
        prob = v/tot
        #I update the dictionary with the probability
        initial_tags[k] = prob
    #print(initial_tags)
    #print(sum(initial_tags.values()))
    #Every f is a sentence, a list of tuple
    for f in corpus:
        #Every i is a tuple, I take one tuple a time
        for i in f:
            #I get every character lower
            #I create a list with all tokens
            lista.append(i[0].lower())
            #I create a dictionary with token and tag
            prova[i[0].lower()] = i[1]
    #print(lista)
    #I create the set of tokens
    k = set(lista)
    #I create a set with all states
    state = set(prova.values())
    #print(prova.keys())
    print(lista.__len__())
    words = {}
    #I evaluate for every token the number of occurences
    for i in lista:
        if i in words.keys():
            words[i] += 1
        else:
            words[i] = 1
    #I sort the dictionary
    sorted_x = sorted(words.items(), key=operator.itemgetter(1))
    #print(sorted_x)
    x = list(words.keys())
    #print(x)
    #I put unknown for every token that occurence only one time
    for i in x:
        words['unknown'] = words.pop(i) if words.get(i) == 1 else False
    #print(words)
    #I've to take only the first word of every sentence to evaluate for the initial probability
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
        #Viene considerata l'ultima? In teoria no perché altrimenti darebbe errore
        for i in range(len(f)-1):
            (a, b) = f[i]
            (c, d) = f[i+1]
            #I create a dictionary with tag and his following and number of occurences
            if (b, d) in transition_tags.keys():
                transition_tags[(b, d)] += 1
            else:
                transition_tags[(b, d)] = 1
    #I take the sum of all occurences
    #Devo prendere solo le occorrenze del tag successivo
    #E quello successivo se è alla fine della frase
    tot = sum(transition_tags.value())
    #I iterate on items of the dictionary
    for k,v in transition_tags.items():
        #I calculate the probability of every tag
        prob = v/tot
        #I update the dictionary with the probability
        transition_tags[k] = prob
    print(transition_tags)
    #print(sum(transition_tags.values()))
    return transition_tags
    
    
    
    
'''
Implement a function for estimating the parameters of the matrix of emission probabilities
Parameters: corpus: list returned by the function import_corpus
Returns: data structure containing the parameters of the matrix of emission probabilities;
            use this data structure for the argument internal_representation of the function emission_probabilities
'''
def estimate_emission_probabilities(corpus):
    emission_tags = {}
    tokens = []
    tags = []
    #Every f is a sentence, a list of tuple
    for f in corpus:
        #Every i is a tuple, I take one tuple a time
        for i in f:
            #I get every character lower
            #I create a list with all tokens
            tokens.append(i[0].lower())
            #I create a list with all tags
            tags.append(i[1])
    #I create the dictionary for the tags
    dict_for_tags = {}
    #I count the occurences of every tags
    for x in tags:
        if x in dict_for_tags.keys():
            dict_for_tags[x] += 1
        else:
            dict_for_tags[x] = 1
    #print(dict_for_tags)
    prob_tags = []
    tot = sum(dict_for_tags.values())
    #For every tag I calculate the probability
    for k,v in dict_for_tags.items():
        #I calculate the probability of every tag
        prob_tags.append(v/tot)
        #I update the dictionary with the probability
        dict_for_tags[k] = v/tot
    #print(dict_for_tags)
    #print(sum(dict_for_tags.values()))
    #tokens = set(tokens)
    #tags = set(tags)
    #print(tokens)
    #print(tags)
    #I create a list with every tag and token, but is not correct, if a word didn't have a token don't need to have it know
    #otherwise the probability is the same for every word and tag, so I've to take the initial list of tuple and calculate the number
    # of occurences
    #Have to be done directly on the corpus
    for f in corpus:
        for i in f:
            #For every tuple of token and tag, I calculate the number of occurences
            (a, b) = i
            #I create a dictionary with tag and token and his following and number of occurences
            if (a, b) in emission_tags.keys():
                emission_tags[(a.lower(), b)] += 1
            else:
                emission_tags[(a.lower(), b)] = 1
    #print(emission_tags)
    #Sommo tutte le occorrenze
    #Anche qui si considera solo il tag della parola
    freq_tag = []
    #freq_tag.append()
    #I iterate on items of the dictionary
    for k,v in emission_tags.items():
        #I calculate the probability of every tag
        #Mi servono dei dizionari non delle stringhe altrimenti come faccio a trovare il tag in comune
        #prob_tag_value.append(v/tot)
        #print(dict_for_tags.get(k[1]))
        #Calcolo la probabilità della tupla token e tag
        #tot = sum(emission_tags.get((x,k[1])))
        #print(emission_tags.get())
        prob = v/tot
        #Vado a dividere la probabilità calcolata prima per la probabilità del tag
        #prob = prob_token_tag/dict_for_tags.get(k[1])
        #I update the dictionary with the probability
        emission_tags[k] = prob
    print(emission_tags)
    print(sum(emission_tags.values()))
    #La print della sum esce maggiore di 1, perciò non va bene
    #print(emission_tags_list)
    return emission_tags

    
    
    
    
    
# Exercise 2 ###################################################################
''''
Implement the Viterbi algorithm for computing the most likely state sequence given a sequence of observed symbols.
Parameters: observed_smbols: list of strings; the sequence of observed symbols
            initial_state_probabilities_parameters: data structure containing the parameters of the probability distribution of the initial states, returned by estimate_initial_state_probabilities
            transition_probabilities_parameters: data structure containing the parameters of the matrix of transition probabilities, returned by estimate_transition_probabilities
            emission_probabilities_parameters: data structure containing the parameters of the matrix of emission probabilities, returned by estimate_emission_probabilities
Returns: list of strings; the most likely state sequence
'''
def most_likely_state_sequence(observed_smbols, initial_state_probabilities_parameters, transition_probabilities_parameters, emission_probabilities_parameters):
    pass


#For transitional and emission is better to create a nested dictionary
if __name__ == '__main__':
    corpus = import_corpus("corpus_ner.txt")
    corpus.pop()
    (corpus_training, corpus_test) = divide_in_training_and_test(corpus)
    corpus_training = preprocess(corpus_training)
    #Dividere il corpus in training e test togliendo una stringa dal corpus
    #Fare il preprocessing mettendo gli unknown
    #Non vengono presi tutti i tag ed i valori delle probabilità sono sbagliati
    #estimate_initial_state_probabilities(x)
    #estimate_transition_probabilities(x)
    #Controllare i commenti della emission
    estimate_emission_probabilities(x)











