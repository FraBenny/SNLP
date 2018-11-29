import operator
from builtins import str
import random

def preprocess_text(text):
    text = text.replace(',,','')
    text = text.replace(';,','')
    text = text.replace(':','')
    text = text.replace('(',' ')
    text = text.replace(')',' ')
    text = text.replace('\'',' ')
    text = text.replace('\n','')
    text = text.replace('-','')
    text = text.replace('?',' ?')
    text = text.replace('!',' !')
    text = text.replace('.',' .EOF ')
    text = text.lower()
    text = text.split('.')
    return text

def calculate_prob_unigram(wordcoun1):
    for w,v in wordcoun1.items():
        tot_freq = v + tot_freq
    wordprob1 = {}
    for w,v in wordcoun1.items():
        prob = v/tot_freq
        wordprob1[w] = prob
    return wordprob1

def calculate_prob_bigram(wordcoun1, wordcoun2):
    wordprob2 = calculate_prob_unigram(wordcoun1)
    for i in range(len(wordcoun2)-1):
        for w,v in wordcoun2[i]:
            freq = wordprob2.get(wordcoun2[i], None)
            prob = v/freq
            wordprob2[wordcoun2[i-1]][wordcoun2[i]] = prob
    return wordprob2

def calculate_prob_trigram(wordcoun1, wordcoun2, wordcoun3):
    wordprob3 = calculate_prob_bigram(wordcoun1, wordcoun2)
    for i in range(len(wordcoun3)-1):
        for w,v in wordcoun3[i][i]:
            freq = wordprob3.get(wordcoun3[i][i], None)
            prob = v/freq
            wordprob3[wordcoun3[i-2]][wordcoun3[i-1]][wordcoun3[i]] = prob
    return wordprob3

def parse_sentence_for_unigram(sentence):
    sentence = sentence.split()
    for i in range(len(sentence)-1):
        if sentence[i] not in wordcount1:
            wordcount1[sentence[i]] = 1
        else:
            wordcount1[sentence[i]] += 1
    return wordcount1

def parse_sentence_for_bigram(sentence2):
    sentence2 = sentence2.split()
    wordcount2 = {}
    for i in range(len(sentence2)-1):
        if i == 0:
            continue
        elif sentence2[i] not in wordcount2.keys():
            wordcount2 = {sentence2[i]: {sentence2[i-1] : 1}}
        else:
            if sentence2[i-1] in wordcount2[sentence2[i]].keys():
                wordcount2[sentence2[i]][sentence2[i-1]] += 1
            else:
                wordcount2[sentence2[i]][sentence2[i-1]] = 1
    return wordcount2


def parse_sentence_for_trigram(sentence3):
    sentence3 = sentence3.split()
    wordcount3 = {}
    for i in range(len(sentence3)-1):
        if i == 0:
            continue
        if i == 1:
            continue
        elif sentence3[i] not in wordcount3.keys():
            wordcount3 = {sentence3[i]: {sentence3[i-1] : {sentence3[i-2] : 1}}}
        else:
            if sentence3[i-1] in wordcount3[sentence3[i]].keys():
                if sentence3[i-2] in wordcount3[sentence3[i]][sentence3[i-1]].keys():
                    wordcount3[sentence3[i]][sentence3[i-1]][sentence3[i-2]] += 1
                else:
                    wordcount3[sentence3[i]][sentence3[i-1]][sentence3[i-2]] = 1
            else:
                wordcount3 = {sentence3[i]: {sentence3[i-1] : {sentence3[i-2] : 1}}}
    return wordcount3

def sampling_function_unigram(wordprob1):
    wordprob = sorted(wordprob1.items(), key=operator.itemgetter(0), reverse=True)
    x = random.uniform(0,1)
    tot = 0
    j = 0
    for i in range(len(wordprob)-1):
        tot = wordprob[i] + tot
        if tot >= x:
            j = i
            print(j)
            break
    return j

def generate_text_unigram(wordprob1):
    string = ''
    while True:
        index = sampling_function_unigram(wordprob1)
        word = wordprob1[index]
        if word == 'EOF':
            break
        else:
            string = string + ' ' + word
    return string


if __name__ == '__main__':
    inputfile = open("corpus.txt","r+")
    outputfile = open('test.csv', 'w')
    wordcount1 = {}
    wordcount2 = {}
    wordcount3 = {}
    text = []
    text = inputfile.read()
    text = preprocess_text(text)
    for sentence in text:
        wordcount1 = parse_sentence_for_unigram(sentence)
    outputfile.write(str(wordcount1) + "\n")
    for sentence in text:
        wordcount1 = parse_sentence_for_bigram(sentence)
    outputfile.write(str(wordcount1) + "\n")
    for sentence in text:
        wordcount1 = parse_sentence_for_trigram(sentence)
    outputfile.write(str(wordcount1) + "\n")
    prob_unigram = calculate_prob_unigram(wordcount1)
    prob_bigram = calculate_prob_bigram(wordcount1, wordcount2)
    prob_trigram = calculate_prob_trigram(wordcount1, wordcount2, wordcount3)
    outputfile.write(str(prob_unigram) + "\n" + str(prob_bigram) + "\n" + str(prob_trigram) + "\n")
    outputfile.write(generate_text_unigram(prob_unigram) + "\n")
    inputfile.close()
    outputfile.close()

