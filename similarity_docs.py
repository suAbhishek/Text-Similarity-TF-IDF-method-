
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string
import math
import sys

import numpy

lst_stopwords = stopwords.words("english")
lst_puncs = set(string.punctuation)

def preprocessing_doc(raw_doc):
    word_lst = raw_doc.split()
    stp_free_text = " ".join([word.lower() for word in word_lst if word.lower() not in lst_stopwords])
    #print(stp_free_words)
    punc_free_words = "".join([ch for ch in stp_free_text if ch not in lst_puncs])
    processed_doc = punc_free_words.split()
    return processed_doc

def create_word_set(docA, docB):
    word_set = set(docA).union(set(docB))
    return word_set

def create_freq_vec(text_doc, all_word_set):
    freqtxt = FreqDist(text_doc)  # counts number of occurrences of a word in the text corpa
    word_count_pair = dict.fromkeys(all_word_set,0) # word-count pairs of all
    for w in text_doc:
        word_count_pair[w] = freqtxt[w]

    return word_count_pair,freqtxt

#Calculating TF which is ...... tf/|D|  D - length of Doc/text
def tf_calculation(text_doc,word_set):
    text_tf_dict, freq_text = create_freq_vec(text_doc, word_set)
    len_text = len(text_doc)
    for word in text_doc:
        text_tf_dict[word] = freq_text[word]/len_text
    return text_tf_dict

#Calculating IDF - which is tf/|D| * log((C)/(Dfw)) --
# C - no of docs in collection, Dfw - no. of docs containing the word w
def idf_calculation( doc1, doc2,word_set):
    idf_dict  = dict.fromkeys(word_set)
    for word in idf_dict.keys():
        idf_dict[word] = float(0)
    no_docs = 2
    for word in idf_dict.keys():
        if word in doc1:
            idf_dict[word] = idf_dict[word] +  1
        if word in doc2:
            idf_dict[word] = idf_dict[word] +  1
    for word,freq in idf_dict.items():
        idf_dict[word] = 1 + math.log(no_docs/float(freq))
    return idf_dict

#TFIDF = TF * IDF for a word
def tfidf_calc(tf_dict,idf_dict,doc,word_set):
    tfidf_dict = dict.fromkeys(word_set)
   # print("---")
    tfidf_dict_new = get_tfidf_dict(doc,tf_dict,tfidf_dict,idf_dict)
    for word,val in tfidf_dict_new.items():
        if tfidf_dict_new[word] == None:
            tfidf_dict_new[word] = float(0)
  #  print(tfidf_dict_new)
    return tfidf_dict_new


def get_tfidf_dict(doc, tf_dict,tfidf_dict,idf_dict):
    temp_dict = tfidf_dict
    for word in doc:
        temp_dict[word] = (tf_dict[word])*(idf_dict[word])
    return temp_dict

def get_similarity(doc1,doc2):
    doc1,doc2 = preprocessing_doc(doc1),preprocessing_doc(doc2)
    word_set = create_word_set(doc1,doc2)
    tf_doc1, tf_doc2 = tf_calculation(doc1,word_set), tf_calculation(doc2,word_set)
#    print(tf_doc2)
    idf_dict_of_docs = idf_calculation(doc1,doc2,word_set)
 #   print(idf_dict_of_docs)
    textA_tfidf = tfidf_calc(tf_doc1,idf_dict_of_docs,doc1,word_set)
    textB_tfidf = tfidf_calc(tf_doc2,idf_dict_of_docs,doc2,word_set)
    #computing cosine values for checking similary
  #  print(textA_tfidf,textB_tfidf)
    vec1 = (list(textA_tfidf.values()))
    vec2 = list(textB_tfidf.values())
  #  print("--------------\n ------------------")
    #print(vec1,vec2)
    return (1-nltk.cluster.cosine_distance(vec1,vec2))*100


if __name__ == '__main__':
    f1 = open("article1","r")
    f2 = open("article2","r")
    text1 = f1.read()
    text2 = f2.read()
    print(get_similarity(text1,text2))
    f1.close()
    f2.close()







