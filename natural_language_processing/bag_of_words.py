import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import re
import numpy as np


def text_clean(corpus, keep_list):
    '''
    Purpose : Function to keep only alphabets, digits and certain words (punctuations, qmarks, tabs etc. removed)

    Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained
            even after the cleaning process

    Output : Returns the cleaned text corpus

    '''
    cleaned_corpus = pd.Series()
    for row in corpus:
        qs = []
        for word in row.split():
            if word not in keep_list:
                p1 = re.sub(pattern='[^a-zA-Z0-9]', repl=' ', string=word)
                p1 = p1.lower()
                qs.append(p1)
            else:
                qs.append(word)
        # cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs)))
        cleaned_corpus = pd.concat([cleaned_corpus,pd.Series(' '.join(qs))])
    return cleaned_corpus

'''
    Stop words are used to connect different parts of a sentence but don't contribute much to the meaning
'''
def stopwords_removal(corpus):
    wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
    stop = set(stopwords.words('english'))
    for word in wh_words:
        stop.remove(word)
    corpus = [[x for x in x.split() if x not in stop] for x in corpus]
    return corpus

'''
    lemmatize looks at the context of the word to update the word unlike stemming that purely updates the suffix and is hence faster
    ex:
        stemmping: History -> Histori
        lemmatization: History -> History
'''
def lemmatize(corpus):
    lem = WordNetLemmatizer()
    corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
    return corpus

'''
    Used to remove suffixes of the words that don't contribute to the meaning
    ex: walking --> walk 
'''
def stem(corpus, stem_type = None):
    if stem_type == 'snowball':
        stemmer = SnowballStemmer(language = 'english')
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    else :
        stemmer = PorterStemmer()
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    return corpus


def preprocess(corpus, keep_list, cleaning=True, stemming=False, stem_type=None, lemmatization=False,
               remove_stopwords=True):
    '''
    Purpose : Function to perform all pre-processing tasks (cleaning, stemming, lemmatization, stopwords removal etc.)

    Input :
    'corpus' - Text corpus on which pre-processing tasks will be performed
    'keep_list' - List of words to be retained during cleaning process
    'cleaning', 'stemming', 'lemmatization', 'remove_stopwords' - Boolean variables indicating whether a particular task should
                                                                  be performed or not
    'stem_type' - Choose between Porter stemmer or Snowball(Porter2) stemmer. Default is "None", which corresponds to Porter
                  Stemmer. 'snowball' corresponds to Snowball Stemmer

    Note : Either stemming or lemmatization should be used. There's no benefit of using both of them together

    Output : Returns the processed text corpus

    '''

    if cleaning == True:
        corpus = text_clean(corpus, keep_list)

    if remove_stopwords == True:
        corpus = stopwords_removal(corpus)
    else:
        corpus = [[x for x in x.split()] for x in corpus]

    if lemmatization == True:
        corpus = lemmatize(corpus)

    if stemming == True:
        corpus = stem(corpus, stem_type)

    corpus = [' '.join(x) for x in corpus]

    return corpus



if __name__ == '__main__':

    common_dot_words = ['U.S.', 'Mr.', 'Mrs.', 'D.C.']

    sentences = ["We are reading about Natural Language Processing Here",
    "Natural Language Processing making computers comprehend language data",
    "The field of Natural Language Processing is evolving everyday"]

    corpus = pd.Series(sentences)

    # print('corpus ' + corpus)
    # corpus.info()

    preprocessed_corpus = preprocess(corpus, keep_list = common_dot_words, stemming = False, \
                                     stem_type = None, lemmatization = True, \
                                     remove_stopwords = True)


    # print(preprocessed_corpus)

    set_of_words = set()
    for sentence in preprocessed_corpus:
        for word in sentence.split():
            set_of_words.add(word)
    vocab = list(set_of_words)
    # print(vocab)

    position = {}
    for i, token in enumerate(vocab):
        position[token] = i

    print(position)

    # creates a matrix IxJ matrix where
    #           I= number of sentences in the input vocabulary
    #           J= number of words found to contribute to the meaning after filtering, stemming, lemmanting etc
    bow_matrix = np.zeros((len(preprocessed_corpus), len(vocab)))

    for i, preprocessed_sentence in enumerate(preprocessed_corpus):
        for token in preprocessed_sentence.split():
            bow_matrix[i][position[token]] = bow_matrix[i][position[token]] + 1

    print(preprocessed_corpus)
    print(position)
    print(bow_matrix)
