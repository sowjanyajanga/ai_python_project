import os.path

import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import FastText
import io
import collections

# Download data from the kaggle at
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# rename train.csv to comment_text.csv

def text_clean(corpus):
    '''
    Purpose : Function to keep only alphabets, digits and certain words (punctuations, qmarks, tabs etc. removed)

    Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained
            even after the cleaning process

    Output : Returns the cleaned text corpus

    '''
    cleaned_corpus = []
    for row in corpus:
        qs = []
        for word in row.split():
            p1 = re.sub(pattern='[^a-zA-Z0-9]', repl=' ', string=word)
            p1 = p1.lower()
            qs.append(p1)
        cleaned_corpus.append(' '.join(qs))
    return cleaned_corpus

def stopwords_removal(corpus):
    wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
    stop = set(stopwords.words('english'))
    for word in wh_words:
        stop.remove(word)
    corpus = [[x for x in x.split() if x not in stop] for x in corpus]
    return corpus

def lemmatize(corpus):
    lem = WordNetLemmatizer()
    corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
    return corpus

def stem(corpus, stem_type = None):
    if stem_type == 'snowball':
        stemmer = SnowballStemmer(language = 'english')
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    else :
        stemmer = PorterStemmer()
        corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    return corpus

def preprocess(corpus, cleaning=True, stemming=False, stem_type=None, lemmatization=False, remove_stopwords=True):
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
        corpus = text_clean(corpus)

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


words = []
data = []
data_file = os.path.dirname(os.path.abspath(__name__)) + "/natural_language_processing/Dataset/comments.txt"

with io.open(data_file, 'r') as file:
    for entry in file:
        entry = entry.strip()
        data.append(entry)
        words.extend(entry.split())

unique_words = []
unique_words = collections.Counter(words)

print(unique_words.most_common(10))

data = preprocess(data)

preprocessed_data = []
for line in data:
    if line != "":
        preprocessed_data.append(line.split())


model = FastText(vector_size=300, window=3, min_count=1, min_n=1, max_n=5)

model.build_vocab(preprocessed_data)

print(len(model.wv.index_to_key))

model.train(corpus_iterable=preprocessed_data, total_examples=len(preprocessed_data), epochs=10)

print(model.wv.most_similar('eplain', topn=5))

print(model.wv.most_similar('reminder', topn=5))

print(model.wv.most_similar('relevnt', topn=5))

print(model.wv.most_similar('purse', topn=5))

# Gensim 4.3.2 FastTest does not have
#   model.wmdistance
# sentence_1 = "Obama speaks to the media in Illinois"
# sentence_2 = "President greets the press in Chicago"
# sentence_3 = "Apple is my favorite company"
#
# word_mover_distance = model.wmdistance(sentence_1, sentence_2)
#
# print(word_mover_distance)
#
# word_mover_distance = model.wmdistance(sentence_2, sentence_3)
#
# print(word_mover_distance)