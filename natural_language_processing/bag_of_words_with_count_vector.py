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
from sklearn.feature_extraction.text import CountVectorizer
import bag_of_words as bow


if __name__ == '__main__':

    sentences = ["We are reading about Natural Language Processing Here",
    "Natural Language Processing making computers comprehend language data",
    "The field of Natural Language Processing is evolving everyday"]

    corpus = pd.Series(sentences)

    # Preprocessing with Lemmatization here
    preprocessed_corpus = bow.preprocess(corpus, keep_list = [], stemming = False, stem_type = None,
                                    lemmatization = True, remove_stopwords = True)
    preprocessed_corpus

    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(preprocessed_corpus)

    # print(vectorizer.get_feature_names_out())
    # print(bow_matrix.toarray())

    # CountVectorizer also has a feature to do tri word analysis by using ngram_range = (1, 3)
    vectorizer_ngram_range = CountVectorizer(analyzer='word', ngram_range = (1, 3))
    bow_matrix_ngram = vectorizer_ngram_range.fit_transform(preprocessed_corpus)
    # print(vectorizer_ngram_range.get_feature_names_out())
    # print(bow_matrix_ngram.toarray())

    # CountVectorizer with ngrams using a maxium of 6 features i.e. max_features = 6
    vectorizer_max_features = CountVectorizer(analyzer='word', ngram_range = (1, 3), max_features = 6)
    bow_matrix_max_features = vectorizer_max_features.fit_transform(preprocessed_corpus)
    # print(vectorizer_max_features.get_feature_names_out())
    # print(bow_matrix_max_features.toarray())

    # CountVectorizer To limit the vocabiulary to a maximum and minimum frequency of occurance of a word mentioned in the vocab
    vectorizer_max_features = CountVectorizer(analyzer='word', ngram_range = (1, 3), max_df = 3, min_df = 2)
    bow_matrix_max_features = vectorizer_max_features.fit_transform(preprocessed_corpus)
    print(vectorizer_max_features.get_feature_names_out())
    print(bow_matrix_max_features.toarray())

