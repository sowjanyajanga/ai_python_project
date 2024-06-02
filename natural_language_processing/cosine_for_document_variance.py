import bag_of_words as bow
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.dot(vector1, vector2) / (np.sqrt(np.sum(vector1**2)) * np.sqrt(np.sum(vector2**2)))



common_dot_words = ['U.S.', 'Mr.', 'Mrs.', 'D.C.']

sentences = ["We are reading about Natural Language Processing Here",
             "Natural Language Processing making computers comprehend language data",
             "The field of Natural Language Processing is evolving everyday"]

corpus = pd.Series(sentences)

# print('corpus ' + corpus)
# corpus.info()

preprocessed_corpus = bow.preprocess(corpus, keep_list=common_dot_words, stemming=False, \
                                 stem_type=None, lemmatization=True, \
                                 remove_stopwords=True)

# Example#1
# With CountVectorizer that goes by the frequency of word alone
# vectorizer = CountVectorizer()
# bow_matrix = vectorizer.fit_transform(preprocessed_corpus)
#
# for i in range(bow_matrix.shape[0]):
#     for j in range(i + 1, bow_matrix.shape[0]):
#         print("The CountVectorizer cosine similarity between the documents ", i, "and", \
#                j, "is: ", cosine_similarity(bow_matrix.toarray()[i], \
#                bow_matrix.toarray()[j]))
#
# print(vectorizer.get_feature_names_out())
# print(bow_matrix.toarray())

#Example #2
# With TfIdVectorizer that assigns higher weight top less frequently used words
vectorizer = TfidfVectorizer()
tf_idf_matrix = vectorizer.fit_transform(preprocessed_corpus)

for i in range(tf_idf_matrix.shape[0]):
    for j in range(i + 1, tf_idf_matrix.shape[0]):
        print("The TFIDFVectorizer cosine similarity between the documents ", i, "and", \
                j, "is: ", cosine_similarity(tf_idf_matrix.toarray()[i], \
                tf_idf_matrix.toarray()[j]))