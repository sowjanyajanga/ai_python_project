import pandas as pd
import bag_of_words as bow
import numpy as np

sentence = ["We are reading about Natural Language Processing Here"]
corpus = pd.Series(sentence)

# Preprocessing with Lemmatization here
preprocessed_corpus = bow.preprocess(corpus, keep_list = [],
                        stemming = False, stem_type = None,lemmatization = True,
                        remove_stopwords = True)

print('preprocessed_corpus')
print(preprocessed_corpus)

set_of_words = set()
for word in preprocessed_corpus[0].split():
    set_of_words.add(word)

vocab = list(set_of_words)
print('vocab')
print(vocab)

position = {}
for i, token in enumerate(vocab):
    position[token] = i

print('position')
print(position)

one_hot_matrix = np.zeros((len(preprocessed_corpus[0].split()),
len(vocab)))
print('one_hot_matrix')
print(one_hot_matrix)

for i, token in enumerate(preprocessed_corpus[0].split()):
    one_hot_matrix[i][position[token]] = 1

print('One hot matrix after applying')
print(one_hot_matrix)