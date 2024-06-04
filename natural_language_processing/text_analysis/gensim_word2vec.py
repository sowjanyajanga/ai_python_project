import gensim
import nltk
nltk.download('abc')
from nltk.corpus import abc

model= gensim.models.Word2Vec(abc.sents())

X= list(model.wv.index_to_key)

data=model.wv.most_similar('science')

print(data)

