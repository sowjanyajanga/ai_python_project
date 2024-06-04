from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
print(documents)

# vector_size=5 means the document will be represented by a vector of 5 floating point values.
# min_count represents the number of times a word has to occur inorder for it to be counted as a vocabulary
# epochs denotes number of times the model will be iterated over the corpus
model = Doc2Vec(documents, vector_size=50, min_count=3, workers=4, epochs = 40)

model.train(documents, total_examples=model.corpus_count,epochs=model.epochs)

print(model.vector_size)

docvecs = model.wv.index_to_key

print(docvecs)
print(len(docvecs))

vector = model.infer_vector(['user', 'interface', 'for','computer'])
print(vector)


# dm=1 denotes distributed memory approach
# dm=0 stands for distributed bag of words
model = Doc2Vec(documents, vector_size=50, min_count=2, epochs=40,dm=1)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
