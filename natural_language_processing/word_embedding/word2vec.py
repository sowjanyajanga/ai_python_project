from gensim.models import Word2Vec
sentences = [["I", "am", "trying", "to", "understand", "Natural", "Language", "Processing"],
             ["Natural", "Language", "Processing", "is", "fun","to", "learn"],
             ["There", "are", "numerous", "use", "cases", "of","Natural", "Language", "Processing"]]

# min_count means that vectors are only created for words that occur more often than the min count
# model = Word2Vec(sentences, min_count=2, vector_size=300)
model = Word2Vec (sentences, min_count=1, vector_size = 300, workers = 2, sg = 1, negative = 1)


print(model.vector_size)

print(str(model.wv.index_to_key))

print(len(model.wv.index_to_key))

# for each in model.wv:
#     print(each)