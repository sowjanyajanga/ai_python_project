from gensim.models import FastText
from gensim.test.utils import common_texts

model = FastText(vector_size=5, window=3, min_count=1)
model.build_vocab(corpus_iterable=common_texts)
model.train(corpus_iterable=common_texts,total_examples=len(common_texts), epochs=10)

# Vocab words gathered
print(model.wv.index_to_key)

print(model.wv['human'])

most_similar_output = model.wv.most_similar(positive=['computer', 'interface'], negative=['human'])

print(most_similar_output)

sentences_to_be_added = [["I", "am", "learning", "Natural", "Language", "Processing"],
                        ["Natural", "Language", "Processing", "is", "cool"]]

model.build_vocab(sentences_to_be_added, update=True)
model.train(corpus_iterable=common_texts,total_examples=len(sentences_to_be_added), epochs=10)