import gensim
from gensim.models import KeyedVectors
import os

path = os.path.dirname(os.path.abspath(__name__)) + "/natural_language_processing/Dataset/GoogleNews-vectors-negative300.bin"

model=KeyedVectors.load_word2vec_format(path, binary=True)

sentence_1 = "Obama speaks to the media in Illinois"
sentence_2 = "President greets the press in Chicago"
sentence_3 = "Apple is my favorite company"

word_mover_distance = model.wmdistance(sentence_1, sentence_2)

print(str(word_mover_distance))

word_mover_distance = model.wmdistance(sentence_1, sentence_3)

print(str(word_mover_distance))

# model.init_sims(replace = True)