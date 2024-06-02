# Performance testing
import cProfile
import pstats
import gensim
from gensim.models import KeyedVectors
import os

model_path = os.path.dirname(os.path.abspath(__name__)) + '/natural_language_processing/Dataset/GoogleNews-vectors-negative300.bin'

model=KeyedVectors.load_word2vec_format(model_path, binary=True)

# print(model.most_similar('Delhi'))

if __name__ == "__main__":
    with cProfile.Profile() as profile:
        # result = model.most_similar(positive=['man', 'queen'], negative=['king'], topn=1)
        # print(result)
        #
        # result = model.most_similar(positive=['man', 'queen'], negative=['king'], topn=2)
        # print(result)
        #
        # result = model.most_similar(positive=['France', 'Rome'], negative=['Italy'], topn=1)
        # print(result)

        result = model.most_similar('Delhi')

        print(result)
        results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.TIME)
        results.print_stats()
        results.dump_stats("results.prof")
