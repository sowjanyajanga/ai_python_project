from sklearn.feature_extraction.text import TfidfVectorizer
import bag_of_words as bow
import pandas as pd

if __name__ == '__main__':

    sentences = ["We are reading about Natural Language Processing Here",
    "Natural Language Processing making computers comprehend language data",
    "The field of Natural Language Processing is evolving everyday"]

    common_dot_words = ['U.S.', 'Mr.', 'Mrs.', 'D.C.']

    corpus = pd.Series(sentences)

    preprocessed_corpus = bow.preprocess(corpus, keep_list = common_dot_words, stemming = False, \
                                     stem_type = None, lemmatization = True, \
                                     remove_stopwords = True)

    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(preprocessed_corpus)

    print(vectorizer.get_feature_names_out())
    print(tf_idf_matrix.toarray())
    print("\nThe shape of the TF - IDF matrix is: ", tf_idf_matrix.shape)

