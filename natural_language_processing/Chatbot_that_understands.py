import ast
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ChatbotWithUnderstanding:

    def __init__(self):
        self.questions = []
        self.answers = []

        data_file_path = os.path.dirname(os.path.abspath(__name__)) + '/natural_language_processing/Dataset/qa_Electronics.json'

        print(data_file_path)
        with open(data_file_path,'r') as f:
            for line in f:
                data = ast.literal_eval(line)
                self.questions.append(data['question'].lower())
                self.answers.append(data['answer'].lower())

        self.vectorizer = CountVectorizer(stop_words='english')
        self.X_vec = self.vectorizer.fit_transform(self.questions)
        self.tfidf = TfidfTransformer(norm='l2')
        self.X_tfidf = self.tfidf.fit_transform(self.X_vec)

    def conversation(self, im):
        Y_vec = self.vectorizer.transform(im)
        Y_tfidf = self.tfidf.fit_transform(Y_vec)
        angle = np.rad2deg(np.arccos(max(cosine_similarity(Y_tfidf, self.X_tfidf)[0])))
        if angle > 60 :
            return "sorry, I did not quite understand that"
        else:
            return self.answers[np.argmax(cosine_similarity(Y_tfidf, self.X_tfidf)[0])]



if __name__ == '__main__':
    usr = input("Please enter your username: ")
    print("“support: Hi, welcome to Q&A support. How can I help you?")
    cb = ChatbotWithUnderstanding()
    while True:
        im = input("{}: ".format(usr))
        if im.lower() == 'bye':
            print("Q & A support: bye!")
            break
        else:
            print("Q & A support: "+ cb.conversation([im]))
