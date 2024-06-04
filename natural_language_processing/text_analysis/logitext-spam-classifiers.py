import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.model_selection import train_test_split
import os

# Dataset from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
path_to_data = os.path.dirname(os.path.abspath(__name__)) + "/natural_language_processing/Dataset/spam.csv"
df = pd.read_csv(path_to_data, sep=',',header=None, encoding='latin-1')

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1],df[0])

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)

classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial')
classifier.fit(X_train, y_train)

X_test = vectorizer.transform(['Customer Loyalty Offer:The NEWNokia6650 Mobile from ONLY å£10 at TXTAUCTION!',
                                'Hi Dear how long have we not heard.'])

predictions = classifier.predict(X_test)
print(predictions)

