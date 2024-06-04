from sklearn.datasets import fetch_20newsgroups
# Objective of this program is to take in a set of lines and for each line, find a classification or category that it belongs to
# Stages involved in running a text classifier through a tf-idf(term frequeny and inverse document frequency) transformer

category_map = {'misc.forsale': 'Sales', 'rec.motorcycles': 'Motorcycles','rec.sport.baseball': 'Baseball', 'sci.crypt': 'Cryptography', 'sci.space': 'Space'}

# 1. Get training data
training_data = fetch_20newsgroups(subset='train', categories=category_map.keys(), shuffle=True,random_state=7)

# Feature extraction
from sklearn.feature_extraction.text import CountVectorizer

# 2. Convert the training data in text to a numeric form
vectorizer = CountVectorizer()
X_train_termcounts = vectorizer.fit_transform(training_data.data)
print("Dimensions of training data:", X_train_termcounts.shape)

# Training a classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

# tf-idf transformer
# 3. run the vectorized traning data through tf-idf transformer so that we have
#    correct weights for both frequent words and infrequent but important words
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_termcounts)

# Multinomial Naive Bayes classifier
# 4. Build a Naive Bayes classifier
#    and train it on tf-if transformed vectored text data
#    and target label
classifier = MultinomialNB().fit(X_train_tfidf, training_data.target)

# 5. Take the test data that you are trying to find target values for and vectorize it just like training data
input_data = [ "The curveballs of right handed pitchers tend to curve to the left", "Caesar cipher is an ancient form of encryption", "This two-wheeler is really good on slippery roads"]
X_input_termcounts = vectorizer.transform(input_data)

# 6. Transform the test data through tf-idf just like traning data
X_input_tfidf = tfidf_transformer.transform(X_input_termcounts)

# Predict the output categories
# 7. Now predict the classifications of the test data
predicted_categories = classifier.predict(X_input_tfidf)

# Print the outputs
for sentence, category in zip(input_data, predicted_categories):
    print('\nInput:', sentence, '\nPredicted category:', category_map[training_data.target_names[category]])

