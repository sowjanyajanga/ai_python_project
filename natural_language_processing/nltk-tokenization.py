import nltk
text = "Who would have thought that computer programs would be analyzing human sentiments"
from nltk.tokenize import word_tokenize
nltk.download('punkt')
tokens = word_tokenize(text)
# print(tokens)

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
newtokens=[word for word in tokens if word not in stopwords]
# print(newtokens)

from nltk.stem import WordNetLemmatizer
nltk.download('')
text = "Who would have thought that computer programs would be analyzing human sentiments"
tokens = word_tokenize(text)
lemmatizer = WordNetLemmatizer()
tokens=[lemmatizer.lemmatize(word) for word in tokens]
print(tokens)