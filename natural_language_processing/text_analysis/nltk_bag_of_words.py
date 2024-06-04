import numpy as np
from nltk.corpus import brown
from nltk_chunking import splitter

if __name__ =='__main__':
    # Read the data from the Brown corpus
    data = ' '.join(brown.words()[:10000])
    # data = 'The brown dog is running. The black dog is in the black room. Running in the room is forbidden.'
    # Number of words in each chunk
    # num_words = 2000 # used for brown corpus
    num_words = 5
    chunks = []
    counter = 0
    text_chunks = splitter(data, num_words)
    for text in text_chunks:
        chunk = {'index': counter, 'text': text}
        chunks.append(chunk)
        counter += 1

    print(chunks)

    # Document term matrix that counts the number occurrences of each word in the document
    # Extract document term matrix
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(min_df=5, max_df=.95) # Used for brown corpus
    # vectorizer = CountVectorizer(min_df=2, max_df=.95)
    doc_term_matrix = vectorizer.fit_transform([chunk['text'] for chunk in chunks])

    print(doc_term_matrix)

    vocab = np.array(vectorizer.get_feature_names_out())
    print("Vocabulary:")
    print(vocab)

    print("Document term matrix:")
    # chunk_names = ['Chunk - 0', 'Chunk - 1', 'Chunk - 2', 'Chunk - 3' , 'Chunk - 4']
    chunk_names = ['Chunk - 0', 'Chunk - 1', 'Chunk - 2', 'Chunk - 3' , 'Chunk - 4']

    formatted_row = '{:>20}' *(len(chunk_names) + 1)
    print('\n', formatted_row.format('Word', *chunk_names), '\n')

    for word, item in zip(vocab, doc_term_matrix.T):
        # ‘item’ is a ‘csr_matrix’ data structure
        output = [str(x) for x in item.data]
        print(formatted_row.format(word, *output))

