import pandas as pd
import numpy as np
import re
from collections import Counter
import math

# Example movie dataset (same as original)
data = {
    'title': ['The Matrix', 'John Wick', 'Inception', 'Interstellar', 'The Notebook'],
    'description': [
        'A computer hacker learns about the true nature of reality and his role in the war against its controllers.',
        'An ex-hitman comes out of retirement to track down the gangsters that killed his dog.',
        'A thief who steals corporate secrets through dream-sharing technology is given a chance to erase his criminal history.',
        'A team of explorers travel through a wormhole in space in an attempt to ensure humanity’s survival.',
        'A young couple fall in love in the 1940s but are separated by social differences and war.'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# --- 1. Manual TF-IDF Vectorization ---

# Define a set of English stop words
stop_words = {'a', 'an', 'the', 'in', 'on', 'of', 'and', 'for', 'to', 'is', 'his', 'her', 'its', 'against'}


# Preprocess and tokenize the documents
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation
    tokens = text.split()
    return [word for word in tokens if word not in stop_words]


# Apply tokenization to all descriptions
tokenized_docs = [tokenize(doc) for doc in df['description']]


# --- Calculate TF (Term Frequency) ---
def compute_tf(tokenized_doc):
    word_counts = Counter(tokenized_doc)
    total_words = len(tokenized_doc)
    tf = {word: count / total_words for word, count in word_counts.items()}
    return tf


tf_docs = [compute_tf(doc) for doc in tokenized_docs]


# --- Calculate IDF (Inverse Document Frequency) ---
def compute_idf(tokenized_docs):
    num_docs = len(tokenized_docs)
    # Count how many documents contain each word
    doc_freq = Counter()
    for doc in tokenized_docs:
        doc_freq.update(set(doc))

    # Calculate IDF for each word in the vocabulary
    idf = {word: math.log(num_docs / (count + 1)) for word, count in doc_freq.items()}
    return idf


idf_scores = compute_idf(tokenized_docs)

# Build the vocabulary (all unique words)
vocab = sorted(idf_scores.keys())

# --- Create the TF-IDF Matrix ---
num_docs = len(tokenized_docs)
num_words = len(vocab)
tfidf_matrix = np.zeros((num_docs, num_words))

# Populate the matrix
for i, doc_tf in enumerate(tf_docs):
    for word, tf_value in doc_tf.items():
        if word in vocab:
            j = vocab.index(word)
            idf_value = idf_scores[word]
            tfidf_matrix[i, j] = tf_value * idf_value


# --- 2. Manual Cosine Similarity Calculation ---

def calculate_cosine_similarity(matrix):
    num_docs = matrix.shape[0]
    similarity_matrix = np.zeros((num_docs, num_docs))

    for i in range(num_docs):
        for j in range(num_docs):
            vec1 = matrix[i]
            vec2 = matrix[j]

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            # Avoid division by zero
            if norm1 > 0 and norm2 > 0:
                similarity_matrix[i, j] = dot_product / (norm1 * norm2)

    return similarity_matrix


cosine_sim = calculate_cosine_similarity(tfidf_matrix)


# --- 3. Function to get recommendations (same as original) ---

def recommend(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df.index[df['title'] == title][0]

    # Get similarity scores for all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 3 most similar movies (excluding itself)
    sim_scores = sim_scores[1:4]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 3 most similar movies
    return df['title'].iloc[movie_indices]


# --- Example usage ---
print("Recommended movies for 'The Matrix':")
print(recommend('The Matrix'))
import pandas as pd
import numpy as np
import re
from collections import Counter
import math

# Example movie dataset
data = {
    'title': ['The Matrix', 'John Wick', 'Inception', 'Interstellar', 'The Notebook'],
    'description': [
        'A computer hacker learns about the true nature of reality and his role in the war against its controllers.',
        'An ex-hitman comes out of retirement to track down the gangsters that killed his dog.',
        'A thief who steals corporate secrets through dream-sharing technology is given a chance to erase his criminal history.',
        'A team of explorers travel through a wormhole in space in an attempt to ensure humanity’s survival.',
        'A young couple fall in love in the 1940s but are separated by social differences and war.'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# --- 1. Manual TF-IDF Vectorization (with improvements) ---

# A standard set of English stop words
stop_words = {
    'a', 'an', 'the', 'in', 'on', 'of', 'and', 'for', 'to', 'is', 'his', 'her', 'its', 'against',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours'
}

# Preprocess and tokenize the documents
def tokenize(text):
    text = text.lower()
    # Keep only letters and spaces, removing punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    # Return words that are not stop words and are longer than 2 characters
    return [word for word in tokens if word not in stop_words and len(word) > 2]

# Apply tokenization to all descriptions
tokenized_docs = [tokenize(doc) for doc in df['description']]

# Calculate TF (Term Frequency)
def compute_tf(tokenized_doc):
    word_counts = Counter(tokenized_doc)
    total_words = len(tokenized_doc)
    # Avoid division by zero for empty descriptions
    return {word: count / total_words for word, count in word_counts.items()} if total_words > 0 else {}
