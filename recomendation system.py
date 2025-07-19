import re
import math
from collections import Counter

# Movie data (no pandas used)
movies = [
    {
        'title': 'The Matrix',
        'description': 'A computer hacker learns about the true nature of reality and his role in the war against its controllers.'
    },
    {
        'title': 'John Wick',
        'description': 'An ex-hitman comes out of retirement to track down the gangsters that killed his dog.'
    },
    {
        'title': 'Inception',
        'description': 'A thief who steals corporate secrets through dream-sharing technology is given a chance to erase his criminal history.'
    },
    {
        'title': 'Interstellar',
        'description': 'A team of explorers travel through a wormhole in space in an attempt to ensure humanity’s survival.'
    },
    {
        'title': 'The Notebook',
        'description': 'A young couple fall in love in the 1940s but are separated by social differences and war.'
    }
]

# --- Text processing ---
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

def compute_tf(tokens):
    total = len(tokens)
    tf = Counter(tokens)
    return {word: count / total for word, count in tf.items()}

def compute_idf(token_lists):
    N = len(token_lists)
    all_words = set(word for tokens in token_lists for word in tokens)
    idf = {}
    for word in all_words:
        doc_count = sum(1 for tokens in token_lists if word in tokens)
        idf[word] = math.log((N + 1) / (doc_count + 1)) + 1
    return idf

def compute_tfidf(tf, idf):
    return {word: tf[word] * idf[word] for word in tf}

def cosine_similarity(vec1, vec2):
    dot = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in set(vec1) | set(vec2))
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# --- Prepare TF-IDF vectors ---
token_lists = [tokenize(movie['description']) for movie in movies]
tfs = [compute_tf(tokens) for tokens in token_lists]
idf = compute_idf(token_lists)
tfidfs = [compute_tfidf(tf, idf) for tf in tfs]

# --- Recommendation function ---
def recommend(title):
    index = next((i for i, movie in enumerate(movies) if movie['title'].lower() == title.lower()), None)
    if index is None:
        return f"Movie '{title}' not found."

    similarities = []
    for i, tfidf in enumerate(tfidfs):
        if i != index:
            sim = cosine_similarity(tfidfs[index], tfidf)
            similarities.append((movies[i]['title'], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    recommendations = [title for title, _ in similarities[:3]]
    return recommendations

# --- Example usage ---
print("Recommended for 'The Matrix':")
print(recommend("The Matrix"))

