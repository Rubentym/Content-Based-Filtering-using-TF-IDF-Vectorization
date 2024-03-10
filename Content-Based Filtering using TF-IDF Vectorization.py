from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample content data
content_data = ['Python is a programming language',
                'Java is also a programming language',
                'Python and Java are popular programming languages']

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(content_data)

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Get recommendations based on similarity scores
movie_index = 0  # Choose a movie to get recommendations for
recommendations = sorted(list(enumerate(cosine_sim[movie_index])), key=lambda x: x[1], reverse=True)[:5]
print("Recommendations for movie", movie_index, ":", recommendations)
