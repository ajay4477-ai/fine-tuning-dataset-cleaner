
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


documents = [
    "Python is a programming language",
    "Python is useful for data analysis",
    "Machine learning uses Python",
    "Explain what Python is",
    "What is Python?"
]

# Convert text documents into TF-IDF vectors.
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

print("\n========== VOCABULARY ==========")
print(feature_names)

print("\n========== TF-IDF MATRIX ==========")
print(tfidf_matrix.toarray().round(3))

print("\n========== DOCUMENT REPRESENTATION ==========")

for index, vector in enumerate(tfidf_matrix.toarray()):
    print(f"\nDocument {index + 1}:")

    for word, score in zip(feature_names, vector):
        if score > 0:
            print(f"{word}: {score:.4f}")

# Calculate cosine similarity between documents.
similarity_matrix = cosine_similarity(tfidf_matrix)

print("\n========== COSINE SIMILARITY MATRIX ==========")
print(similarity_matrix.round(2))

# Compare the first two documents.
similarity_score = similarity_matrix[0, 1]

print("\nSimilarity between Document 1 and Document 2:")
print(round(similarity_score, 4))
