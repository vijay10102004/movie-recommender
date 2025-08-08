import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Step 1: Load and clean dataset
# -------------------------------

# Load dataset (adjust path if needed)
df = pd.read_csv("../data/tmdb_5000_movies.csv")

# Keep relevant columns (overview + tagline + genres if available)
cols = ['title', 'overview', 'tagline', 'genres']
df = df[[c for c in cols if c in df.columns]]

# Drop rows where overview is missing
df.dropna(subset=['overview'], inplace=True)

print("Dataset shape after cleaning:", df.shape)

# -------------------------------
# Step 2: Combine text for TF-IDF
# -------------------------------

# Combine overview + tagline + genres into a single 'content' column
df['content'] = df['overview'].fillna('') + " " + \
                df.get('tagline', '').fillna('') + " " + \
                df.get('genres', '').fillna('')

# -------------------------------
# Step 3: TF-IDF Vectorization
# -------------------------------

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df['content'])

print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("Sample words:", tfidf.get_feature_names_out()[1000:1010])

# -------------------------------
# Step 4: Cosine Similarity
# -------------------------------

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("Cosine similarity matrix shape:", cosine_sim.shape)

# Reset index and create title → index mapping
df = df.reset_index()
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# -------------------------------
# Step 5: Recommendation Function
# -------------------------------

def recommend_movies(title, top_n=5):
    # Check if title exists
    if title not in indices:
        return f"❌ Movie '{title}' not found in dataset."

    # Index of the movie
    idx = indices[title]

    # Similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip the first one (the movie itself)
    sim_scores = sim_scores[1:top_n+1]

    # Get indices of similar movies
    movie_indices = [i[0] for i in sim_scores]

    # Return movie titles
    return df['title'].iloc[movie_indices].tolist()

# -------------------------------
# Step 6: Test Recommendations
# -------------------------------

test_movie = "Inception"
print(f"\nTop 5 recommendations for '{test_movie}':")
recommendations = recommend_movies(test_movie, top_n=5)
for i, movie in enumerate(recommendations, 1):
    print(f"{i}. {movie}")
