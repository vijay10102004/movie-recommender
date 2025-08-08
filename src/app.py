import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("../data/tmdb_5000_movies.csv")

# Keep relevant columns
cols = ['title', 'overview', 'tagline', 'genres']
df = df[[c for c in cols if c in df.columns]]
df.dropna(subset=['overview'], inplace=True)

# Combine text for TF-IDF
df['content'] = df['overview'].fillna('') + " " + \
                df.get('tagline', '').fillna('') + " " + \
                df.get('genres', '').fillna('')

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df['content'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Reset index and create mapping
df = df.reset_index()
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# -------------------------------
# Recommendation function
# -------------------------------
def recommend_movies(title, top_n=10):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    # Return only titles
    return df.iloc[movie_indices][['title']]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("🎬 Movie Recommender")
st.sidebar.write("Select a movie to get top similar movie suggestions.")

st.title("🎬 Content-Based Movie Recommender")
st.markdown("Find movies similar to your favorite ones!")

selected_movie = st.selectbox("Choose a movie you like:", df['title'].values)

if st.button("Show Recommendations"):
    recommendations = recommend_movies(selected_movie, top_n=10)
    
    if not recommendations.empty:
        st.subheader(f"Top 10 movies similar to '{selected_movie}':")
        for i, row in recommendations.iterrows():
            st.write(f"{i+1}. {row['title']}")
    else:
        st.write("❌ Movie not found.")
