import pandas as pd

# go one folder up from src/ to data/
df = pd.read_csv("../data/tmdb_5000_movies.csv")

print("✅ File loaded successfully!")
print("Shape:", df.shape)
print(df.head().to_string())
