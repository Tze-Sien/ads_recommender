from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def load_data():
    import pandas as pd

    # Assuming the parquet files are in the same directory
    df1 = pd.read_parquet("./data/train-00000-of-00002-6e587552aa3c8ac8.parquet")
    df2 = pd.read_parquet("./data/train-00001-of-00002-823ac5dae71e0e87.parquet")

    # Combine dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)

    return combined_df


def vectorize_text_data(df, text_column="text"):
    vectorizer = TfidfVectorizer()
    text_data = df[text_column].fillna("").astype(str)  # Ensure all data is string
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix, vectorizer


def load_or_create_vectorized_data(
    df, text_column="text", matrix_path="./vectorizer/tfidf_matrix.npz"
):
    from scipy import sparse
    import os
    import pickle

    if os.path.exists(matrix_path):
        print("Loading existing vectorized data...")
        tfidf_matrix = sparse.load_npz(matrix_path)
        with open("./vectorizer/vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        print("Vectorized data loaded successfully.")
    else:
        print("Vectorized data not found. Creating new vectorized data...")
        tfidf_matrix, vectorizer = vectorize_text_data(df, text_column)
        save_vectorized_data(tfidf_matrix, vectorizer, matrix_path)
        print("New vectorized data created and saved successfully.")

    return tfidf_matrix, vectorizer


def save_vectorized_data(tfidf_matrix, vectorizer, output_path="./vectorizer/tfidf_matrix.npz"):
    from scipy import sparse
    import os

    # Ensure the directory exists
    os.makedirs("./vectorizer", exist_ok=True)

    sparse.save_npz(output_path, tfidf_matrix)
    with open("./vectorizer/vectorizer.pkl", "wb") as f:
        import pickle

        pickle.dump(vectorizer, f)
    print(f"Vectorized data saved to {output_path} and vectorizer to ./vectorizer/vectorizer.pkl")


print("Loading data from parquet files...")
combined_df = load_data()
print("Data loaded successfully. Number of rows:", len(combined_df))

print("Loading or creating vectorized data...")
tfidf_matrix, vectorizer = load_or_create_vectorized_data(combined_df)
print("Vectorized data loaded or created successfully.")


def get_recommendations(user_input_ids, tfidf_matrix, top_n=5):
    # Get the TF-IDF vectors for the user input posts
    user_input_vectors = tfidf_matrix[user_input_ids]

    # Calculate cosine similarity between user input vectors and all other vectors
    cosine_similarities = cosine_similarity(user_input_vectors, tfidf_matrix)

    # Average the similarities across the user input posts
    avg_similarities = np.mean(cosine_similarities, axis=0)

    # Get indices of the top N similar posts, excluding the user input posts
    similar_indices = np.argsort(avg_similarities)[::-1]
    similar_indices = [i for i in similar_indices if i not in user_input_ids][:top_n]

    return similar_indices


def get_recommendations_with_scores(user_input_ids, tfidf_matrix, top_n=5):
    """Enhanced version that returns both indices and similarity scores"""
    # Get the TF-IDF vectors for the user input posts
    user_input_vectors = tfidf_matrix[user_input_ids]

    # Calculate cosine similarity between user input vectors and all other vectors
    cosine_similarities = cosine_similarity(user_input_vectors, tfidf_matrix)

    # Average the similarities across the user input posts
    avg_similarities = np.mean(cosine_similarities, axis=0)

    # Get indices of the top N similar posts, excluding the user input posts
    similar_indices = np.argsort(avg_similarities)[::-1]

    # Filter out user input posts and get top N
    filtered_results = []
    for idx in similar_indices:
        if idx not in user_input_ids:
            filtered_results.append((idx, avg_similarities[idx]))
        if len(filtered_results) >= top_n:
            break

    # Separate indices and scores
    indices = [result[0] for result in filtered_results]
    scores = [result[1] for result in filtered_results]

    return indices, scores


def get_recommendations_weighted_recent(user_input_ids, tfidf_matrix, top_n=5, recent_weight=2.0):
    """
    Get recommendations with higher weight for more recent liked items.
    Assumes user_input_ids are ordered by recency (most recent last).
    """
    if len(user_input_ids) == 0:
        return [], []

    # Get the TF-IDF vectors for the user input posts
    user_input_vectors = tfidf_matrix[user_input_ids]

    # Calculate cosine similarity between user input vectors and all other vectors
    cosine_similarities = cosine_similarity(user_input_vectors, tfidf_matrix)

    # Create weights that favor more recent items
    num_items = len(user_input_ids)
    weights = np.ones(num_items)

    # Give more recent items higher weight (exponential decay for older items)
    if num_items > 1:
        for i in range(num_items):
            # More recent items (higher index) get higher weight
            recency_factor = (i + 1) / num_items
            weights[i] = 1.0 + (recent_weight - 1.0) * recency_factor

    # Calculate weighted average of similarities
    weighted_similarities = np.average(cosine_similarities, axis=0, weights=weights)

    # Get indices of the top N similar posts, excluding the user input posts
    similar_indices = np.argsort(weighted_similarities)[::-1]

    # Filter out user input posts and get top N
    filtered_results = []
    for idx in similar_indices:
        if idx not in user_input_ids:
            filtered_results.append((idx, weighted_similarities[idx]))
        if len(filtered_results) >= top_n:
            break

    # Separate indices and scores
    indices = [result[0] for result in filtered_results]
    scores = [result[1] for result in filtered_results]

    return indices, scores


def get_recommendations_max_similarity(user_input_ids, tfidf_matrix, top_n=5):
    """
    Get recommendations based on maximum similarity to any liked item.
    This approach works well when users have diverse interests.
    """
    if len(user_input_ids) == 0:
        return [], []

    # Get the TF-IDF vectors for the user input posts
    user_input_vectors = tfidf_matrix[user_input_ids]

    # Calculate cosine similarity between user input vectors and all other vectors
    cosine_similarities = cosine_similarity(user_input_vectors, tfidf_matrix)

    # Take the maximum similarity across all user input posts
    max_similarities = np.max(cosine_similarities, axis=0)

    # Get indices of the top N similar posts, excluding the user input posts
    similar_indices = np.argsort(max_similarities)[::-1]

    # Filter out user input posts and get top N
    filtered_results = []
    for idx in similar_indices:
        if idx not in user_input_ids:
            filtered_results.append((idx, max_similarities[idx]))
        if len(filtered_results) >= top_n:
            break

    # Separate indices and scores
    indices = [result[0] for result in filtered_results]
    scores = [result[1] for result in filtered_results]

    return indices, scores


def get_recommendations_clustered(user_input_ids, tfidf_matrix, top_n=5, cluster_threshold=0.3):
    """
    Get recommendations using clustering approach for diverse user interests.
    Groups similar liked items and recommends based on clusters.
    """
    if len(user_input_ids) == 0:
        return [], []

    if len(user_input_ids) == 1:
        # Fall back to simple similarity for single item
        return get_recommendations_with_scores(user_input_ids, tfidf_matrix, top_n)

    # Get vectors for liked items
    user_vectors = tfidf_matrix[user_input_ids]

    # Calculate similarity matrix between liked items
    similarity_matrix = cosine_similarity(user_vectors)

    # Convert to distance matrix
    distance_matrix = 1 - similarity_matrix

    # Cluster the liked items
    try:
        n_clusters = min(3, len(user_input_ids))  # Limit clusters to avoid over-segmentation
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
        clusters = clustering.fit_predict(distance_matrix)
    except Exception:
        # Fall back to simple approach if clustering fails
        return get_recommendations_with_scores(user_input_ids, tfidf_matrix, top_n)

    # Calculate recommendations for each cluster
    cluster_recommendations = {}
    for cluster_id in np.unique(clusters):
        cluster_indices = [
            user_input_ids[i] for i in range(len(user_input_ids)) if clusters[i] == cluster_id
        ]
        cluster_vectors = tfidf_matrix[cluster_indices]

        # Calculate similarity for this cluster
        cosine_similarities = cosine_similarity(cluster_vectors, tfidf_matrix)
        avg_similarities = np.mean(cosine_similarities, axis=0)

        cluster_recommendations[cluster_id] = avg_similarities

    # Combine cluster recommendations (weighted by cluster size)
    cluster_sizes = np.bincount(clusters)
    combined_similarities = np.zeros(tfidf_matrix.shape[0])

    for cluster_id, similarities in cluster_recommendations.items():
        weight = cluster_sizes[cluster_id] / len(user_input_ids)
        combined_similarities += weight * similarities

    # Get top recommendations
    similar_indices = np.argsort(combined_similarities)[::-1]

    # Filter out user input posts and get top N
    filtered_results = []
    for idx in similar_indices:
        if idx not in user_input_ids:
            filtered_results.append((idx, combined_similarities[idx]))
        if len(filtered_results) >= top_n:
            break

    # Separate indices and scores
    indices = [result[0] for result in filtered_results]
    scores = [result[1] for result in filtered_results]

    return indices, scores


def get_adaptive_recommendations(user_input_ids, tfidf_matrix, top_n=5):
    """
    Adaptive recommendation function that chooses the best strategy based on
    the number of liked items and their diversity.
    """
    if len(user_input_ids) == 0:
        return [], []

    if len(user_input_ids) == 1:
        # For single item, use simple similarity
        return get_recommendations_with_scores(user_input_ids, tfidf_matrix, top_n)

    # Calculate diversity of liked items
    user_vectors = tfidf_matrix[user_input_ids]
    similarity_matrix = cosine_similarity(user_vectors)

    # Remove diagonal (self-similarity)
    mask = np.eye(similarity_matrix.shape[0], dtype=bool)
    similarities = similarity_matrix[~mask]
    avg_internal_similarity = np.mean(similarities)

    # Choose strategy based on diversity and number of items
    if len(user_input_ids) <= 3:
        # Few items: use weighted recent approach
        return get_recommendations_weighted_recent(user_input_ids, tfidf_matrix, top_n)
    elif avg_internal_similarity > 0.5:
        # High similarity between liked items: use average approach
        return get_recommendations_with_scores(user_input_ids, tfidf_matrix, top_n)
    elif avg_internal_similarity < 0.2:
        # Very diverse interests: use max similarity approach
        return get_recommendations_max_similarity(user_input_ids, tfidf_matrix, top_n)
    else:
        # Moderate diversity: use clustering approach
        return get_recommendations_clustered(user_input_ids, tfidf_matrix, top_n)
