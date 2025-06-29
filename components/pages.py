"""
Page components for the ads gallery application.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_and_merge_datasets
from utils.styles import add_custom_css
from components.ui_components import (
    create_gallery_header,
    create_search_and_pagination_controls,
    create_gallery_grid,
    display_page_info,
)
from core.database import get_database
from core.recommender import (
    get_recommendations_with_scores,
    get_recommendations_weighted_recent,
    get_recommendations_max_similarity,
    get_recommendations_clustered,
    get_adaptive_recommendations,
    tfidf_matrix,
)


def show_liked_posts_page():
    """Display the liked posts page showing all posts liked by the current user."""
    add_custom_css()

    # Initialize database and user session
    db = get_database()

    # Use first user for prototype (no session management needed)
    if "user_id" not in st.session_state:
        st.session_state.user_id = db.get_or_create_first_user()

    # Get user's liked post IDs
    liked_ad_ids = db.get_user_likes(st.session_state.user_id)

    # Header
    create_gallery_header()

    # Page title
    st.markdown("### Your Liked Posts")

    if not liked_ad_ids:
        st.info("You haven't liked any posts yet. Go to the main gallery to start liking posts!")
        return

    # Load full dataset
    with st.spinner("Loading your liked posts..."):
        df = load_and_merge_datasets()

    if df is None:
        st.error(
            "Failed to load datasets. Please check if the parquet files exist in the data folder."
        )
        return

    # Filter dataset to only include liked posts
    # Convert ad_id column to string for comparison
    df["ad_id_str"] = df["ad_id"].astype(str)
    liked_df = df[df["ad_id_str"].isin(liked_ad_ids)].copy()

    if len(liked_df) == 0:
        st.warning("Could not find the posts you liked in the current dataset.")
        return

    # Search and pagination controls
    search_term, page_size, page_num = create_search_and_pagination_controls(len(liked_df))

    # Filter liked posts by search term if provided
    if search_term:
        # Create a searchable text column combining title and description
        liked_df["searchable_text"] = (
            liked_df.get("title", "").astype(str)
            + " "
            + liked_df.get("description", "").astype(str)
            + " "
            + liked_df.get("text", "").astype(str)
        ).str.lower()

        search_term_lower = search_term.lower()
        filtered_liked_df = liked_df[
            liked_df["searchable_text"].str.contains(search_term_lower, na=False)
        ]

        st.info(f"Found {len(filtered_liked_df)} liked posts matching '{search_term}'")
    else:
        filtered_liked_df = liked_df

    # Display gallery
    if len(filtered_liked_df) > 0:
        st.markdown(f"**Showing {len(filtered_liked_df)} liked posts**")
        create_gallery_grid(filtered_liked_df, page_size, page_num)
        display_page_info(page_num, page_size, len(filtered_liked_df))
    else:
        st.warning("No liked posts found matching your search criteria.")


def show_main_gallery_page():
    """Display the main gallery page with all posts."""
    add_custom_css()

    # Initialize database and user session
    db = get_database()

    # Use first user for prototype (no session management needed)
    if "user_id" not in st.session_state:
        st.session_state.user_id = db.get_or_create_first_user()

    # Load data
    with st.spinner("Loading datasets..."):
        df = load_and_merge_datasets()

    if df is None:
        st.error(
            "Failed to load datasets. Please check if the parquet files exist in the data folder."
        )
        return

    # Header
    create_gallery_header()

    # Search and pagination controls in the same row
    search_term, page_size, page_num = create_search_and_pagination_controls(len(df))

    # Filter data if search term provided
    from utils.data_loader import filter_data_by_search

    filtered_df = filter_data_by_search(df, search_term)

    if search_term:
        st.info(f"Found {len(filtered_df)} ads matching '{search_term}'")

        # Recalculate pagination for filtered data
        from utils.data_loader import calculate_pagination

        total_pages = calculate_pagination(len(filtered_df), page_size)
        if page_num >= total_pages:
            page_num = 0

    # Display gallery
    st.markdown("### Advertisements")
    if len(filtered_df) > 0:
        create_gallery_grid(filtered_df, page_size, page_num)
        display_page_info(page_num, page_size, len(filtered_df))
    else:
        st.warning("No ads found matching your search criteria.")


def show_recommendations_page():
    """Display the recommendations page showing personalized recommendations based on user's liked posts."""
    add_custom_css()

    # Initialize database and user session
    db = get_database()

    # Use first user for prototype (no session management needed)
    if "user_id" not in st.session_state:
        st.session_state.user_id = db.get_or_create_first_user()

    # Get user's liked post IDs
    liked_ad_ids = db.get_user_likes(st.session_state.user_id)

    # Header
    create_gallery_header()

    # Page title
    st.markdown("### Recommended For You")

    # Load full dataset
    with st.spinner("Loading datasets..."):
        df = load_and_merge_datasets()

    if df is None:
        st.error(
            "Failed to load datasets. Please check if the parquet files exist in the data folder."
        )
        return

    # Get recommendations
    with st.spinner("Generating personalized recommendations..."):
        try:
            if not liked_ad_ids:
                st.info(
                    "Start liking some posts in the main gallery to get personalized recommendations!"
                )
                return

            # Convert liked ad IDs from strings to integers and find their indices in the dataset
            liked_post_indices = []
            df["ad_id_str"] = df["ad_id"].astype(str)

            for ad_id_str in liked_ad_ids:
                try:
                    # Find the ad in the current dataset to get its index
                    matching_rows = df[df["ad_id_str"] == ad_id_str]
                    if not matching_rows.empty:
                        # Get the index of this ad in the dataframe
                        ad_index = matching_rows.index[0]
                        liked_post_indices.append(ad_index)
                except (ValueError, TypeError):
                    # Skip invalid ad IDs
                    continue

            if not liked_post_indices:
                st.warning("Could not find your liked posts in the recommendation dataset.")
                return

            # Recommendation parameters
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.selectbox("Number of recommendations:", [10, 20, 30, 50], index=0)
            with col2:
                strategy = st.selectbox(
                    "Recommendation strategy:",
                    [
                        "Adaptive (Recommended)",
                        "Average Similarity",
                        "Recent Items Weighted",
                        "Max Similarity",
                        "Clustered Interests",
                    ],
                    index=0,
                    help="Choose how to handle multiple liked items. Adaptive automatically selects the best strategy.",
                )

            # Get recommendations using the selected strategy
            if strategy == "Adaptive (Recommended)":
                recommendation_indices, similarity_scores = get_adaptive_recommendations(
                    user_input_ids=liked_post_indices, tfidf_matrix=tfidf_matrix, top_n=top_k
                )
            elif strategy == "Average Similarity":
                recommendation_indices, similarity_scores = get_recommendations_with_scores(
                    user_input_ids=liked_post_indices, tfidf_matrix=tfidf_matrix, top_n=top_k
                )
            elif strategy == "Recent Items Weighted":
                recommendation_indices, similarity_scores = get_recommendations_weighted_recent(
                    user_input_ids=liked_post_indices, tfidf_matrix=tfidf_matrix, top_n=top_k
                )
            elif strategy == "Max Similarity":
                recommendation_indices, similarity_scores = get_recommendations_max_similarity(
                    user_input_ids=liked_post_indices, tfidf_matrix=tfidf_matrix, top_n=top_k
                )
            else:  # Clustered Interests
                recommendation_indices, similarity_scores = get_recommendations_clustered(
                    user_input_ids=liked_post_indices, tfidf_matrix=tfidf_matrix, top_n=top_k
                )

            if not recommendation_indices:
                st.warning("No recommendations available at this time.")
                return

            # Create a dataframe for the recommended ads
            recommended_df = df.iloc[recommendation_indices].copy()

            # Add recommendation rank and similarity scores
            recommended_df["recommendation_rank"] = range(1, len(recommended_df) + 1)
            recommended_df["recommendation_score"] = similarity_scores

            # Display recommendation info and interest analysis
            num_liked = len(liked_post_indices)

            # Calculate interest diversity if multiple items
            diversity_info = ""
            if num_liked > 1:
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np

                user_vectors = tfidf_matrix[liked_post_indices]
                similarity_matrix = cosine_similarity(user_vectors)
                mask = np.eye(similarity_matrix.shape[0], dtype=bool)
                similarities = similarity_matrix[~mask]
                avg_similarity = np.mean(similarities)

                if avg_similarity > 0.5:
                    diversity_level = "Focused"
                    diversity_color = "🎯"
                elif avg_similarity > 0.2:
                    diversity_level = "Moderate"
                    diversity_color = "🎨"
                else:
                    diversity_level = "Diverse"
                    diversity_color = "🌈"

                diversity_info = f" | {diversity_color} Interest diversity: **{diversity_level}** (similarity: {avg_similarity:.2f})"

            st.success(
                f"🎯 Based on your {num_liked} liked posts{diversity_info}, here are your top {len(recommendation_indices)} personalized recommendations using **{strategy}** strategy!"
            )

            # Show similarity score range
            if similarity_scores:
                max_score = max(similarity_scores)
                min_score = min(similarity_scores)
                st.info(f"📊 Similarity scores range from {min_score:.3f} to {max_score:.3f}")

            st.info("📊 Recommendations generated using text similarity analysis")

            # Search and pagination controls
            search_term, page_size, page_num = create_search_and_pagination_controls(
                len(recommended_df)
            )

            # Filter recommendations by search term if provided
            if search_term:
                # Create a searchable text column combining available text fields
                searchable_parts = []

                # Add title if it exists
                if "title" in recommended_df.columns:
                    searchable_parts.append(recommended_df["title"].fillna("").astype(str))
                else:
                    searchable_parts.append(pd.Series([""] * len(recommended_df)))

                # Add description if it exists
                if "description" in recommended_df.columns:
                    searchable_parts.append(recommended_df["description"].fillna("").astype(str))
                else:
                    searchable_parts.append(pd.Series([""] * len(recommended_df)))

                # Add text if it exists
                if "text" in recommended_df.columns:
                    searchable_parts.append(recommended_df["text"].fillna("").astype(str))
                else:
                    searchable_parts.append(pd.Series([""] * len(recommended_df)))

                # Combine all parts
                recommended_df["searchable_text"] = (
                    searchable_parts[0] + " " + searchable_parts[1] + " " + searchable_parts[2]
                ).str.lower()

                search_term_lower = search_term.lower()
                filtered_recommended_df = recommended_df[
                    recommended_df["searchable_text"].str.contains(search_term_lower, na=False)
                ].copy()

                st.info(
                    f"Found {len(filtered_recommended_df)} recommendations matching '{search_term}'"
                )
            else:
                filtered_recommended_df = recommended_df.copy()

            # Display gallery
            if len(filtered_recommended_df) > 0:
                st.markdown(
                    f"**Showing {len(filtered_recommended_df)} personalized recommendations**"
                )
                create_gallery_grid(filtered_recommended_df, page_size, page_num)
                display_page_info(page_num, page_size, len(filtered_recommended_df))

                # Show simple recommendation info
                with st.expander("ℹ️ About These Recommendations", expanded=False):
                    st.write("**Recommendation Method:** TF-IDF Text Similarity")
                    st.write(f"**Based on:** {len(liked_post_indices)} posts you liked")
                    st.write("**Algorithm:** Cosine similarity of text features")
                    st.write(
                        "**Similarity Scores:** Each recommendation shows a similarity percentage indicating how closely it matches your liked content"
                    )
                    st.write(
                        "**How it works:** We analyze the text content of posts you liked and find similar content using TF-IDF vectorization."
                    )
            else:
                st.warning("No recommendations found matching your search criteria.")

        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            st.info("Please try again later or contact support if the problem persists.")
            # Show the actual error for debugging
            st.exception(e)
