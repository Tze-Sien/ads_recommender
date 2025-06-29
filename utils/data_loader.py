"""Data loading and processing utilities for the ads gallery application."""

import streamlit as st
import pandas as pd
from pathlib import Path
from config.gallery_config import DATA_PATH, TRAIN_FILE_1, TRAIN_FILE_2


@st.cache_data
def load_and_merge_datasets():
    """Load both parquet files and merge them together."""
    try:
        # Define data paths
        data_path = Path(DATA_PATH)
        file1 = data_path / TRAIN_FILE_1
        file2 = data_path / TRAIN_FILE_2

        # Load datasets
        df1 = pd.read_parquet(file1)
        df2 = pd.read_parquet(file2)

        # Merge datasets
        df = pd.concat([df1, df2], ignore_index=True)

        # Use sequence index as ad_id
        df["ad_id"] = df.index

        return df
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
        return None


def filter_data_by_search(df, search_term):
    """Filter dataframe by search term in text column."""
    if search_term:
        return df[df["text"].str.contains(search_term, case=False, na=False)]
    return df


def calculate_pagination(total_items, page_size):
    """Calculate total pages based on items and page size."""
    return (total_items - 1) // page_size + 1 if total_items > 0 else 1


def get_page_data(df, page_size, page_num):
    """Get data for specific page."""
    start_idx = page_num * page_size
    end_idx = start_idx + page_size
    return df.iloc[start_idx:end_idx]
