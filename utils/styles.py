"""CSS styling for the ads gallery application."""

import streamlit as st


def add_custom_css():
    """Add custom CSS for Instagram-like styling."""
    st.markdown(
        """
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Instagram-like card styling */
    .stImage > img {
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .stImage > img:hover {
        transform: scale(1.02);
    }
    
    /* Header styling */
    .gallery-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    /* Pagination styling */
    .pagination-container {
        text-align: center;
        padding: 2rem 0;
    }
    
    /* Stats styling */
    .stats-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
