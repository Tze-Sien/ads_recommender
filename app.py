"""
Ads Gallery Application

A Streamlit application for displaying advertisement data in an Instagram-like gallery format.
"""

import streamlit as st
from components.pages import (
    show_main_gallery_page,
    show_liked_posts_page,
    show_recommendations_page,
)
from config.gallery_config import APP_TITLE, APP_ICON

# Set page config for wide layout
st.set_page_config(
    page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded"
)


def main():
    """Main application with navigation between pages."""
    # Initialize session state for page navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Gallery"

    # Create navigation in sidebar
    st.sidebar.title("Navigation")

    # Navigation buttons
    if st.sidebar.button("🏠 Main Gallery", use_container_width=True):
        st.session_state.current_page = "Gallery"
        st.rerun()

    if st.sidebar.button("❤️ Liked Posts", use_container_width=True):
        st.session_state.current_page = "Liked Posts"
        st.rerun()

    if st.sidebar.button("🎯 Recommendations", use_container_width=True):
        st.session_state.current_page = "Recommendations"
        st.rerun()

    # Display current page indicator in sidebar
    st.sidebar.markdown(f"**Current Page:** {st.session_state.current_page}")

    # Route to appropriate page
    if st.session_state.current_page == "Gallery":
        show_main_gallery_page()
    elif st.session_state.current_page == "Liked Posts":
        show_liked_posts_page()
    elif st.session_state.current_page == "Recommendations":
        show_recommendations_page()


if __name__ == "__main__":
    main()
