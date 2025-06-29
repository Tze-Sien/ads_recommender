"""UI components for the ads gallery application."""

import streamlit as st
import pandas as pd
from utils.image_utils import decode_image, resize_image_to_square, create_image_placeholder
from config.gallery_config import (
    PAGE_SIZE_OPTIONS,
    COLS_PER_ROW,
    TEXT_PREVIEW_LENGTH,
    DEFAULT_PAGE_SIZE,
)
from core.database import get_database


def handle_like_toggle(ad_id, like_key):
    """Handle like button toggle and update database."""
    # Toggle the like state
    new_like_state = not st.session_state[like_key]
    st.session_state[like_key] = new_like_state

    # Update database if user is logged in
    if "user_id" in st.session_state:
        db = get_database()
        success = db.toggle_like(st.session_state.user_id, str(ad_id), new_like_state)
        if not success:
            # Revert state if database update failed
            st.session_state[like_key] = not new_like_state
            st.error("Failed to update like status. Please try again.")

    st.rerun()


def create_gallery_header():
    """Create the main header for the gallery."""
    st.markdown(
        """
    <div class="gallery-header">
        <h1>Smart Ads Gallery</h1>
    </div>
    """,
        unsafe_allow_html=True,
    )


def create_search_and_pagination_controls(total_items, page_size_options=PAGE_SIZE_OPTIONS):
    """Create search and pagination controls in the same row and return search_term, page_size and page_num."""
    # Create three columns for Search, Page, and Page Size
    col1, col2, col3 = st.columns([2, 1, 1])

    # Search filter
    with col1:
        search_term = st.text_input("Search", key="search")

    # First get page_size to calculate total_pages
    with col3:
        page_size = st.selectbox("Per page", page_size_options, index=1, key="page_size")

    with col2:
        # Calculate total pages after page_size is determined
        total_pages = (total_items - 1) // page_size + 1 if total_items > 0 else 1
        # Page navigation
        page_num = (
            st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                key="page_num",
            )
            - 1
        )

    return search_term, page_size, page_num


def render_ad_card(row):
    """Render a single ad card."""
    # Get database instance
    db = get_database()

    # Create container for Instagram-like card
    with st.container():
        # Try to display image
        image = decode_image(row["image"]["bytes"])

        if image is not None:
            # Resize image to square for Instagram look
            image_resized = resize_image_to_square(image)
            st.image(image_resized)
        else:
            # Placeholder for missing images
            st.markdown(create_image_placeholder(), unsafe_allow_html=True)

        # Ad info
        st.markdown(f"**Ad ID:** {row['ad_id']}")

        # Show recommendation score if available
        if "recommendation_score" in row and pd.notna(row["recommendation_score"]):
            score = row["recommendation_score"]
            # Convert score to percentage and format nicely
            score_percentage = score * 100
            st.markdown(f"**Similarity:** {score_percentage:.1f}%")

        # Create unique key for like state
        like_key = f"liked_{row['ad_id']}"

        # Initialize session state for like if not exists
        # Check database for actual like status
        if like_key not in st.session_state:
            if "user_id" in st.session_state:
                st.session_state[like_key] = db.is_ad_liked(
                    st.session_state.user_id, str(row["ad_id"])
                )
            else:
                st.session_state[like_key] = False

        # Text preview with expandable "See more" functionality
        if pd.notna(row["text"]) and row["text"]:
            full_text = str(row["text"])
            text_preview = (
                full_text[:TEXT_PREVIEW_LENGTH] + "..."
                if len(full_text) > TEXT_PREVIEW_LENGTH
                else full_text
            )

            # Create unique key for this ad's text state
            text_key = f"show_full_text_{row['ad_id']}"

            # Initialize session state for this text if not exists
            if text_key not in st.session_state:
                st.session_state[text_key] = False

            # Display text based on state
            if st.session_state[text_key]:
                # Show full text in a styled container
                st.markdown(
                    f"""
                    <div style="
                        color: #666;
                        font-style: italic;
                        line-height: 1.4;
                        max-height: 200px;
                        overflow-y: auto;
                        padding: 8px;
                        background-color: #f8f9fa;
                        border-radius: 4px;
                        border: 1px solid #e9ecef;
                        margin: 10px 0;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    ">{full_text}</div>
                    """,
                    unsafe_allow_html=True,
                )
                # Bottom buttons row
                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    if st.button("See less", key=f"less_{row['ad_id']}", type="secondary"):
                        st.session_state[text_key] = False
                        st.rerun()
                with btn_col2:
                    like_icon = "❤️" if st.session_state[like_key] else "🤍"
                    if st.button(
                        f"{like_icon} Like", key=f"like_expanded_{row['ad_id']}", type="secondary"
                    ):
                        handle_like_toggle(row["ad_id"], like_key)
            else:
                # Show truncated text
                st.markdown(f"*{text_preview}*")

                # Bottom buttons row - always show Like button at right
                if len(full_text) > TEXT_PREVIEW_LENGTH:
                    btn_col1, btn_col2 = st.columns([1, 1])
                    with btn_col1:
                        if st.button("See more", key=f"more_{row['ad_id']}", type="secondary"):
                            st.session_state[text_key] = True
                            st.rerun()
                    with btn_col2:
                        like_icon = "❤️" if st.session_state[like_key] else "🤍"
                        if st.button(
                            f"{like_icon} Like",
                            key=f"like_collapsed_{row['ad_id']}",
                            type="secondary",
                        ):
                            handle_like_toggle(row["ad_id"], like_key)
                else:
                    # No "See more" needed, but still show Like button on the right
                    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
                    like_icon = "❤️" if st.session_state[like_key] else "🤍"
                    if st.button(
                        f"{like_icon} Like", key=f"like_only_{row['ad_id']}", type="secondary"
                    ):
                        handle_like_toggle(row["ad_id"], like_key)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            # No text, but still show Like button at bottom right
            st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
            like_icon = "❤️" if st.session_state[like_key] else "🤍"
            if st.button(f"{like_icon} Like", key=f"like_no_text_{row['ad_id']}", type="secondary"):
                handle_like_toggle(row["ad_id"], like_key)
            st.markdown("</div>", unsafe_allow_html=True)


def create_gallery_grid(df, page_size=DEFAULT_PAGE_SIZE, page_num=0):
    """Create Instagram-like gallery grid."""
    start_idx = page_num * page_size
    end_idx = start_idx + page_size
    page_data = df.iloc[start_idx:end_idx]

    # Create responsive columns (adjust number based on screen)
    cols_per_row = COLS_PER_ROW

    for i in range(0, len(page_data), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            if i + j < len(page_data):
                row = page_data.iloc[i + j]
                with col:
                    render_ad_card(row)


def display_page_info(page_num, page_size, total_items):
    """Display current page information."""
    start_item = page_num * page_size + 1
    end_item = min((page_num + 1) * page_size, total_items)
    st.markdown(f"*Showing {start_item}-{end_item} of {total_items} ads*")
