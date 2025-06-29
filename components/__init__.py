"""Components package for the ads gallery application."""

from .ui_components import (
    create_gallery_header,
    create_search_and_pagination_controls,
    create_gallery_grid,
    display_page_info,
    render_ad_card,
)

__all__ = [
    "create_gallery_header",
    "create_search_and_pagination_controls",
    "create_gallery_grid",
    "display_page_info",
    "render_ad_card",
]
