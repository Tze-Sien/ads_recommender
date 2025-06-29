#!/usr/bin/env python3
"""
Pre-load models to warm up the cache before starting the main application.
This can be run as a separate process to prepare models ahead of time.
"""

import logging
import time

# from recommendation_service import RecommendationService
from core.database import DatabaseHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def init():
    """Pre-load all models and data."""
    start_time = time.time()

    logging.info("Starting model preloading...")

    # Initialize database and ensure tables are created
    db = DatabaseHandler()
    logging.info("Database initialized and tables created")

    # Create first user if none exists (prototype mode)
    first_user_id = db.get_or_create_first_user()
    logging.info(f"First user initialized with ID: {first_user_id}")

    # Initialize recommendation service (this will load all models)
    # recommender = RecommendationService()

    load_time = time.time() - start_time
    logging.info(f"Models preloaded successfully in {load_time:.2f} seconds")

    return db


if __name__ == "__main__":
    init()
