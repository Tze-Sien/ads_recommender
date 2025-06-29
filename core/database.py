"""
Database handler for the ads recommendation system.
Manages SQLite database operations including user likes.
"""

import sqlite3
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseHandler:
    """Handles all database operations for the ads recommendation system."""

    def __init__(self, db_path: str = "ads_recommendation.db"):
        """Initialize database handler with the given database path."""
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database and create necessary tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create users table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create user_likes table to store like interactions
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_likes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        ad_id TEXT NOT NULL,
                        liked BOOLEAN NOT NULL DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id),
                        UNIQUE(user_id, ad_id)
                    )
                """)

                # Create index for faster queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_likes_user_ad 
                    ON user_likes(user_id, ad_id)
                """)

                conn.commit()
                logger.info("Database initialized successfully")

        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def get_or_create_first_user(self) -> int:
        """Get the first user or create one if none exists (prototype mode)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Try to find the first user
                cursor.execute("SELECT user_id FROM users ORDER BY user_id LIMIT 1")
                result = cursor.fetchone()

                if result:
                    return result[0]

                # Create the first user if none exists
                cursor.execute("INSERT INTO users (session_id) VALUES (?)", ("prototype_user",))
                user_id = cursor.lastrowid
                conn.commit()

                logger.info(f"Created first prototype user with ID: {user_id}")
                return user_id

        except sqlite3.Error as e:
            logger.error(f"Error getting/creating first user: {e}")
            raise

    def get_or_create_user(self, session_id: str) -> int:
        """Get existing user or create new user based on session ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Try to find existing user
                cursor.execute("SELECT user_id FROM users WHERE session_id = ?", (session_id,))
                result = cursor.fetchone()

                if result:
                    return result[0]

                # Create new user
                cursor.execute("INSERT INTO users (session_id) VALUES (?)", (session_id,))
                user_id = cursor.lastrowid
                conn.commit()

                logger.info(f"Created new user with ID: {user_id}")
                return user_id

        except sqlite3.Error as e:
            logger.error(f"Error getting/creating user: {e}")
            raise

    def toggle_like(self, user_id: int, ad_id: str, liked: bool) -> bool:
        """Toggle like status for an ad by a user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if liked:
                    # Insert or update like status
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO user_likes (user_id, ad_id, liked, updated_at)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                        (user_id, ad_id, liked),
                    )
                else:
                    # Delete the record if unliked
                    cursor.execute(
                        """
                        DELETE FROM user_likes 
                        WHERE user_id = ? AND ad_id = ?
                    """,
                        (user_id, ad_id),
                    )

                conn.commit()

                action = "liked" if liked else "unliked"
                logger.info(f"User {user_id} {action} ad {ad_id}")
                return True

        except sqlite3.Error as e:
            logger.error(f"Error toggling like: {e}")
            return False

    def get_user_likes(self, user_id: int) -> List[str]:
        """Get all ad IDs that a user has liked."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT ad_id FROM user_likes 
                    WHERE user_id = ? AND liked = 1
                """,
                    (user_id,),
                )

                return [row[0] for row in cursor.fetchall()]

        except sqlite3.Error as e:
            logger.error(f"Error getting user likes: {e}")
            return []

    def is_ad_liked(self, user_id: int, ad_id: str) -> bool:
        """Check if a specific ad is liked by a user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT liked FROM user_likes 
                    WHERE user_id = ? AND ad_id = ?
                """,
                    (user_id, ad_id),
                )

                result = cursor.fetchone()
                return result[0] if result else False

        except sqlite3.Error as e:
            logger.error(f"Error checking if ad is liked: {e}")
            return False

    def get_like_stats(self) -> dict:
        """Get overall like statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Total likes
                cursor.execute("SELECT COUNT(*) FROM user_likes WHERE liked = 1")
                total_likes = cursor.fetchone()[0]

                # Total unique users who liked
                cursor.execute("""
                    SELECT COUNT(DISTINCT user_id) FROM user_likes WHERE liked = 1
                """)
                unique_users = cursor.fetchone()[0]

                # Most liked ads
                cursor.execute("""
                    SELECT ad_id, COUNT(*) as like_count 
                    FROM user_likes 
                    WHERE liked = 1 
                    GROUP BY ad_id 
                    ORDER BY like_count DESC 
                    LIMIT 10
                """)
                most_liked = cursor.fetchall()

                return {
                    "total_likes": total_likes,
                    "unique_users": unique_users,
                    "most_liked_ads": most_liked,
                }

        except sqlite3.Error as e:
            logger.error(f"Error getting like stats: {e}")
            return {}

    def close(self):
        """Close database connection (cleanup method)."""
        # Since we're using context managers, no explicit close needed
        logger.info("Database handler closed")


# Global database instance
db_handler = None


def get_database() -> DatabaseHandler:
    """Get the global database handler instance."""
    global db_handler
    if db_handler is None:
        db_handler = DatabaseHandler()
    return db_handler
