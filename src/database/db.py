"""
Local SQLite database adapter for fraud detection system.
Replaces AWS RDS PostgreSQL for local development and demos.
"""

import sqlite3
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DATABASE_PATH


class LocalDatabase:
    """Handles SQLite database operations for fraud detection records."""

    def __init__(self, db_path=None):
        """
        Initialize database connection.

        Args:
            db_path (str or Path, optional): Path to SQLite database file.
                If None, uses default path from src.config.DATABASE_PATH
        """
        if db_path is None:
            db_path = DATABASE_PATH

        # Convert to string if Path object
        self.db_path = str(db_path) if isinstance(db_path, Path) else db_path

        # Ensure parent directory exists
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self._initialize_schema()

    def _initialize_schema(self):
        """Create fraud_records table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS fraud_records (
            object_id INTEGER PRIMARY KEY,
            fraud_proba REAL,
            record_predicted_datetime TEXT,
            event_created TEXT,
            country TEXT,
            avg_ticket_cost REAL,
            total_ticket_value REAL,
            body_length INTEGER,
            channels INTEGER,
            delivery_method INTEGER,
            fb_published INTEGER,
            gts INTEGER,
            has_analytics INTEGER,
            has_header INTEGER,
            has_logo INTEGER,
            listed INTEGER,
            name_length INTEGER,
            num_order INTEGER,
            num_payouts INTEGER,
            org_facebook INTEGER,
            org_twitter INTEGER,
            user_type INTEGER,
            sale_duration INTEGER,
            sale_duration2 INTEGER,
            show_map INTEGER,
            user_age INTEGER,
            venue_latitude REAL,
            venue_longitude REAL,
            num_previous_payouts INTEGER,
            previous_payouts_total REAL,
            num_ticket_types INTEGER,
            num_tickets_available INTEGER,
            known_payee_name INTEGER,
            known_venue_name INTEGER,
            known_payout_type INTEGER,
            total_empty_values INTEGER,
            name_proba REAL,
            description_proba REAL,
            org_name_proba REAL,
            org_desc_proba REAL,
            name TEXT,
            description TEXT,
            org_name TEXT,
            org_desc TEXT
        )
        """
        with self.engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()

    def load_csv_data(self, csv_path):
        """
        Load fraud records from CSV file into database.

        Args:
            csv_path (str): Path to CSV file

        Returns:
            int: Number of records loaded
        """
        df = pd.read_csv(csv_path)

        # Load into SQLite, replacing existing data
        df.to_sql("fraud_records", self.engine, if_exists="replace", index=False)

        return len(df)

    def add_record(self, record_dict):
        """
        Add a single fraud record to the database.

        Args:
            record_dict (dict): Record data as dictionary

        Returns:
            bool: True if successful
        """
        df = pd.DataFrame([record_dict])
        df.to_sql("fraud_records", self.engine, if_exists="append", index=False)
        return True

    def get_record_count(self):
        """Get total number of records in database."""
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM fraud_records"))
            return result.fetchone()[0]

    def get_risk_counts(self):
        """
        Get counts and percentages of records by risk level.

        Returns:
            dict: Contains counts and percentages for high, medium, low risk
        """
        with self.engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM fraud_records")).fetchone()[
                0
            ]

            high = conn.execute(
                text("SELECT COUNT(*) FROM fraud_records WHERE fraud_proba >= 0.10")
            ).fetchone()[0]

            med = conn.execute(
                text(
                    "SELECT COUNT(*) FROM fraud_records WHERE fraud_proba >= 0.03 AND fraud_proba < 0.10"
                )
            ).fetchone()[0]

            low = conn.execute(
                text("SELECT COUNT(*) FROM fraud_records WHERE fraud_proba < 0.03")
            ).fetchone()[0]

            if total > 0:
                high_perc = round((high / total) * 100, 1)
                med_perc = round((med / total) * 100, 1)
                low_perc = round((low / total) * 100, 1)
            else:
                high_perc = med_perc = low_perc = 0

            return {
                "record_count": total,
                "high_count": high,
                "med_count": med,
                "low_count": low,
                "high_perc": high_perc,
                "med_perc": med_perc,
                "low_perc": low_perc,
            }

    def get_all_records(self, order_by="fraud_proba DESC"):
        """
        Get all fraud records from database.

        Args:
            order_by (str): SQL ORDER BY clause

        Returns:
            list: List of record tuples
        """
        query = f"""
            SELECT object_id,
                   event_created,
                   country,
                   avg_ticket_cost,
                   total_ticket_value,
                   fraud_proba,
                   CASE
                       WHEN fraud_proba >= 0.10 THEN 'High'
                       WHEN fraud_proba >= 0.03 AND fraud_proba < 0.10 THEN 'Medium'
                       ELSE 'Low'
                   END AS fraud_risk_level
            FROM fraud_records
            ORDER BY {order_by}, country
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return result.fetchall()

    def record_exists(self, object_id):
        """
        Check if a record with given object_id exists.

        Args:
            object_id (int): Record ID to check

        Returns:
            bool: True if exists
        """
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM fraud_records WHERE object_id = :object_id"),
                {"object_id": object_id},
            )
            return result.fetchone()[0] > 0


if __name__ == "__main__":
    # Test the database
    db = LocalDatabase()
    print(f"Database initialized at: {db.db_path}")
    print(f"Record count: {db.get_record_count()}")

    # Try loading CSV if it exists
    from src.config import DATA_RAW

    csv_path = DATA_RAW / "fraud_records_db_table.csv"
    if csv_path.exists():
        print(f"\nLoading data from {csv_path}...")
        count = db.load_csv_data(csv_path)
        print(f"Loaded {count} records")

        risk_stats = db.get_risk_counts()
        print(f"\nRisk Statistics:")
        print(f"  High risk: {risk_stats['high_count']} ({risk_stats['high_perc']}%)")
        print(f"  Medium risk: {risk_stats['med_count']} ({risk_stats['med_perc']}%)")
        print(f"  Low risk: {risk_stats['low_count']} ({risk_stats['low_perc']}%)")
    else:
        print(f"\nCSV file not found at {csv_path}")
