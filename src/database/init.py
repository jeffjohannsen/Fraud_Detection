#!/usr/bin/env python3
"""
Verify fraud detection database status and display statistics.

This script checks that the SQLite database exists and is ready to use,
then displays current risk distribution statistics. Useful for verifying
setup after cloning the repository or regenerating the database.

USAGE:
  $ python src/database/init.py

  Script will:
  - Verify database file exists
  - Display risk distribution statistics
  - Confirm system is ready to run

OUTPUT:
  - Database location and status
  - Record counts by risk classification
  - Total records in database

WHEN TO RUN:
  - After cloning repository to verify database exists
  - After running regenerate_database.py to check results
  - To quickly check current database statistics

REQUIREMENTS:
  - Database file at data/databases/fraud_detection_local.db
  - Active Python virtual environment
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATABASE_PATH
from src.database.db import LocalDatabase


def main():
    """
    Main execution function for database verification.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    print("\n" + "=" * 70)
    print("DATABASE STATUS - Fraud Detection System")
    print("=" * 70)

    # Locate database file (use centralized config)
    db_path = DATABASE_PATH

    if not db_path.exists():
        print(f"\n✗ Error: Database not found at:")
        print(f"   {db_path}")
        print(f"\n   The database should be included in the repository.")
        print(f"   If missing, run: python scripts/regenerate_database.py")
        print("=" * 70 + "\n")
        return 1

    print(f"\n✓ Database found:")
    print(f"   {db_path}")

    # Get file size
    size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")

    # Connect and retrieve statistics
    try:
        db = LocalDatabase()  # Uses DATABASE_PATH from config
        risk_stats = db.get_risk_counts()

        print(f"\n📊 Risk Distribution:")
        print(f"   Total records:         {risk_stats['record_count']:>6,}")
        print(
            f"   High risk (≥0.10):     {risk_stats['high_count']:>6,}  ({risk_stats['high_perc']:>5.1f}%)"
        )
        print(
            f"   Medium risk (≥0.03):   {risk_stats['med_count']:>6,}  ({risk_stats['med_perc']:>5.1f}%)"
        )
        print(
            f"   Low risk (<0.03):      {risk_stats['low_count']:>6,}  ({risk_stats['low_perc']:>5.1f}%)"
        )

        fraud_predictions = risk_stats["high_count"] + risk_stats["med_count"]
        print(f"\n   Dual-threshold flagged: {fraud_predictions:>6,}  (High + Medium)")

    except Exception as e:
        print(f"\n✗ Error reading database: {e}")
        print("=" * 70 + "\n")
        return 1

    print(f"\n✅ Database ready to use!")
    print(f"   Run Flask app: python run_app.py")
    print(f"   Validate models: python scripts/test_holdout.py")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
