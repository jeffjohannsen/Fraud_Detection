#!/usr/bin/env python3
"""
Fraud Detection Web Application Entry Point

Run this script from the project root to start the Flask web app:
    python run_app.py

The app will be available at http://127.0.0.1:5000
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# Import and run the Flask app
from src.webapp.app import APP as app

if __name__ == "__main__":
    print("=" * 70)
    print("FRAUD DETECTION WEB APPLICATION")
    print("=" * 70)
    print("\nStarting Flask development server...")
    print("Access the application at: http://127.0.0.1:5000")
    print("\nAvailable endpoints:")
    print("  /           - Home page with recent predictions")
    print("  /dashboard  - Analytics dashboard")
    print("  /simulate   - Simulate fraud detection on random events")
    print("  /score      - API endpoint for scoring events (POST)")
    print("\nPress CTRL+C to quit")
    print("=" * 70)
    print()

    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
