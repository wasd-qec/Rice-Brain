"""
test_parcel_db.py - Automated test suite for Parcel Database & Web Server logic.
"""

import os
import sys
import json
import sqlite3
import unittest
from io import BytesIO

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import web_server
from web_server import init_db_if_needed, get_db_connection, get_image_date, get_image_jpeg_bytes
from src.inference import RiceFieldPredictor

TEST_DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "_test_parcels.db"))

class TestParcelDatabase(unittest.TestCase):
    def setUp(self):
        # Isolate tests to a temporary database file so farm_parcels.db is NEVER touched
        self.orig_db = web_server.DB_FILE
        web_server.DB_FILE = TEST_DB
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

    def tearDown(self):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        web_server.DB_FILE = self.orig_db

    def test_01_lazy_db_creation(self):
        self.assertFalse(os.path.exists(TEST_DB), "Test DB should NOT exist before classification")

    def test_02_schema_and_upsert(self):
        init_db_if_needed()
        self.assertTrue(os.path.exists(TEST_DB), "Test DB should exist after init")

        conn = get_db_connection()
        cur = conn.cursor()

        # Insert parcel with dummy coordinate
        coord = "13.444127, 103.273590"
        dummy_pic = b"FAKE_JPEG_BLOB"
        cur.execute("""
            INSERT INTO parcels (coordinate, picture, date, status, flag, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (coord, dummy_pic, "2026-09-09 00:00:00", "Planted", 1, 0.95))
        conn.commit()

        # Verify row exists
        cur.execute("SELECT coordinate, status, flag, picture FROM parcels WHERE coordinate = ?", (coord,))
        row = cur.fetchone()
        self.assertEqual(row["coordinate"], coord)
        self.assertEqual(row["status"], "Planted")
        self.assertEqual(row["flag"], 1)
        self.assertEqual(row["picture"], dummy_pic)

        # Test Primary Key rewrite on conflict
        new_date = "2026-09-09 12:00:00"
        cur.execute("""
            INSERT INTO parcels (coordinate, picture, date, status, flag, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(coordinate) DO UPDATE SET
                picture = excluded.picture,
                date = excluded.date,
                status = excluded.status,
                flag = excluded.flag,
                confidence = excluded.confidence;
        """, (coord, None, new_date, "Flooded", 0, 0.98))
        conn.commit()

        # Check that the row was rewritten and NOT duplicated
        cur.execute("SELECT COUNT(*) FROM parcels")
        total_rows = cur.fetchone()[0]
        self.assertEqual(total_rows, 1, "Coordinate primary key should rewrite row, not duplicate")

        cur.execute("SELECT coordinate, status, flag, picture, date FROM parcels WHERE coordinate = ?", (coord,))
        updated_row = cur.fetchone()
        self.assertEqual(updated_row["status"], "Flooded")
        self.assertEqual(updated_row["flag"], 0)
        self.assertIsNone(updated_row["picture"], "Picture should be NULL when flag is 0")
        self.assertEqual(updated_row["date"], new_date)

        conn.close()

    def test_03_operator_review_rules(self):
        init_db_if_needed()
        conn = get_db_connection()
        cur = conn.cursor()

        coord1 = "12.000000, 105.000000"
        coord2 = "13.000000, 104.000000"

        # Case 1: Flagged parcel reviewed as 'Flooded' -> picture cleared, flag = 0
        cur.execute("""
            INSERT INTO parcels (coordinate, picture, date, status, flag, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (coord1, b"JPEG_DATA_1", "2026-09-09", "Planted", 1, 0.70))

        # Case 2: Flagged parcel reviewed as 'Planted' -> picture kept, flag = 1
        cur.execute("""
            INSERT INTO parcels (coordinate, picture, date, status, flag, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (coord2, b"JPEG_DATA_2", "2026-09-09", "Others", 1, 0.60))
        conn.commit()

        # Apply operator decision 1: Change to Flooded
        cur.execute("UPDATE parcels SET status = ?, flag = 0, picture = NULL WHERE coordinate = ?", ("Flooded", coord1))
        # Apply operator decision 2: Change to Planted
        cur.execute("UPDATE parcels SET status = 'Planted', flag = 1 WHERE coordinate = ?", (coord2,))
        conn.commit()

        # Assertions
        cur.execute("SELECT status, flag, picture FROM parcels WHERE coordinate = ?", (coord1,))
        r1 = cur.fetchone()
        self.assertEqual(r1["status"], "Flooded")
        self.assertEqual(r1["flag"], 0)
        self.assertIsNone(r1["picture"])

        cur.execute("SELECT status, flag, picture FROM parcels WHERE coordinate = ?", (coord2,))
        r2 = cur.fetchone()
        self.assertEqual(r2["status"], "Planted")
        self.assertEqual(r2["flag"], 1)
        self.assertEqual(r2["picture"], b"JPEG_DATA_2")

        conn.close()


if __name__ == "__main__":
    unittest.main()
