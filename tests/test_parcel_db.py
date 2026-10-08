"""
test_parcel_db.py - Automated test suite for Parcel Database & Directory Storage logic.
"""

import os
import sys
import shutil
import sqlite3
import unittest
from io import BytesIO
from PIL import Image

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import web_server
from web_server import (
    init_db_if_needed,
    get_db_connection,
    save_parcel_picture,
    remove_parcel_picture,
    resolve_picture_path,
)

TEST_DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "_test_parcels.db"))
TEST_PICTURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "_test_parcel_pictures"))


def create_dummy_image_bytes():
    buf = BytesIO()
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestParcelDatabase(unittest.TestCase):
    def setUp(self):
        # Isolate tests to a temporary database file and directory
        self.orig_db = web_server.DB_FILE
        self.orig_pics = web_server.PICTURES_DIR
        web_server.DB_FILE = TEST_DB
        web_server.PICTURES_DIR = TEST_PICTURES_DIR

        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        if os.path.exists(TEST_PICTURES_DIR):
            shutil.rmtree(TEST_PICTURES_DIR, ignore_errors=True)

    def tearDown(self):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        if os.path.exists(TEST_PICTURES_DIR):
            shutil.rmtree(TEST_PICTURES_DIR, ignore_errors=True)
        web_server.DB_FILE = self.orig_db
        web_server.PICTURES_DIR = self.orig_pics

    def test_01_lazy_db_creation(self):
        self.assertFalse(os.path.exists(TEST_DB), "Test DB should NOT exist before classification")

    def test_02_schema_and_upsert(self):
        init_db_if_needed()
        self.assertTrue(os.path.exists(TEST_DB), "Test DB should exist after init")

        conn = get_db_connection()
        cur = conn.cursor()

        # Insert parcel with dummy image saved to directory
        coord = "13.444127, 103.273590"
        dummy_pic = create_dummy_image_bytes()
        saved_rel_path = save_parcel_picture(dummy_pic, coord)

        self.assertIsNotNone(saved_rel_path)
        full_path = resolve_picture_path(saved_rel_path)
        self.assertTrue(os.path.isfile(full_path), "Picture file must be created on disk")

        cur.execute("""
            INSERT INTO parcels (coordinate, picture_path, date, status, flag, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (coord, saved_rel_path, "2026-09-09 00:00:00", "Planted", 1, 0.95))
        conn.commit()

        # Verify row exists
        cur.execute("SELECT coordinate, status, flag, picture_path FROM parcels WHERE coordinate = ?", (coord,))
        row = cur.fetchone()
        self.assertEqual(row["coordinate"], coord)
        self.assertEqual(row["status"], "Planted")
        self.assertEqual(row["flag"], 1)
        self.assertEqual(row["picture_path"], saved_rel_path)

        # Test Primary Key rewrite on conflict: status changes to Flooded (not flagged)
        new_date = "2026-09-09 12:00:00"
        remove_parcel_picture(saved_rel_path)

        cur.execute("""
            INSERT INTO parcels (coordinate, picture_path, date, status, flag, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(coordinate) DO UPDATE SET
                picture_path = excluded.picture_path,
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

        cur.execute("SELECT coordinate, status, flag, picture_path, date FROM parcels WHERE coordinate = ?", (coord,))
        updated_row = cur.fetchone()
        self.assertEqual(updated_row["status"], "Flooded")
        self.assertEqual(updated_row["flag"], 0)
        self.assertIsNone(updated_row["picture_path"], "Picture path should be NULL when flag is 0")
        self.assertEqual(updated_row["date"], new_date)
        self.assertFalse(os.path.exists(full_path), "Old picture file should be removed from disk")

        conn.close()

    def test_03_operator_review_rules(self):
        init_db_if_needed()
        conn = get_db_connection()
        cur = conn.cursor()

        coord1 = "12.000000, 105.000000"
        coord2 = "13.000000, 104.000000"
        pic1_path = save_parcel_picture(create_dummy_image_bytes(), coord1)
        pic2_path = save_parcel_picture(create_dummy_image_bytes(), coord2)

        # Case 1: Flagged parcel reviewed as 'Flooded' -> picture cleared, flag = 0
        cur.execute("""
            INSERT INTO parcels (coordinate, picture_path, date, status, flag, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (coord1, pic1_path, "2026-09-09", "Planted", 1, 0.70))

        # Case 2: Flagged parcel reviewed as 'Planted' -> picture kept, flag = 1
        cur.execute("""
            INSERT INTO parcels (coordinate, picture_path, date, status, flag, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (coord2, pic2_path, "2026-09-09", "Others", 1, 0.60))
        conn.commit()

        # Apply operator decision 1: Change to Flooded -> remove file from disk
        remove_parcel_picture(pic1_path)
        cur.execute("UPDATE parcels SET status = ?, flag = 0, picture_path = NULL WHERE coordinate = ?", ("Flooded", coord1))

        # Apply operator decision 2: Change to Planted -> retain file and flag
        cur.execute("UPDATE parcels SET status = 'Planted', flag = 1 WHERE coordinate = ?", (coord2,))
        conn.commit()

        # Assertions
        cur.execute("SELECT status, flag, picture_path FROM parcels WHERE coordinate = ?", (coord1,))
        r1 = cur.fetchone()
        self.assertEqual(r1["status"], "Flooded")
        self.assertEqual(r1["flag"], 0)
        self.assertIsNone(r1["picture_path"])
        self.assertFalse(os.path.exists(resolve_picture_path(pic1_path)), "Pic 1 should be deleted from disk")

        cur.execute("SELECT status, flag, picture_path FROM parcels WHERE coordinate = ?", (coord2,))
        r2 = cur.fetchone()
        self.assertEqual(r2["status"], "Planted")
        self.assertEqual(r2["flag"], 1)
        self.assertEqual(r2["picture_path"], pic2_path)
        self.assertTrue(os.path.exists(resolve_picture_path(pic2_path)), "Pic 2 must be retained on disk")

        conn.close()

    def test_04_legacy_blob_migration(self):
        # Create legacy database table with picture BLOB
        conn = sqlite3.connect(TEST_DB)
        conn.execute("""
            CREATE TABLE parcels (
                coordinate TEXT PRIMARY KEY,
                picture BLOB,
                date TEXT,
                status TEXT,
                flag INTEGER,
                confidence REAL
            );
        """)
        coord = "11.111111, 106.222222"
        dummy_pic = create_dummy_image_bytes()
        conn.execute("""
            INSERT INTO parcels (coordinate, picture, date, status, flag, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (coord, dummy_pic, "2026-09-10 10:00:00", "Planted", 1, 0.92))
        conn.commit()
        conn.close()

        # Run init_db_if_needed to trigger automatic migration
        init_db_if_needed()

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(parcels)")
        cols = [r["name"] for r in cur.fetchall()]
        self.assertIn("picture_path", cols, "Migrated table must have picture_path column")
        self.assertNotIn("picture", cols, "Migrated table should not have legacy picture column")

        cur.execute("SELECT coordinate, status, flag, picture_path FROM parcels WHERE coordinate = ?", (coord,))
        migrated_row = cur.fetchone()
        self.assertIsNotNone(migrated_row["picture_path"])
        full_path = resolve_picture_path(migrated_row["picture_path"])
        self.assertTrue(os.path.isfile(full_path), "Migrated picture file must exist on disk")
        conn.close()

    def test_05_preserves_filename_and_exif(self):
        """Verifies that saving a parcel picture keeps original filename and retains GPS EXIF."""
        init_db_if_needed()
        from embed_coordinates import embed_gps_metadata
        from src.inference import extract_image_gps

        test_src_path = os.path.join(TEST_PICTURES_DIR, "source_dry_sample.jpg")
        os.makedirs(TEST_PICTURES_DIR, exist_ok=True)

        # Create image with embedded EXIF GPS
        img = Image.new("RGB", (64, 64), color=(34, 139, 34))
        img.save(test_src_path, format="JPEG")
        embed_gps_metadata(test_src_path, lat=11.556400, lon=104.928200)

        # Read GPS before inference
        gps_before = extract_image_gps(test_src_path)
        self.assertIsNotNone(gps_before, "Source image must have valid GPS EXIF")

        # Save through save_parcel_picture with original filename
        saved_rel = save_parcel_picture(test_src_path, coordinate="11.556400, 104.928200", original_filename="source_dry_sample.jpg")
        self.assertTrue(saved_rel.endswith("source_dry_sample.jpg"), f"Saved filename should be source_dry_sample.jpg, got: {saved_rel}")

        saved_full = resolve_picture_path(saved_rel)
        self.assertTrue(os.path.isfile(saved_full), "Saved file must exist on disk")

        # Read GPS after saving
        gps_after = extract_image_gps(saved_full)
        self.assertIsNotNone(gps_after, "GPS EXIF must NOT be lost after saving")
        self.assertAlmostEqual(gps_after["latitude"], gps_before["latitude"], places=4)
        self.assertAlmostEqual(gps_after["longitude"], gps_before["longitude"], places=4)

    def test_06_deletion_logic(self):
        """Verifies deleting parcel removes both DB row and image file from disk."""
        init_db_if_needed()
        conn = get_db_connection()
        cur = conn.cursor()

        coord1 = "11.100000, 105.100000"
        coord2 = "12.200000, 106.200000"
        pic1_path = save_parcel_picture(create_dummy_image_bytes(), coord1, original_filename="field_1.jpg")
        pic2_path = save_parcel_picture(create_dummy_image_bytes(), coord2, original_filename="field_2.jpg")

        cur.execute("INSERT INTO parcels (coordinate, picture_path, date, status, flag, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                    (coord1, pic1_path, "2026-09-10", "Green rice", 1, 0.90))
        cur.execute("INSERT INTO parcels (coordinate, picture_path, date, status, flag, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                    (coord2, pic2_path, "2026-09-10", "Dry", 0, 0.95))
        conn.commit()

        full_pic1 = resolve_picture_path(pic1_path)
        self.assertTrue(os.path.isfile(full_pic1))

        # Delete parcel 1
        remove_parcel_picture(pic1_path)
        cur.execute("DELETE FROM parcels WHERE coordinate = ?", (coord1,))
        conn.commit()

        self.assertFalse(os.path.exists(full_pic1), "Pic 1 file must be removed upon deletion")
        cur.execute("SELECT * FROM parcels WHERE coordinate = ?", (coord1,))
        self.assertIsNone(cur.fetchone(), "Parcel 1 row must be deleted")

        # Parcel 2 should still exist
        cur.execute("SELECT * FROM parcels WHERE coordinate = ?", (coord2,))
        self.assertIsNotNone(cur.fetchone(), "Parcel 2 row must remain")
        conn.close()


if __name__ == "__main__":
    unittest.main()
