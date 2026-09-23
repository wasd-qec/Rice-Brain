"""
web_server.py - Lightweight REST API & Web Dashboard Server for Rice Farm Parcel Database.

Features:
1. Lazy database initialization: 'farm_parcels.db' is NOT created until /api/classify is triggered.
2. Manages 'parcels' table:
   [coordinate (PRIMARY KEY), picture_path (TEXT), date, status, flag, confidence]
3. Classifies all images in 'Input/':
   - If status in ('Green rice', 'Planted') or confidence < 0.80 -> flag = 1 and saves clean JPEG into 'parcel_pictures/' directory with path in DB.
   - Otherwise -> flag = 0 and picture_path = NULL (lose the picture).
   - If coordinate already exists, replaces the row (coordinate is primary key).
4. Operator review:
   - If decided as 'Green rice' / 'Planted' -> keep flag = 1 and keep picture in storage directory.
   - If decided as other categories -> set flag = 0, delete picture file from disk, and set picture_path = NULL.
"""

import os
import sys
import re
import json
import sqlite3
import urllib.parse
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
from PIL import Image

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model import CLASSES
from src.inference import RiceFieldPredictor, extract_image_gps

DB_FILE = os.path.abspath("farm_parcels.db")
INPUT_DIR = os.path.abspath("Input")
PICTURES_DIR = os.path.abspath("parcel_pictures")
STATIC_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "web")

# Global predictor (lazy-loaded)
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        model_path = "rice_field_classifier.pth"
        if os.path.exists(model_path):
            _predictor = RiceFieldPredictor(model_path)
    return _predictor


def get_db_connection():
    """Connects to SQLite DB with row factory."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def coordinate_to_filename(coordinate):
    """Generates a safe filename based on coordinates (e.g., '11.877700_106.177700.jpg')."""
    safe = re.sub(r"[^0-9a-zA-Z._-]", "_", coordinate.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return f"{safe}.jpg"


def _save_clean_jpeg(img, target_path):
    """Converts image to RGB mode and writes JPEG at quality 92."""
    if img.mode in ('RGBA', 'LA', 'P'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        alpha_img = img.convert('RGBA')
        if 'transparency' in img.info or img.mode in ('RGBA', 'LA'):
            bg.paste(alpha_img, mask=alpha_img.split()[3])
        else:
            bg.paste(alpha_img)
        work_img = bg
    else:
        work_img = img.convert('RGB')
    work_img.save(target_path, format='JPEG', quality=92)


def save_parcel_picture(img_source, coordinate):
    """
    Saves clean JPEG into PICTURES_DIR with filename based on coordinate.
    img_source can be a filesystem path (str) or raw image bytes.
    Returns relative path string (e.g. 'parcel_pictures/11.877700_106.177700.jpg').
    """
    os.makedirs(PICTURES_DIR, exist_ok=True)
    filename = coordinate_to_filename(coordinate)
    full_path = os.path.join(PICTURES_DIR, filename)

    if isinstance(img_source, (bytes, bytearray)):
        with Image.open(BytesIO(img_source)) as img:
            _save_clean_jpeg(img, full_path)
    elif isinstance(img_source, str) and os.path.exists(img_source):
        with Image.open(img_source) as img:
            _save_clean_jpeg(img, full_path)
    else:
        return None

    # Compute path relative to DB_FILE directory for portability
    base_dir = os.path.dirname(DB_FILE)
    return os.path.relpath(full_path, start=base_dir).replace("\\", "/")


def resolve_picture_path(picture_path):
    """Resolves relative or absolute picture_path to an absolute filesystem path."""
    if not picture_path:
        return None
    if os.path.isabs(picture_path):
        return picture_path
    base_dir = os.path.dirname(DB_FILE)
    return os.path.normpath(os.path.join(base_dir, picture_path))


def remove_parcel_picture(picture_path):
    """Deletes image file from disk if it exists."""
    full_path = resolve_picture_path(picture_path)
    if full_path and os.path.isfile(full_path):
        try:
            os.remove(full_path)
        except OSError as e:
            print(f"[!] Failed to remove picture {full_path}: {e}")


def init_db_if_needed():
    """Initializes SQLite database and table if not present, and migrates legacy BLOB schemas."""
    os.makedirs(PICTURES_DIR, exist_ok=True)
    conn = get_db_connection()
    with conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parcels'")
        table_exists = cur.fetchone() is not None

        if not table_exists:
            conn.execute("""
                CREATE TABLE parcels (
                    coordinate TEXT PRIMARY KEY,
                    picture_path TEXT,
                    date TEXT,
                    status TEXT,
                    flag INTEGER,
                    confidence REAL
                );
            """)
        else:
            cur.execute("PRAGMA table_info(parcels)")
            cols = [row["name"] for row in cur.fetchall()]

            # Automatic migration: legacy DB stored BLOB in 'picture'
            if "picture" in cols and "picture_path" not in cols:
                print("[*] Migrating database: moving picture BLOBs to disk directory...")
                cur.execute("SELECT coordinate, picture, date, status, flag, confidence FROM parcels")
                rows = cur.fetchall()
                migrated = []
                for r in rows:
                    pic_blob = r["picture"]
                    coord = r["coordinate"]
                    rel_p = None
                    if pic_blob:
                        try:
                            rel_p = save_parcel_picture(pic_blob, coord)
                        except Exception as e:
                            print(f"[!] Migration error saving image for {coord}: {e}")
                    migrated.append((coord, rel_p, r["date"], r["status"], r["flag"], r["confidence"]))

                cur.execute("DROP TABLE parcels")
                cur.execute("""
                    CREATE TABLE parcels (
                        coordinate TEXT PRIMARY KEY,
                        picture_path TEXT,
                        date TEXT,
                        status TEXT,
                        flag INTEGER,
                        confidence REAL
                    );
                """)
                cur.executemany("""
                    INSERT INTO parcels (coordinate, picture_path, date, status, flag, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, migrated)
                print(f"[*] Successfully migrated {len(migrated)} parcel record(s) to directory storage.")
    conn.close()


def get_image_date(img_path):
    """Extracts date from image EXIF or returns file modification timestamp."""
    try:
        with Image.open(img_path) as img:
            exif = img.getexif()
            # 306 = DateTime, 36867 = DateTimeOriginal
            if 36867 in exif:
                return str(exif[36867])
            if 306 in exif:
                return str(exif[306])
            # GPSDateStamp is tag 29 in GPS IFD
            gps_ifd = exif.get_ifd(0x8825)
            if gps_ifd and 29 in gps_ifd:
                return f"{gps_ifd[29]} 12:00:00"
    except Exception:
        pass
    mtime = os.path.getmtime(img_path)
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")


class ParcelRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/status":
            self.handle_api_status()
        elif path == "/api/parcels":
            self.handle_api_parcels()
        elif path == "/api/picture":
            coord = query.get("coordinate", [None])[0]
            self.handle_api_picture(coord)
        else:
            self.serve_static(path)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/classify":
            self.handle_api_classify()
        elif path == "/api/review":
            self.handle_api_review()
        else:
            self.send_error(404, "Endpoint not found")

    # --------------------------------------------------------------------------
    # API HANDLERS
    # --------------------------------------------------------------------------
    def handle_api_status(self):
        if not os.path.exists(DB_FILE):
            self.send_json(200, {
                "db_exists": False,
                "total_parcels": 0,
                "flagged_count": 0,
                "summary": {c: 0 for c in CLASSES}
            })
            return

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM parcels")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM parcels WHERE flag = 1")
        flagged = cur.fetchone()[0]

        summary = {c: 0 for c in CLASSES}
        cur.execute("SELECT status, COUNT(*) FROM parcels GROUP BY status")
        for row in cur.fetchall():
            if row[0] in summary:
                summary[row[0]] = row[1]

        conn.close()

        self.send_json(200, {
            "db_exists": True,
            "total_parcels": total,
            "flagged_count": flagged,
            "summary": summary
        })

    def handle_api_parcels(self):
        if not os.path.exists(DB_FILE):
            self.send_json(200, {"parcels": [], "db_exists": False})
            return

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT coordinate, date, status, flag, confidence, picture_path
            FROM parcels
            ORDER BY flag DESC, date DESC
        """)
        rows = cur.fetchall()
        parcels = []
        for r in rows:
            pic_p = r["picture_path"]
            abs_p = resolve_picture_path(pic_p) if pic_p else None
            has_pic = bool(pic_p and abs_p and os.path.exists(abs_p))
            parcels.append({
                "coordinate": r["coordinate"],
                "date": r["date"],
                "status": r["status"],
                "flag": bool(r["flag"]),
                "confidence": round(r["confidence"], 4) if r["confidence"] is not None else None,
                "has_picture": has_pic,
                "picture_path": pic_p
            })
        conn.close()

        self.send_json(200, {"parcels": parcels, "db_exists": True})

    def handle_api_picture(self, coordinate):
        if not coordinate or not os.path.exists(DB_FILE):
            self.send_error(404, "Database or coordinate not found")
            return

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT picture_path FROM parcels WHERE coordinate = ?", (coordinate,))
        row = cur.fetchone()
        conn.close()

        if not row or not row["picture_path"]:
            self.send_error(404, "Picture not found or not stored for this coordinate")
            return

        full_path = resolve_picture_path(row["picture_path"])
        if not full_path or not os.path.isfile(full_path):
            self.send_error(404, "Picture file not found on disk")
            return

        try:
            with open(full_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading picture file: {e}")

    def handle_api_classify(self):
        """Scans Input/ directory, classifies images, and populates database."""
        # 1. Initialize DB now
        init_db_if_needed()

        predictor = get_predictor()
        if not predictor:
            self.send_json(500, {"error": "Model weights ('rice_field_classifier.pth') not loaded."})
            return

        if not os.path.exists(INPUT_DIR):
            self.send_json(400, {"error": f"Input directory '{INPUT_DIR}' not found."})
            return

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        image_files = []
        for root, _, files in os.walk(INPUT_DIR):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in valid_exts:
                    image_files.append(os.path.join(root, f))

        if not image_files:
            self.send_json(200, {"message": "No images found in 'Input/' directory.", "classified": 0})
            return

        conn = get_db_connection()
        cur = conn.cursor()
        classified_count = 0
        flagged_count = 0

        for img_p in image_files:
            try:
                # 1. Extract GPS Coordinate
                gps = extract_image_gps(img_p)
                if gps and gps.get("latitude") is not None:
                    coord_str = f"{gps['latitude']:.6f}, {gps['longitude']:.6f}"
                else:
                    # Deterministic fallback coordinate if missing EXIF
                    base_name = os.path.basename(img_p)
                    h = abs(hash(base_name))
                    fallback_lat = 11.500000 + (h % 20000) / 10000.0
                    fallback_lon = 104.800000 + (h % 30000) / 10000.0
                    coord_str = f"{fallback_lat:.6f}, {fallback_lon:.6f}"

                # 2. Date
                date_str = get_image_date(img_p)

                # 3. Classify with AI
                pred = predictor.predict(img_p)
                status = pred["status"]
                conf = pred["confidence"]

                # 4. Flag Condition: 'Green rice' (Planted) OR low confidence (< 0.80)
                is_flagged = (status in ("Green rice", "Planted") or conf < 0.80)

                # Check if coordinate already has a picture saved
                cur.execute("SELECT picture_path FROM parcels WHERE coordinate = ?", (coord_str,))
                existing_row = cur.fetchone()
                existing_pic_path = existing_row["picture_path"] if existing_row else None

                # 5. Picture retention rule: save to directory if flagged, else remove/lose it
                if is_flagged:
                    pic_path = save_parcel_picture(img_p, coord_str)
                    flagged_count += 1
                else:
                    pic_path = None
                    if existing_pic_path:
                        remove_parcel_picture(existing_pic_path)

                # 6. Upsert into database (coordinate is primary key)
                cur.execute("""
                    INSERT INTO parcels (coordinate, picture_path, date, status, flag, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(coordinate) DO UPDATE SET
                        picture_path = excluded.picture_path,
                        date = excluded.date,
                        status = excluded.status,
                        flag = excluded.flag,
                        confidence = excluded.confidence;
                """, (coord_str, pic_path, date_str, status, 1 if is_flagged else 0, conf))

                classified_count += 1
            except Exception as err:
                print(f"[!] Error processing {img_p}: {err}")

        conn.commit()
        conn.close()

        self.send_json(200, {
            "success": True,
            "classified": classified_count,
            "flagged": flagged_count,
            "message": f"Successfully classified {classified_count} parcel(s). {flagged_count} require operator review."
        })

    def handle_api_review(self):
        """Operator reviews a parcel and updates category."""
        if not os.path.exists(DB_FILE):
            self.send_json(400, {"error": "Database not initialized yet."})
            return

        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len)
        try:
            body = json.loads(post_data.decode("utf-8"))
        except Exception:
            self.send_json(400, {"error": "Invalid JSON body"})
            return

        coordinate = body.get("coordinate")
        decision = body.get("decision")

        if not coordinate or decision not in CLASSES:
            self.send_json(400, {"error": f"Valid 'coordinate' and 'decision' ({CLASSES}) required."})
            return

        conn = get_db_connection()
        cur = conn.cursor()

        # Check existing row
        cur.execute("SELECT * FROM parcels WHERE coordinate = ?", (coordinate,))
        row = cur.fetchone()
        if not row:
            conn.close()
            self.send_json(404, {"error": "Coordinate not found in database."})
            return

        # Review Rule:
        # If operator chooses 'Green rice' (or legacy 'Planted'): keep flag = 1 and keep picture path.
        # If operator chooses another class ('Dry', 'Water', 'Wet', 'Green weed', 'Straw', 'Others'): flag = 0, delete picture file from disk, picture_path = NULL.
        if decision in ("Green rice", "Planted"):
            cur.execute("""
                UPDATE parcels
                SET status = ?, flag = 1
                WHERE coordinate = ?;
            """, (decision, coordinate))
            msg = f"Confirmed '{coordinate}' as {decision}. Picture and review flag retained in storage."
        else:
            if row["picture_path"]:
                remove_parcel_picture(row["picture_path"])
            cur.execute("""
                UPDATE parcels
                SET status = ?, flag = 0, picture_path = NULL
                WHERE coordinate = ?;
            """, (decision, coordinate))
            msg = f"Updated '{coordinate}' to {decision}. Picture discarded and review flag cleared."

        conn.commit()
        conn.close()

        self.send_json(200, {
            "success": True,
            "coordinate": coordinate,
            "new_status": decision,
            "message": msg
        })

    # --------------------------------------------------------------------------
    # HELPERS & STATIC SERVING
    # --------------------------------------------------------------------------
    def send_json(self, code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, path):
        if path in ("/", ""):
            filename = "index.html"
        else:
            filename = path.lstrip("/")

        file_path = os.path.join(STATIC_DIR, filename)
        if not os.path.isfile(file_path):
            self.send_error(404, f"File not found: {filename}")
            return

        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".svg": "image/svg+xml"
        }
        content_type = mime_types.get(ext, "application/octet-stream")

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")


def run_server(port=5000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, ParcelRequestHandler)
    print(f"============================================================")
    print(f"[+] Rice Farm Parcel Web Server running on http://localhost:{port}")
    print(f"[+] Static Directory:   {STATIC_DIR}")
    print(f"[+] Pictures Directory: {PICTURES_DIR}")
    print(f"[+] Database Path:      {DB_FILE} (created on first classify)")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Shutting down web server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    run_server(port)
