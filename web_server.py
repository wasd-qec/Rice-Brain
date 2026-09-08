"""
web_server.py - Lightweight REST API & Web Dashboard Server for Rice Farm Parcel Database.

Features:
1. Lazy database initialization: 'farm_parcels.db' is NOT created until /api/classify is triggered.
2. Manages 'parcels' table:
   [coordinate (PRIMARY KEY), picture (BLOB), date, status, flag, confidence]
3. Classifies all images in 'Input/':
   - If status == 'Planted' or confidence < 0.80 -> flag = 1 and saves JPEG BLOB into DB.
   - Otherwise -> flag = 0 and picture = NULL (lose the picture).
   - If coordinate already exists, replaces the row (coordinate is primary key).
4. Operator review:
   - If decided as 'Planted' -> keep flag = 1 and keep picture.
   - If decided as 'Dry'/'Flooded'/'Others' -> set flag = 0 and set picture = NULL.
"""

import os
import sys
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


def init_db_if_needed():
    """Initializes the SQLite database and table if not present."""
    conn = get_db_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS parcels (
                coordinate TEXT PRIMARY KEY,
                picture BLOB,
                date TEXT,
                status TEXT,
                flag INTEGER,
                confidence REAL
            );
        """)
    conn.close()


def get_image_jpeg_bytes(img_path):
    """Reads and returns clean JPEG bytes for storing as a BLOB."""
    with Image.open(img_path) as img:
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

        buf = BytesIO()
        work_img.save(buf, format='JPEG', quality=92)
        return buf.getvalue()


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
            SELECT coordinate, date, status, flag, confidence,
                   CASE WHEN picture IS NOT NULL THEN 1 ELSE 0 END AS has_picture
            FROM parcels
            ORDER BY flag DESC, date DESC
        """)
        rows = cur.fetchall()
        parcels = [
            {
                "coordinate": r["coordinate"],
                "date": r["date"],
                "status": r["status"],
                "flag": bool(r["flag"]),
                "confidence": round(r["confidence"], 4) if r["confidence"] is not None else None,
                "has_picture": bool(r["has_picture"])
            }
            for r in rows
        ]
        conn.close()

        self.send_json(200, {"parcels": parcels, "db_exists": True})

    def handle_api_picture(self, coordinate):
        if not coordinate or not os.path.exists(DB_FILE):
            self.send_error(404, "Database or coordinate not found")
            return

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT picture FROM parcels WHERE coordinate = ?", (coordinate,))
        row = cur.fetchone()
        conn.close()

        if not row or not row["picture"]:
            self.send_error(404, "Picture not found or not stored for this coordinate")
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(row["picture"])))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(row["picture"])

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

                # 4. Flag Condition: 'Planted' OR low confidence (< 0.80)
                is_flagged = (status == "Planted" or conf < 0.80)

                # 5. Picture retention rule: keep if flagged, else lose it (None)
                if is_flagged:
                    pic_bytes = get_image_jpeg_bytes(img_p)
                    flagged_count += 1
                else:
                    pic_bytes = None

                # 6. Upsert into database (coordinate is primary key)
                cur.execute("""
                    INSERT INTO parcels (coordinate, picture, date, status, flag, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(coordinate) DO UPDATE SET
                        picture = excluded.picture,
                        date = excluded.date,
                        status = excluded.status,
                        flag = excluded.flag,
                        confidence = excluded.confidence;
                """, (coord_str, pic_bytes, date_str, status, 1 if is_flagged else 0, conf))

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
        # If operator chooses 'Planted': keep flag = 1 and keep picture.
        # If operator chooses another class ('Dry', 'Flooded', 'Others'): flag = 0 and picture = NULL.
        if decision == "Planted":
            cur.execute("""
                UPDATE parcels
                SET status = 'Planted', flag = 1
                WHERE coordinate = ?;
            """, (coordinate,))
            msg = f"Confirmed '{coordinate}' as Planted. Picture and review flag retained in DB."
        else:
            cur.execute("""
                UPDATE parcels
                SET status = ?, flag = 0, picture = NULL
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
    print(f"[+] Static Directory: {STATIC_DIR}")
    print(f"[+] Database Path:    {DB_FILE} (created on first classify)")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Shutting down web server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    run_server(port)
