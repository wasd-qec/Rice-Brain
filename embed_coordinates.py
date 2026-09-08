"""
embed_coordinates.py

Utility script to embed and read standard EXIF GPS coordinates into JPEG (and PNG) images.
Defaults to generating realistic farmland coordinates across Cambodia's major rice-growing provinces
(e.g., Battambang, Takeo, Prey Veng, Siem Reap, Pursat, Kampong Cham).
"""

import os
import sys
import random
from fractions import Fraction
from datetime import datetime
from PIL import Image
from PIL.ExifTags import IFD

# Bounding boxes for major rice-growing provinces in Cambodia
CAMBODIA_REGIONS = {
    "Battambang": {"lat": (12.85, 13.35), "lon": (102.90, 103.45), "alt": (10.0, 25.0)},
    "Takeo": {"lat": (10.80, 11.25), "lon": (104.60, 105.10), "alt": (5.0, 18.0)},
    "Prey Veng": {"lat": (11.20, 11.75), "lon": (105.15, 105.65), "alt": (6.0, 15.0)},
    "Siem Reap": {"lat": (13.15, 13.60), "lon": (103.70, 104.20), "alt": (12.0, 30.0)},
    "Pursat": {"lat": (12.30, 12.75), "lon": (103.60, 104.15), "alt": (15.0, 35.0)},
    "Kampong Cham": {"lat": (11.80, 12.25), "lon": (105.20, 105.80), "alt": (8.0, 22.0)},
    "Banteay Meanchey": {"lat": (13.40, 13.85), "lon": (102.80, 103.30), "alt": (14.0, 28.0)},
    "Kandal": {"lat": (11.30, 11.70), "lon": (104.80, 105.25), "alt": (7.0, 16.0)},
}

def to_dms_fractions(deg_float):
    """Converts a decimal degree float into (degrees, minutes, seconds) Fraction tuples."""
    deg = int(deg_float)
    min_float = (deg_float - deg) * 60
    minute = int(min_float)
    sec_float = (min_float - minute) * 60
    return (
        Fraction(deg, 1),
        Fraction(minute, 1),
        Fraction(int(round(sec_float * 10000)), 10000)
    )

def generate_random_cambodia_coord():
    """Selects a random rice-growing region in Cambodia and generates coordinates."""
    region_name = random.choice(list(CAMBODIA_REGIONS.keys()))
    reg = CAMBODIA_REGIONS[region_name]
    lat = round(random.uniform(reg["lat"][0], reg["lat"][1]), 6)
    lon = round(random.uniform(reg["lon"][0], reg["lon"][1]), 6)
    alt = round(random.uniform(reg["alt"][0], reg["alt"][1]), 2)
    return lat, lon, alt, region_name

def embed_gps_metadata(image_path, lat=None, lon=None, alt=None, region_hint=None, save_path=None):
    """
    Embeds standard EXIF GPS tags into an image file.
    If lat/lon/alt are not provided, generates random coordinates in Cambodia.
    """
    if lat is None or lon is None:
        lat, lon, alt, region_hint = generate_random_cambodia_coord()
    elif alt is None:
        alt = round(random.uniform(10.0, 30.0), 2)

    save_path = save_path or image_path
    
    with Image.open(image_path) as img:
        # Convert RGBA/P to RGB if saving as JPEG
        fmt = os.path.splitext(save_path)[1].lower()
        if fmt in ('.jpg', '.jpeg') and img.mode in ('RGBA', 'LA', 'P'):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            alpha_img = img.convert('RGBA')
            if 'transparency' in img.info or img.mode in ('RGBA', 'LA'):
                bg.paste(alpha_img, mask=alpha_img.split()[3])
            else:
                bg.paste(alpha_img)
            work_img = bg
        else:
            work_img = img.copy()

        exif = work_img.getexif()
        gps_ifd = exif.get_ifd(IFD.GPSInfo)

        # Standard EXIF GPS Tags
        gps_ifd[0] = b'\x02\x02\x00\x00'                     # GPSVersionID
        gps_ifd[1] = 'N' if lat >= 0 else 'S'                # GPSLatitudeRef
        gps_ifd[2] = to_dms_fractions(abs(lat))              # GPSLatitude
        gps_ifd[3] = 'E' if lon >= 0 else 'W'                # GPSLongitudeRef
        gps_ifd[4] = to_dms_fractions(abs(lon))              # GPSLongitude
        gps_ifd[5] = b'\x00'                                 # GPSAltitudeRef (0 = Above Sea Level)
        gps_ifd[6] = Fraction(int(round(alt * 100)), 100)    # GPSAltitude
        gps_ifd[29] = datetime.now().strftime("%Y:%m:%d")    # GPSDateStamp

        exif[0x8825] = gps_ifd

        if fmt in ('.jpg', '.jpeg'):
            work_img.save(save_path, 'JPEG', quality=95, exif=exif)
        else:
            work_img.save(save_path, exif=exif)

    return {
        "file": save_path,
        "latitude": lat,
        "longitude": lon,
        "altitude": alt,
        "region": region_hint,
        "google_maps": f"https://www.google.com/maps?q={lat},{lon}"
    }

def read_gps_metadata(image_path):
    """
    Reads EXIF GPS tags from an image and returns decimal coordinates.
    """
    with Image.open(image_path) as img:
        exif = img.getexif()
        gps_ifd = exif.get_ifd(IFD.GPSInfo)
        if not gps_ifd or 2 not in gps_ifd or 4 not in gps_ifd:
            return None

        lat_dms = gps_ifd[2]
        lat_ref = gps_ifd.get(1, 'N')
        lon_dms = gps_ifd[4]
        lon_ref = gps_ifd.get(3, 'E')
        alt_val = gps_ifd.get(6, None)

        lat = float(lat_dms[0]) + float(lat_dms[1]) / 60.0 + float(lat_dms[2]) / 3600.0
        if lat_ref == 'S':
            lat = -lat

        lon = float(lon_dms[0]) + float(lon_dms[1]) / 60.0 + float(lon_dms[2]) / 3600.0
        if lon_ref == 'W':
            lon = -lon

        alt = float(alt_val) if alt_val is not None else None

        return {
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "altitude": round(alt, 2) if alt is not None else None,
            "google_maps": f"https://www.google.com/maps?q={round(lat, 6)},{round(lon, 6)}"
        }

def embed_directory(directory_path, recursive=True):
    """Scans directory and embeds GPS metadata into all JPEG/PNG images."""
    supported = ('.jpg', '.jpeg', '.png')
    results = []

    for root, _, files in os.walk(directory_path):
        for f in sorted(files):
            if f.lower().endswith(supported):
                img_p = os.path.join(root, f)
                info = embed_gps_metadata(img_p)
                results.append(info)
                print(f"[+] {img_p} -> {info['latitude']} N, {info['longitude']} E ({info['region']}, {info['altitude']}m)")
        if not recursive:
            break

    return results

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "Dataset"
    print(f"[*] Embedding random Cambodia GPS coordinates into '{target}'...")
    if os.path.isdir(target):
        res = embed_directory(target)
        print(f"\n[OK] Successfully embedded GPS coordinates into {len(res)} image(s)!")
    elif os.path.isfile(target):
        info = embed_gps_metadata(target)
        print(f"[OK] Successfully embedded GPS into {target}:")
        print(f"     Coords: {info['latitude']}, {info['longitude']} ({info['region']})")
        print(f"     Maps:   {info['google_maps']}")
    else:
        print(f"[!] Target '{target}' not found.")
