"""Download the open datasets a case study is built from into a raw cache.

Layout of the cache (``--raw-dir``, default ``<repo>/.casestudy_raw``)::

    amazon/route_data.json      Amazon Last Mile Routing Research Challenge (CC BY-NC 4.0)
    osm/<region>-latest.osm.pbf Geofabrik OpenStreetMap extract (ODbL)
    afdc/<ST>.json              NREL/NLR Alternative Fuels Data Center, public DC-fast chargers
    terrain/<z>/<x>/<y>.png     AWS terrain tiles (Terrarium encoding; 3DEP/NED over the US)

Everything is fetched only when missing, so re-running is cheap.
"""
from __future__ import annotations

import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AMAZON_URL = (
    "https://amazon-last-mile-challenges.s3.us-west-2.amazonaws.com/almrrc2021/"
    "almrrc2021-data-training/model_build_inputs/route_data.json"
)
GEOFABRIK = "https://download.geofabrik.de/{region}-latest.osm.pbf"
AFDC_URL = (
    "https://developer.nlr.gov/api/alt-fuel-stations/v1.json?api_key={key}"
    "&fuel_type=ELEC&state={state}&status=E&limit=all"
)
TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TERRAIN_ZOOM = 13  # ~13 m/pixel at 47 N; the US layer is 10 m NED/3DEP


def default_raw_dir() -> Path:
    env = os.environ.get("EVRP_CASESTUDY_RAW")
    return Path(env) if env else Path(__file__).resolve().parents[2] / ".casestudy_raw"


def _download(url: str, dest: Path, retries: int = 8) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "evrp-casestudy-builder/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001 - retry any network error (DNS hiccups happen)
            last = e
            time.sleep(min(30, 1.5 ** attempt))
    raise RuntimeError(f"download failed: {url}: {last}")


def ensure_amazon(raw: Path) -> Path:
    p = raw / "amazon" / "route_data.json"
    if not p.exists():
        print("fetching Amazon route data (~75 MB)")
        _download(AMAZON_URL, p)
    return p


def ensure_osm(raw: Path, region: str) -> Path:
    """Geofabrik extract for one region such as north-america/us/washington."""
    p = raw / "osm" / (region.split("/")[-1] + "-latest.osm.pbf")
    if not p.exists():
        print(f"fetching Geofabrik extract {region} (hundreds of MB)")
        _download(GEOFABRIK.format(region=region), p)
    return p


def ensure_afdc(raw: Path, state: str) -> Path:
    """AFDC electric stations of one state (two-letter code)."""
    p = raw / "afdc" / f"{state}.json"
    if not p.exists():
        key = os.environ.get("NREL_API_KEY", "DEMO_KEY")
        print(f"fetching AFDC electric stations for {state}")
        _download(AFDC_URL.format(key=key, state=state), p)
    return p


def tile_xy(lat: float, lon: float, z: int) -> tuple[float, float]:
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def ensure_terrain(raw: Path, bbox: tuple[float, float, float, float], z: int = TERRAIN_ZOOM) -> tuple[int, int, int, int]:
    """Fetch every terrain tile covering bbox=(min_lat, min_lon, max_lat, max_lon).

    Returns the tile index range (x0, x1, y0, y1), inclusive.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    x0, y1 = (int(v) for v in tile_xy(min_lat, min_lon, z))
    x1, y0 = (int(v) for v in tile_xy(max_lat, max_lon, z))
    todo = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
            if not (raw / "terrain" / str(z) / str(x) / f"{y}.png").exists()]
    if todo:
        print(f"fetching {len(todo)} terrain tiles")

        def get(xy):
            x, y = xy
            _download(TERRAIN_URL.format(z=z, x=x, y=y), raw / "terrain" / str(z) / str(x) / f"{y}.png")

        with ThreadPoolExecutor(8) as ex:
            list(ex.map(get, todo))
    return x0, x1, y0, y1


if __name__ == "__main__":
    import yaml

    preset = yaml.safe_load(open(sys.argv[1]))
    raw = Path(sys.argv[2]) if len(sys.argv) > 2 else default_raw_dir()
    ensure_amazon(raw)
    ensure_osm(raw, preset["geofabrik"])
    ensure_afdc(raw, preset["state"])
    print("raw cache ready at", raw)
