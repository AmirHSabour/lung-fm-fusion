"""Measure the true pixel size (microns per pixel) of each CLWD JPG.

The Hugging Face JPGs are downsampled copies of the original Aperio SVS scans, by a
factor that differs per slide (D3). We read the original's level-0 size and MPP from
its TIFF header (HTTP range requests, a few KB per slide) and the JPG's size from its
own header, then: jpg_mpp = svs_mpp * svs_width / jpg_width.
"""
from __future__ import annotations

import re
import struct
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # header read only; the slides are legitimately huge


def _range(url: str, off: int, n: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + n - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _read_first_ifd(url: str) -> tuple[dict, int]:
    """Width, height and description tags of a remote TIFF's first IFD, plus the file size."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-15"})
    with urllib.request.urlopen(req, timeout=60) as r:
        head = r.read()
        total = int(r.headers["Content-Range"].split("/")[1])
    if head[:2] != b"II":
        raise ValueError("only little-endian TIFF is handled")
    big = head[2:4] == b"+\x00"  # BigTIFF
    off = struct.unpack("<Q", head[8:16])[0] if big else struct.unpack("<I", head[4:8])[0]
    if big:
        n, esz, cfmt, ptr = struct.unpack("<Q", _range(url, off, 8))[0], 20, "<HHQ", "<Q"
        raw = _range(url, off + 8, n * esz)
    else:
        n, esz, cfmt, ptr = struct.unpack("<H", _range(url, off, 2))[0], 12, "<HHI", "<I"
        raw = _range(url, off + 2, n * esz)
    hdr = 12 if big else 8
    tags: dict[int, object] = {}
    for i in range(n):
        e = raw[i * esz:(i + 1) * esz]
        tag, typ, cnt = struct.unpack(cfmt, e[:hdr])
        val = e[hdr:]
        if tag in (256, 257):  # ImageWidth / ImageLength (SHORT, LONG or LONG8)
            fmt = {3: "<H", 4: "<I", 16: "<Q"}[typ]
            tags[tag] = struct.unpack(fmt, val[:struct.calcsize(fmt)])[0]
        elif tag == 270:  # ImageDescription (ASCII), inline or at an offset
            data = val[:cnt] if cnt <= len(val) else _range(url, struct.unpack(ptr, val)[0], cnt)
            tags[tag] = data.split(b"\0")[0].decode(errors="replace")
    return tags, total


def svs_header(base_url: str, name: str, retries: int = 3) -> dict:
    """Level-0 width/height, MPP, AppMag and file size of a remote SVS; svs_error on failure."""
    url = base_url + urllib.parse.quote(name)
    err = "no attempt made"
    for _ in range(max(retries, 1)):
        try:
            tags, total = _read_first_ifd(url)
            desc = str(tags.get(270, ""))
            mpp = re.search(r"MPP\s*=\s*([\d.]+)", desc)
            mag = re.search(r"AppMag\s*=\s*([\d.]+)", desc)
            return dict(svs_width=tags[256], svs_height=tags[257], svs_bytes=total,
                        svs_mpp=float(mpp.group(1)) if mpp else None,
                        svs_appmag=float(mag.group(1)) if mag else None)
        except Exception as exc:  # noqa: BLE001 - network hiccup or missing file; retried
            err = repr(exc)
    return dict(svs_error=err)


def jpg_header(path: Path) -> dict:
    """Width, height and file size of a local JPG (header only; nothing is decoded)."""
    if not path.exists():
        return {}
    with Image.open(path) as im:
        w, h = im.size
    return dict(jpg_width=w, jpg_height=h, jpg_bytes=path.stat().st_size)
