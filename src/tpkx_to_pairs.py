"""Converte dois pacotes de tiles do Esri Wayback (.tpkx) em um par de imagens alinhadas.

Uso:
  python src/tpkx_to_pairs.py antes.tpkx depois.tpkx --antes-data 2014-12-03 --depois-data 2026-04-30
Gera data/niteroi/A/niteroi.png (antes), data/niteroi/B/niteroi.png (depois) e meta.json
(com os metros por pixel reais, usados pelo infer.py). Usa só os tiles presentes nos dois pacotes.
"""
import argparse, io, json, math, re, struct, zipfile
from pathlib import Path

from PIL import Image

ORIGIN = 20037508.342787  # origem do esquema de tiles (Web Mercator, EPSG:3857)
TILE = 256


def read_tiles(tpkx, level):
    """Lê os tiles de um nível: {(linha, coluna): bytes JPEG}. Formato: compact cache v2."""
    tiles = {}
    with zipfile.ZipFile(tpkx) as z:
        for name in z.namelist():
            m = re.search(rf"tile/L{level:02d}/R([0-9a-f]+)C([0-9a-f]+)\.bundle$", name)
            if not m:
                continue
            r0, c0 = int(m[1], 16), int(m[2], 16)  # linha/coluna inicial do bundle (128x128 tiles)
            data = z.read(name)
            for i, v in enumerate(struct.unpack("<16384Q", data[64:64 + 16384 * 8])):
                size, off = v >> 40, v & 0xFFFFFFFFFF  # 24 bits de tamanho + 40 bits de offset
                if size:
                    tiles[(r0 + i // 128, c0 + i % 128)] = data[off:off + size]
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("antes"); ap.add_argument("depois")
    ap.add_argument("--level", type=int, default=18)
    ap.add_argument("--antes-data", default="?"); ap.add_argument("--depois-data", default="?")
    ap.add_argument("--out", default="data/niteroi")
    a = ap.parse_args()

    ta, tb = read_tiles(a.antes, a.level), read_tiles(a.depois, a.level)
    comum = sorted(set(ta) & set(tb))
    if not comum:
        raise SystemExit("Nenhum tile em comum: confira se as duas exportações cobrem a mesma área e o nível.")
    r0, r1 = min(r for r, _ in comum), max(r for r, _ in comum)
    c0, c1 = min(c for _, c in comum), max(c for _, c in comum)
    print(f"tiles: antes {len(ta)}, depois {len(tb)}, em comum {len(comum)}")

    for pasta, tiles in (("A", ta), ("B", tb)):
        img = Image.new("RGB", ((c1 - c0 + 1) * TILE, (r1 - r0 + 1) * TILE))
        for r, c in comum:  # só cola onde os dois pacotes têm o tile, para manter o par alinhado
            img.paste(Image.open(io.BytesIO(tiles[(r, c)])).convert("RGB"), ((c - c0) * TILE, (r - r0) * TILE))
        Path(a.out, pasta).mkdir(parents=True, exist_ok=True)
        img.save(Path(a.out, pasta, "niteroi.png"))
    print("imagem:", img.size)

    res = 2 * ORIGIN / (TILE * 2 ** a.level)             # metros Mercator por pixel
    y_top, y_bot = ORIGIN - r0 * TILE * res, ORIGIN - (r1 + 1) * TILE * res
    lat = lambda y: math.degrees(2 * math.atan(math.exp(y / 6378137.0)) - math.pi / 2)
    lat_c = lat((y_top + y_bot) / 2)
    meta = {"nivel": a.level, "antes": a.antes_data, "depois": a.depois_data,
            "mpp": round(res * math.cos(math.radians(lat_c)), 4),  # metros reais por pixel
            "lat_norte": round(lat(y_top), 6), "lat_sul": round(lat(y_bot), 6),
            "lon_oeste": round(math.degrees((-ORIGIN + c0 * TILE * res) / 6378137.0), 6),
            "lon_leste": round(math.degrees((-ORIGIN + (c1 + 1) * TILE * res) / 6378137.0), 6)}
    json.dump(meta, open(Path(a.out, "meta.json"), "w"), indent=2)
    print(meta)


if __name__ == "__main__":
    main()
