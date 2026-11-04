"""Aplica o modelo treinado a pares de imagens de Niterói (capturas do Esri Wayback).

Entrada: data/niteroi/A/<nome>.png (antes) e data/niteroi/B/<nome>.png (depois),
         capturadas na mesma extensão, no mesmo zoom e com a mesma janela.
Uso:  python src/infer.py --weights results/best.pth
Saídas em results/niteroi/: máscara, sobreposição (depois + mudanças) e resumo.json.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train import build_model, stack_pair, to_array


@torch.no_grad()
def predict_full(model, a, b, dev, tile=256, thr=0.5):
    """Divide a imagem em tiles 256x256, prevê cada um e remonta a máscara."""
    h, w = a.shape[:2]
    pad = ((0, (-h) % tile), (0, (-w) % tile), (0, 0))
    a, b = np.pad(a, pad, mode="reflect"), np.pad(b, pad, mode="reflect")
    mask = np.zeros(a.shape[:2], bool)
    for r in range(0, a.shape[0], tile):
        for c in range(0, a.shape[1], tile):
            x = stack_pair(a[r:r + tile, c:c + tile], b[r:r + tile, c:c + tile])[None].to(dev)
            mask[r:r + tile, c:c + tile] = torch.sigmoid(model(x))[0, 0].cpu().numpy() > thr
    return mask[:h, :w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="results/best.pth")
    ap.add_argument("--dir", default="data/niteroi")
    ap.add_argument("--out", default="results/niteroi")
    ap.add_argument("--mpp", type=float, default=None, help="metros por pixel (padrão: lido de meta.json)")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    meta = Path(args.dir, "meta.json")
    if args.mpp is None:
        args.mpp = json.load(open(meta))["mpp"] if meta.exists() else 0.55
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(pretrained=False).to(dev)
    model.load_state_dict(torch.load(args.weights, map_location=dev))
    model.eval()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    resumo = {}
    for pa in sorted(Path(args.dir, "A").glob("*.png")):
        pb = Path(args.dir, "B", pa.name)
        if not pb.exists():
            print("sem par em B:", pa.name); continue
        ia, ib = Image.open(pa).convert("RGB"), Image.open(pb).convert("RGB")
        if ib.size != ia.size:
            print(f"aviso: {pa.name} tem tamanhos diferentes; redimensionando B")
            ib = ib.resize(ia.size)
        mask = predict_full(model, to_array(ia), to_array(ib), dev, thr=args.thr)

        Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"{pa.stem}_mascara.png")
        ov = np.asarray(ib).astype(np.float32)
        ov[mask] = 0.5 * ov[mask] + 0.5 * np.array([214, 51, 108], dtype=np.float32)
        Image.fromarray(ov.astype(np.uint8)).save(out / f"{pa.stem}_sobreposicao.png")
        resumo[pa.stem] = {"pixels_mudanca": int(mask.sum()),
                           "area_mudanca_m2": round(float(mask.sum()) * args.mpp ** 2, 1),
                           "percentual": round(100 * float(mask.mean()), 2)}
        print(pa.stem, resumo[pa.stem])
    json.dump(resumo, open(out / "resumo.json", "w"), indent=2)


if __name__ == "__main__":
    main()
