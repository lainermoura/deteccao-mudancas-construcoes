"""Treina e avalia uma U-Net (6 canais: antes + depois) no LEVIR-CD.

Dados: versão 256x256 do LEVIR-CD no Hugging Face (colunas imageA, imageB, label).
Uso:  python src/train.py --epochs 25 --batch 16 --lr 1e-4
Saídas: results/best.pth e results/metricas.json (hiperparâmetros + métricas no teste).
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

MEAN = np.array([0.485, 0.456, 0.406] * 2, dtype=np.float32)  # estatísticas do ImageNet
STD = np.array([0.229, 0.224, 0.225] * 2, dtype=np.float32)


def to_array(img):
    """Imagem PIL -> array float HxWx3 em [0, 1]."""
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def stack_pair(a, b):
    """Empilha antes e depois (HxWx3 cada) em um tensor normalizado 6xHxW."""
    x = (np.concatenate([a, b], axis=-1) - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def build_model(pretrained=True):
    """U-Net com encoder ResNet34; a 1ª convolução é adaptada para 6 canais."""
    return smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None,
                    in_channels=6, classes=1)


class LevirCD(Dataset):
    def __init__(self, split, augment=False):
        self.ds = load_dataset("ericyu/LEVIRCD_Cropped256", split=split)
        self.augment = augment

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        m = np.asarray(ex["label"])
        m = (m[..., 0] if m.ndim == 3 else m) > 0
        x = stack_pair(to_array(ex["imageA"]), to_array(ex["imageB"]))
        y = torch.from_numpy(m.astype(np.float32))[None]
        if self.augment:  # mesma transformação para as duas imagens e a máscara
            k = random.randint(0, 3)
            x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
            if random.random() < 0.5:
                x, y = x.flip(2), y.flip(2)
        return x, y


@torch.no_grad()
def evaluate(model, loader, dev, thr=0.5):
    """Precision, Recall, F1 e IoU da classe 'mudança'."""
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        p = torch.sigmoid(model(x.to(dev))).cpu() > thr
        t = y > 0.5
        tp += (p & t).sum().item()
        fp += (p & ~t).sum().item()
        fn += (~p & t).sum().item()
    pr, rc = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
    return dict(precision=pr, recall=rc, f1=2 * pr * rc / (pr + rc + 1e-9),
                iou=tp / (tp + fp + fn + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mk = lambda s, aug, sh: DataLoader(LevirCD(s, aug), batch_size=args.batch,
                                       shuffle=sh, num_workers=2)
    tr, va, te = mk("train", True, True), mk("val", False, False), mk("test", False, False)

    model = build_model().to(dev)
    dice = smp.losses.DiceLoss("binary")          # lida melhor com a classe rara
    bce = smp.losses.SoftBCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Path("results").mkdir(exist_ok=True)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            loss = dice(out, y) + bce(out, y)     # perda = Dice + BCE
            opt.zero_grad(); loss.backward(); opt.step()
        m = evaluate(model, va, dev)
        print(f"época {ep}: perda {loss.item():.3f} | val F1 {m['f1']:.3f} IoU {m['iou']:.3f}")
        if m["f1"] > best:                         # guarda o melhor modelo na validação
            best = m["f1"]
            torch.save(model.state_dict(), "results/best.pth")

    model.load_state_dict(torch.load("results/best.pth", map_location=dev))
    test = evaluate(model, te, dev)
    print("TESTE:", {k: round(v, 4) for k, v in test.items()})
    json.dump({"hiperparametros": vars(args), "val_f1": best, "teste": test},
              open("results/metricas.json", "w"), indent=2)


if __name__ == "__main__":
    main()
