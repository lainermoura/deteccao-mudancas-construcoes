---
license: mit
library_name: pytorch
pipeline_tag: image-segmentation
language: pt
tags:
  - change-detection
  - remote-sensing
  - building-detection
  - unet
datasets:
  - ericyu/LEVIRCD_Cropped256
---

# Detecção de novas construções em imagens de satélite (U-Net bitemporal)

Modelos de detecção de mudanças em construções a partir de **duas imagens de satélite da mesma área em datas diferentes**. Projeto final do Future Tech Leaders Brasil (Spate Tech Bootcamp 2026), ligado ao ODS 11 (cidades sustentáveis).

- Código, relatório e página: https://github.com/lainermoura/deteccao-mudancas-construcoes
- Entrada: imagem "antes" e "depois" (RGB, 256 × 256), empilhadas em 6 canais.
- Saída: probabilidade de mudança (nova construção) por pixel.

## Arquivos

| Arquivo | Treino | F1 (teste LEVIR-CD) | IoU |
|---|---|---|---|
| `best.pth` | original (aumento de dados geométrico) | 0,8761 | 0,7794 |
| `best_robusto.pth` | variações de cor e nitidez em todos os exemplos | 0,8709 | 0,7714 |
| `best_misto.pth` | variações de cor e nitidez em metade dos exemplos | 0,8760 | 0,7794 |

Arquitetura: U-Net com encoder ResNet34 (pré-treinado no ImageNet), 6 canais de entrada, Adam, learning rate 1e-4, batch 16, 25 épocas, perda Dice + BCE. Dados: LEVIR-CD, versão 256 × 256 (7.120 pares de treino, 1.024 de validação e 2.048 de teste).

## Como usar

```python
import numpy as np, torch
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download

pesos = hf_hub_download("lainermoura/ftl-deteccao-mudancas-unet", "best.pth")
model = smp.Unet("resnet34", encoder_weights=None, in_channels=6, classes=1)
model.load_state_dict(torch.load(pesos, map_location="cpu")); model.eval()

MEAN = np.array([0.485, 0.456, 0.406] * 2, dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225] * 2, dtype=np.float32)

def prever(antes, depois, limiar=0.5):
    """antes e depois: arrays 256x256x3 com valores em [0, 1]. Retorna a máscara booleana."""
    x = (np.concatenate([antes, depois], axis=-1) - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1).copy())[None]
    with torch.no_grad():
        return torch.sigmoid(model(x))[0, 0].numpy() > limiar
```

Para imagens maiores e para combinar os três modelos (união das probabilidades), use `src/infer.py` do repositório.

## Desempenho em Niterói (estudo de caso)

Aplicados sem ajuste fino a dois pares de imagens do Esri World Imagery Wayback, com conferência visual de cada região marcada pela união dos três modelos:

| Área | Regiões marcadas | Construções novas | Falsos alarmes |
|---|---|---|---|
| B (2020 → 2026, imagens de boa qualidade) | 12 | 10 | 2 |
| A (2014 → 2026, imagem de 2014 mais pixelada) | 16 | 6 | 10 |

## Limitações

- Treinados com imagens dos EUA; o desempenho em outras regiões depende muito do par de imagens.
- Sensíveis a diferenças de nitidez, cor e exposição entre as datas, que podem gerar falsos alarmes.
- O modelo original ignora telhados cinzas e o robusto perde parte dos de cor de tijolo; a união dos três reduz o problema.
- A avaliação em Niterói foi visual, em duas áreas pequenas; a revocação não foi medida de forma independente.

## Dados e licenças

Os pesos e o código seguem a licença MIT. O LEVIR-CD é um conjunto de terceiros (Chen e Shi, 2020), sujeito aos termos dos autores. As imagens de Niterói pertencem à Esri e a seus provedores e não são distribuídas.

## Referências

- Chen, H.; Shi, Z. (2020). A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection. *Remote Sensing*, 12(10), 1662.
- Daudt, R. C.; Le Saux, B.; Boulch, A. (2018). Fully Convolutional Siamese Networks for Change Detection. *ICIP*.
- He, K.; Zhang, X.; Ren, S.; Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Ronneberger, O.; Fischer, P.; Brox, T. (2015). U-Net. *MICCAI*.
