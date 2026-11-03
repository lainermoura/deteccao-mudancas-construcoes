# Detecção de mudanças em construções com imagens de satélite (Niterói, RJ)

Projeto final (Capstone) do **Future Tech Leaders Brasil: Spate Tech Bootcamp 2026**.

## Objetivo

Identificar **novas construções** comparando duas imagens de satélite da mesma área em datas diferentes (ODS 11, cidades sustentáveis). O modelo é treinado no benchmark público **LEVIR-CD** e aplicado, como estudo de caso, a imagens de **Niterói** (centro da área: -22.9303, -43.0281) obtidas do **Esri World Imagery Wayback**.

## Estrutura

```
src/train.py      # dados, U-Net (6 canais), treino e avaliação no LEVIR-CD
src/infer.py      # aplica o modelo aos pares de Niterói e gera máscaras/sobreposições
data/niteroi/     # A/ (antes) e B/ (depois): NÃO versionado (termos de uso da Esri)
results/          # metricas.json, best.pth (ignorado no git) e resultados de Niterói
docs/index.html   # página do GitHub Pages
```

## Instalação e execução

```bash
pip install -r requirements.txt
python src/train.py --epochs 25 --batch 16 --lr 1e-4   # use GPU (Colab/Kaggle)
python src/infer.py --weights results/best.pth --mpp 0.55
```

No Colab: clone o repositório, ative a GPU e rode os comandos com `!` na frente.

## Como obter o par de imagens de Niterói (Esri Wayback)

1. Abra o app **World Imagery Wayback** (Esri) e vá até -22.9303, -43.0281.
2. Escolha duas versões que tenham **mudança local** na área, com 2 a 3 anos ou mais de diferença. Anote as datas.
3. Use **zoom 18** (cerca de 0,55 m/pixel, escala próxima à do LEVIR-CD). Mantenha o zoom e o tamanho da janela do navegador **idênticos** nas duas datas.
4. Capture a mesma extensão nas duas datas, com o mesmo nome de arquivo, e salve em `data/niteroi/A/` (antes) e `data/niteroi/B/` (depois). Use PNG.
5. Rode `python src/infer.py`. Se usar zoom diferente de 18, ajuste `--mpp`.

**Termos de uso:** as imagens pertencem à Esri e seus provedores e estão sujeitas aos termos da Esri; a camada padrão não se destina à exportação para uso offline. Confira os termos antes de usar, não publique as imagens originais neste repositório e cite a fonte (Esri World Imagery Wayback e as datas das versões).

## Reproduzindo os resultados

`results/metricas.json` guarda os hiperparâmetros e as métricas (Precision, Recall, F1 e IoU) no conjunto de teste do LEVIR-CD. As previsões de Niterói ficam em `results/niteroi/` e não têm gabarito: a avaliação é visual, ou quantitativa em tiles rotulados manualmente.

## Limitações

O modelo foi treinado com imagens dos EUA; telhados, densidade e relevo de Niterói diferem. Sombras, vegetação sazonal e diferenças de ângulo geram falsos positivos. Capturas de tela não são georreferenciadas, então as áreas em m² são aproximadas.

## Referências

- Chen, H.; Shi, Z. (2020). A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection. *Remote Sensing*, 12(10), 1662.
- Daudt, R. C.; Le Saux, B.; Boulch, A. (2018). Fully Convolutional Siamese Networks for Change Detection. *ICIP*.
- Ronneberger, O.; Fischer, P.; Brox, T. (2015). U-Net. *MICCAI*.

Licença: MIT.
