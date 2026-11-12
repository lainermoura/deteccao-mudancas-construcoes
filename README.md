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
python src/train.py --robusto --nome robusto             # variante com variações de cor e nitidez (opcional)
python src/infer.py --weights results/best.pth
python src/infer.py --weights results/best.pth results/best_robusto.pth results/best_misto.pth   # união dos três modelos
```

No Colab: clone o repositório, ative a GPU e rode os comandos com `!` na frente.

## Como obter o par de imagens de Niterói (Esri Wayback Export)

1. No app **World Imagery Wayback** (Esri), vá até -22.9303, -43.0281 e escolha duas versões com **mudança local**, com vários anos de diferença (neste projeto: 2014-12-03 e 2026-04-30).
2. Para cada versão, use a opção de exportar (*Tile Export*), com o **mesmo enquadramento do mapa** nas duas, e baixe o pacote `.tpkx`. Os níveis 1 a 18 bastam; o nível 18 é o que as duas versões têm em comum (cerca de 0,55 m/pixel).
3. Converta os pacotes em um par de imagens alinhadas, usando só os tiles presentes nos dois:
   ```bash
   python src/tpkx_to_pairs.py antes.tpkx depois.tpkx --antes-data 2014-12-03 --depois-data 2026-04-30
   ```
   Isso gera `data/niteroi/A/niteroi.png`, `data/niteroi/B/niteroi.png` e `data/niteroi/meta.json` (com a resolução real e os limites geográficos da área).
4. Rode `python src/infer.py`.

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
