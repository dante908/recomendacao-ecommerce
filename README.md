# recomendacao-ecommerce

Projeto de portfolio para recomendacao de produtos com feedback implicito e similaridade item-item.

## O que o projeto faz
- Gera historico de interacoes implicitas (`view`, `cart`, `purchase`) por usuario e item.
- Treina recomendador item-item com similaridade cosseno.
- Avalia recomendacoes via HitRate@10 com split temporal por usuario.
- Salva recomendacoes top-10 por usuario e metricas do modelo.

## Estrutura de saida
- `data/interactions_synthetic.csv`
- `data/recommendations_top10.csv`
- `models/model_info.json`
- `models/metrics.json`
- `notebooks/analysis_notes.md`
- `reports/event_distribution.png`
- `reports/rank_coverage.png`
- `reports/top_recommended_items.png`

## Resultados atuais
- Modelo selecionado: **item_item_cosine**
- HitRate@10: **0.1333**
- MRR@10: **0.0531**
- Usuarios avaliados: **450**

## Instalacao minima
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Como reproduzir
```bash
python3 -m pip install -r requirements.txt
python3 src/main.py
```

## Execucao em lote (raiz do repositorio)
```bash
make run-all
```
